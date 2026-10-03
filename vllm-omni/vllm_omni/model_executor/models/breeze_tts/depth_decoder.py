# Breeze model implementation adapted from Hugging Face Transformers.
#
# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_breeze_base import BreezeConfig, BreezeDepthDecoderConfig

logger = logging.get_logger(__name__)


def depth_causal_mask(inputs_embeds, cache_position, attention_mask=None):
    """Small-frame causal mask, independent of Transformers 4/5 APIs."""
    key_length = int(cache_position[-1]) + 1
    keys = torch.arange(key_length, device=inputs_embeds.device)
    allowed = keys.unsqueeze(0) <= cache_position.unsqueeze(1)
    mask = torch.zeros(allowed.shape, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    mask.masked_fill_(~allowed, torch.finfo(inputs_embeds.dtype).min)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[-1] != key_length:
            raise ValueError("Depth attention mask must cover the full frame prefix")
        mask = mask.expand(attention_mask.shape[0], -1, -1, -1).clone()
        mask.masked_fill_(~attention_mask[:, None, None, :].bool(), torch.finfo(inputs_embeds.dtype).min)
    return mask


@dataclass
class BreezeOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    depth_decoder_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction) of the depth decoder model.
    depth_decoder_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the depth decoder (scores for each vocabulary token before SoftMax).
    depth_decoder_past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    depth_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    depth_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
    backbone_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction) of the backbone model.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    depth_decoder_loss: torch.FloatTensor | None = None
    depth_decoder_logits: torch.FloatTensor = None
    depth_decoder_past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    depth_decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    depth_decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    backbone_loss: torch.FloatTensor | None = None


class BreezeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BreezeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BreezeRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: BreezeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default":
            self.rope_init_fn = self._default_rope
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def _default_rope(config, device=None, seq_len=None):
        # Transformers 5 removed the default entry from ROPE_INIT_FUNCTIONS.
        # Keep the exact reference formula, independent of that registry.
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        return inv_freq, 1.0

    compute_default_rope_parameters = _default_rope

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BreezeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BreezeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BreezeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BreezeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BreezeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = BreezeAttention(config=config, layer_idx=layer_idx)

        self.mlp = BreezeMLP(config)
        self.input_layernorm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class BreezePreTrainedModel(PreTrainedModel):
    config_class = BreezeConfig
    config: BreezeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BreezeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # does not because of Mimi codec model
    # _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BreezeDecoderLayer,
        "attentions": BreezeAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, BreezeCodebooksHead):
            num_codebooks = module.num_codebooks
            for i in range(num_codebooks - 1):
                module.weight.data[i].normal_(mean=0.0, std=self.config.initializer_range)


class BreezeDepthDecoderModel(BreezePreTrainedModel):
    config: BreezeDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if hasattr(config, "audio_embed_size") and config.audio_embed_size:
            self.audio_embed_size = config.audio_embed_size
        else:
            self.audio_embed_size = config.backbone_hidden_size
        self.embed_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), self.audio_embed_size)
        self.layers = nn.ModuleList(
            [BreezeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BreezeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BreezeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.inputs_embeds_projector = nn.Linear(self.audio_embed_size, config.hidden_size, bias=False)

        if config.backbone_hidden_size != self.audio_embed_size:
            self.backbone_hidden_state_projector = nn.Linear(
                config.backbone_hidden_size, self.audio_embed_size, bias=False
            )
        else:
            self.backbone_hidden_state_projector = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*):
            The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
            is provided in the `input_ids` argument.
        """
        if position_ids is not None and not torch.compiler.is_compiling():
            logger.warning_once(
                "Custom `position_ids` were provided but will be ignored. Breeze depth decoder automatically determines position_ids "
                "from `cache_position` and as it requires them to be identical across the batch, the provided position_ids will be ignored."
            )
            position_ids = None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)

        if inputs_embeds is None:
            codebook_idxs = torch.clamp(cache_position - 1, min=0)
            offset = codebook_idxs * self.vocab_size
            inputs_embeds = self.embed_tokens(input_ids + offset)

            input_ids_are_first_codebook = cache_position[0] == 0
            if backbone_last_hidden_state is not None:
                if self.backbone_hidden_state_projector is not None:
                    inputs_embeds[:, 0] = self.backbone_hidden_state_projector(backbone_last_hidden_state)
                else:
                    inputs_embeds[:, 0] = backbone_last_hidden_state
            else:
                if not torch.compiler.is_compiling() and input_ids_are_first_codebook:
                    logger.warning(
                        "When the first codebook token is provided, `backbone_last_hidden_state` should also be provided for correct inference."
                    )

        inputs_embeds = self.inputs_embeds_projector(inputs_embeds)

        causal_mask = depth_causal_mask(inputs_embeds, cache_position, attention_mask)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class BreezeCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(torch.empty(self.num_codebooks - 1, hidden_size, vocab_size))

    def forward(self, hidden_states, cache_position=None):
        if cache_position is None:
            seq_length = hidden_states.shape[1]
            codebook_weight = self.weight[torch.arange(seq_length)]
        else:
            codebook_idxs = cache_position - 1
            codebook_weight = self.weight[codebook_idxs]

        hidden_states = [
            nn.functional.linear(hidden_states[:, codebook_idx, :], codebook_weight[codebook_idx].T)
            for codebook_idx in range(codebook_weight.shape[0])
        ]
        hidden_states = torch.stack(hidden_states, dim=1)

        return hidden_states

    # 重写 extra_repr 方法以显示参数细节
    def extra_repr(self):
        return f"weight_shape={list(self.weight.shape)}, num_codebooks={self.num_codebooks}"


class BreezeDepthDecoderForCausalLM(BreezePreTrainedModel, GenerationMixin):
    _tied_weights_keys = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, config):
        super().__init__(config)
        self.model = BreezeDepthDecoderModel(config)
        self.vocab_size = config.vocab_size
        self.codebooks_head = BreezeCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)

        # Compute per-codebook loss weights
        # The depth decoder predicts num_codebooks-1 codebooks (31 for default config)
        # We distribute weights by prepending a virtual codebook to make division even
        codebook_loss_weights_list = getattr(config, "codebook_loss_weights", [1.0])
        num_depth_codebooks = config.num_codebooks - 1  # 31 for default 32 codebooks
        num_weights = len(codebook_loss_weights_list)
        num_codebooks_with_virtual = num_depth_codebooks + 1
        codebooks_per_weight = num_codebooks_with_virtual // num_weights
        weights_expanded = []
        for weight in codebook_loss_weights_list:
            weights_expanded.extend([weight] * codebooks_per_weight)
        weights_expanded = weights_expanded[1:]
        self.codebook_loss_weights = weights_expanded

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*):
            The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
            is provided in the `input_ids` argument.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=backbone_last_hidden_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        loss = None
        logits = None

        logits = self.codebooks_head(
            hidden_states[:, slice_indices, :],
            cache_position[slice_indices] if cache_position is not None else None,
        )
        logits = logits.contiguous()

        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_function(
                logits=logits,
                labels=None,
                vocab_size=self.config.vocab_size,
                shift_labels=shift_labels,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            **kwargs,
        )

        is_first_generation_step = model_inputs["cache_position"][0] == 0
        if not is_first_generation_step:
            model_inputs.pop("backbone_last_hidden_state")

        # breeze depth decoder does not use position_ids
        model_inputs.pop("position_ids")

        return model_inputs
