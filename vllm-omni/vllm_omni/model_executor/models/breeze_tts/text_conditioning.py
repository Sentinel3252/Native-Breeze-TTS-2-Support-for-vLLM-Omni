# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent text-segment encoding and Breeze layer feature fusion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .configuration_breeze_tts import config_dict
from .t5gemma2_encoder import T5Gemma2TextConfig, T5Gemma2TextEncoder


def speech_prompt_ids(tokenizer: Any, text: str, instructions: str | None = None) -> list[int]:
    """Match the official [S0] and instruction template including specials."""
    segment = "[S0]" + (f"<ins_bos>{instructions}<ins_eos>" if instructions else "") + text
    first = tokenizer.encode(segment, add_special_tokens=True)
    rendered = tokenizer.decode(first, skip_special_tokens=False)
    ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    if not ids or min(ids) < 0:
        raise ValueError("Breeze tokenizer produced an empty or invalid speech prompt")
    return ids


class TextConditioning(nn.Module):
    """Checkpoint-named encoder/projections; segments never attend to each other."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.text_encoder = None
        self.text_encoder_proj = None
        self.text_encoder_layer_projs = None
        self.feature_indices = getattr(config, "text_encoder_feature_layer_idx", -1)
        if isinstance(self.feature_indices, int):
            self.feature_indices = (self.feature_indices,)
        else:
            self.feature_indices = tuple(self.feature_indices)
        if not self.feature_indices:
            raise ValueError("text_encoder_feature_layer_idx cannot be empty")
        self.fuse_first = bool(getattr(config, "text_encoder_dimfusion_fuse_first_layer", False))
        if config.text_encoder_config is None:
            if self.fuse_first:
                raise ValueError("DimFusion requires a text encoder")
            return
        raw = config_dict(config.text_encoder_config)
        model_type = raw.pop("model_type")
        if model_type == "t5gemma2_text":
            encoder_config = T5Gemma2TextConfig(**raw)
            encoder_config._attn_implementation = "sdpa"
            self.text_encoder = T5Gemma2TextEncoder(encoder_config)
        elif model_type == "t5_gemma_module":
            from transformers.models.t5gemma.configuration_t5gemma import T5GemmaModuleConfig
            from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

            encoder_config = T5GemmaModuleConfig(**raw)
            encoder_config._attn_implementation = "sdpa"
            self.text_encoder = T5GemmaEncoder(encoder_config)
        else:
            encoder_config = AutoConfig.for_model(model_type, **raw)
            encoder_config._attn_implementation = "sdpa"
            self.text_encoder = AutoModel.from_config(encoder_config)
        encoder_size = encoder_config.hidden_size
        projection = getattr(config, "text_encoder_proj_type", "linear")
        feature_size = encoder_size * len(self.feature_indices)
        if projection == "linear":
            self.text_encoder_proj = nn.Linear(feature_size, config.hidden_size, bias=False)
        elif projection == "mlp":
            self.text_encoder_proj = nn.Sequential(
                nn.Linear(feature_size, config.hidden_size * 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False),
            )
        elif projection == "breeze_dimfusion":
            if config.hidden_size % 2:
                raise ValueError("DimFusion requires an even backbone hidden_size")
            self.text_encoder_proj = nn.Linear(
                feature_size, config.hidden_size // 2 if self.fuse_first else config.hidden_size, bias=True
            )
            self.text_encoder_layer_projs = nn.ModuleList(
                nn.Linear(encoder_size, config.hidden_size // 2, bias=False) for _ in range(config.num_hidden_layers)
            )
        else:
            raise ValueError(f"Unsupported Breeze text projection: {projection}")
        if self.fuse_first and self.text_encoder_layer_projs is None:
            raise ValueError("First-layer fusion requires breeze_dimfusion")
        self.text_encoder.eval()

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, segment_lengths: list[int]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        ids = ids.reshape(-1).long()
        if not segment_lengths or any(length <= 0 for length in segment_lengths) or sum(segment_lengths) != ids.numel():
            raise ValueError("Text segment lengths must cover the complete prompt exactly")
        vocab = (
            self.embed_text_tokens.num_embeddings if self.text_encoder is None else self.text_encoder.config.vocab_size
        )
        if bool(((ids < 0) | (ids >= vocab)).any()):
            raise ValueError("Text tokens exceed the checkpoint's text vocabulary")
        if self.text_encoder is None:
            return self.embed_text_tokens(ids), []
        projected = []
        layer_parts: list[list[torch.Tensor]] = [[] for _ in range(self.config.num_hidden_layers)]
        needs_hidden = self.text_encoder_layer_projs is not None or self.feature_indices != (-1,)
        for segment in ids.split(segment_lengths):
            # Independent calls also reset position IDs for every segment.
            result = self.text_encoder(
                input_ids=segment.unsqueeze(0),
                attention_mask=torch.ones_like(segment).unsqueeze(0),
                position_ids=torch.arange(segment.numel(), device=ids.device).unsqueeze(0),
                output_hidden_states=needs_hidden,
                return_dict=True,
            )
            features = (
                result.last_hidden_state
                if self.feature_indices == (-1,)
                else torch.cat([result.hidden_states[index] for index in self.feature_indices], dim=-1)
            )
            projected.append(self.text_encoder_proj(features).squeeze(0))
            if self.text_encoder_layer_projs is not None:
                selected = list(result.hidden_states)[
                    getattr(self.config, "text_encoder_dimfusion_layer_start_idx", 1) : getattr(
                        self.config, "text_encoder_dimfusion_layer_end_idx", None
                    )
                ]
                if not selected:
                    raise ValueError("DimFusion selected no text encoder layers")
                count = self.config.num_hidden_layers
                selected = (selected + [selected[-1]] * max(0, count - len(selected)))[-count:]
                for index, projection in enumerate(self.text_encoder_layer_projs):
                    layer_parts[index].append(projection(selected[index]).squeeze(0))
        embeds = torch.cat(projected)
        layers = [torch.cat(parts) for parts in layer_parts] if self.text_encoder_layer_projs is not None else []
        if self.fuse_first:
            embeds = torch.cat((embeds, layers[0]), dim=-1)
        return embeds, layers


def fuse_text_features(hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the second half only at text positions, before normalization."""
    if (
        mask.dtype != torch.bool
        or mask.numel() != hidden.shape[0]
        or features.shape != (int(mask.sum()), hidden.shape[-1] // 2)
    ):
        raise ValueError("DimFusion features do not match the text positions")
    hidden = hidden.clone()
    hidden[mask, hidden.shape[-1] // 2 :] = features.to(hidden)
    return hidden
