# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paged-attention Breeze AR with same-step complete-frame feedback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .depth_decoder import BreezeDepthDecoderForCausalLM
from .frame_decoder import DepthSampling, complete_frame, embed_audio_frame
from .text_conditioning import TextConditioning, fuse_text_features
from .weight_coverage import validate_weight_coverage


def _scalar(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item() if value.numel() else default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class BreezeTTSTalkerForConditionalGeneration(nn.Module):
    """The scheduler samples c0; the post-sampling hook completes that frame."""

    post_sample_all_requests = True
    post_sample_drop_negative_frames = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "backbone_model.layers.": "model.layers.",
            "backbone_model.norm.": "model.norm.",
            "backbone_model.embed_tokens.embed_audio_tokens.": "audio_embedding.",
            "embed_text_tokens.": "conditioning.embed_text_tokens.",
            "text_encoder.": "conditioning.text_encoder.",
            "text_encoder_proj.": "conditioning.text_encoder_proj.",
            "text_encoder_layer_projs.": "conditioning.text_encoder_layer_projs.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Breeze currently requires tensor/pipeline parallel size 1")
        if vllm_config.quant_config is not None or vllm_config.speculative_config is not None:
            raise ValueError("Breeze quantization and speculative decoding are not implemented")
        if not vllm_config.model_config.enforce_eager or vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Breeze requires enforce_eager=True and enable_prefix_caching=False")
        if getattr(vllm_config.scheduler_config, "async_scheduling", False):
            raise ValueError("Breeze requires async_scheduling=False for same-step depth feedback")
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.omni_pooler_payload_include_hidden = False
        self.use_async_omni_output = False
        backbone_cls = Qwen3Model if self.config.backbone_model_type == "qwen3" else LlamaModel
        self.model = backbone_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # The checkpoint uses RVQ embeddings, not the native LM token table.
        self.model.embed_tokens = nn.Identity()
        self.conditioning = TextConditioning(self.config)
        self.audio_embedding = nn.Embedding(self.config.num_codebooks * self.config.vocab_size, self.config.hidden_size)
        self.depth_decoder = BreezeDepthDecoderForCausalLM(self.config.depth_decoder_config)
        if self.config.tie_codebooks_embeddings:
            if self.depth_decoder.model.embed_tokens.weight.shape != self.audio_embedding.weight.shape:
                raise ValueError("Tied depth/backbone embeddings have incompatible shapes")
            self.depth_decoder.model.embed_tokens = self.audio_embedding
        self.lm_head = ParallelLMHead(
            self.config.vocab_size + 1, self.config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size + 1)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = {}
        self._depth_generators: dict[str, torch.Generator] = {}
        self._frame_history: dict[str, list[torch.Tensor]] = {}
        allowed = torch.zeros(self.config.vocab_size + 1, dtype=torch.bool)
        allowed[: self.config.codec_codebook_size] = True
        allowed[self.config.backbone_eos_token_id] = True
        self.register_buffer("_invalid_codec_logits", ~allowed, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Called before request metadata is available. preprocess supplies the
        # actual embeddings for both text prefill and generated audio frames.
        return torch.zeros(
            (input_ids.numel(), self.config.hidden_size),
            device=input_ids.device,
            dtype=self.audio_embedding.weight.dtype,
        )

    @torch.inference_mode()
    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None = None, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        request_id = info.get("request_id")
        if not request_id:
            raise ValueError("Breeze preprocessing requires a stable request_id")
        prompt_len = int(info["_omni_prompt_len"])
        start = int(info["_omni_num_computed_tokens"])
        end = start + input_ids.numel()
        pieces = []
        layer_features = []
        mask = torch.zeros(input_ids.numel(), device=input_ids.device, dtype=torch.bool)
        if start < prompt_len:
            if request_id not in self._prompt_cache:
                full_ids = torch.as_tensor(
                    info.get("breeze_prompt_ids", []), device=input_ids.device, dtype=torch.long
                ).reshape(-1)
                if full_ids.numel() != int(info["_omni_prompt_len"]):
                    raise ValueError("Breeze requires the complete prompt IDs even during chunked prefill")
                lengths = [int(value) for value in info.get("breeze_text_segment_lengths", [full_ids.numel()])]
                embeds, features = self.conditioning(full_ids, lengths)
                self._prompt_cache[request_id] = (full_ids, embeds, features)
            full_ids, embeds, features = self._prompt_cache[request_id]
            text_end = min(end, prompt_len)
            text_count = text_end - start
            if bool((input_ids[:text_count] != 0).any()):
                raise ValueError(
                    "Breeze text prefill requires scheduling placeholders matching the conditioning prompt length"
                )
            pieces.append(embeds[start:text_end])
            layer_features = [layer[start:text_end] for layer in features]
            mask[:text_count] = True
            if end >= prompt_len:
                self._prompt_cache.pop(request_id, None)
        if end > prompt_len:
            # KV-cache preemption can replay several old audio inputs in a
            # prefill span. Keep their complete frames; never resample depth.
            audio_start = max(start, prompt_len) - prompt_len
            audio_end = end - prompt_len
            history = self._frame_history.get(request_id, [])
            if audio_end > len(history):
                raise ValueError("Breeze decode requires complete audio frame history")
            codes = torch.cat(history[audio_start:audio_end]).to(input_ids.device)
            audio_ids = input_ids[max(prompt_len - start, 0) :].long()
            if not torch.equal(codes[:, 0], audio_ids):
                raise ValueError("Breeze audio feedback is out of step with the AR sampler")
            pieces.append(
                embed_audio_frame(self.audio_embedding, codes, self.config.vocab_size, self.config.num_codebooks)
            )
        return input_ids, torch.cat(pieces), {"breeze_layer_features": layer_features, "breeze_text_mask": mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("Breeze pipeline-parallel tensors are unsupported")
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        spans = kwargs.get("request_token_spans") or []
        if infos and len(infos) != len(spans):
            raise ValueError("Breeze request metadata does not match the scheduled spans")
        mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        feature_parts: list[list[torch.Tensor]] = [[] for _ in self.model.layers]
        for info, (start, end) in zip(infos, spans, strict=True):
            features = info.get("breeze_layer_features", [])
            if not features:
                continue
            if len(features) != len(feature_parts):
                raise ValueError("Breeze text features do not cover every backbone layer")
            text_mask = info["breeze_text_mask"].to(hidden.device)
            mask[int(start) : int(end)] = text_mask
            for index, feature in enumerate(features):
                if feature.shape[0] != int(text_mask.sum()):
                    raise ValueError("Breeze text layer feature count does not match its scheduled positions")
                feature_parts[index].append(feature.to(hidden.device))
        residual = None
        for index, layer in enumerate(self.model.layers):
            if index > 0 and feature_parts[index]:
                # Native layers defer residual addition. Materialize it before
                # replacing text dimensions, then normalize the fused activation.
                if residual is not None:
                    hidden = hidden + residual
                    residual = None
                hidden = fuse_text_features(hidden, torch.cat(feature_parts[index]), mask)
            hidden, residual = layer(positions, hidden, residual)
        return self.model.norm(hidden, residual)[0] if residual is not None else self.model.norm(hidden)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        logits = self.logits_processor(self.lm_head, hidden_states)
        return None if logits is None else logits.masked_fill(self._invalid_codec_logits, float("-inf"))

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **_: Any) -> OmniOutput:
        return (
            model_outputs
            if isinstance(model_outputs, OmniOutput)
            else OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})
        )

    @torch.inference_mode()
    def post_sample_talker_mtp(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        req_ids: list[str],
        req_infos: list[dict[str, Any]],
    ) -> torch.Tensor:
        frames = torch.full((len(req_ids), self.config.num_codebooks), -1, dtype=torch.long, device=input_ids.device)
        for index, (request_id, info) in enumerate(zip(req_ids, req_infos, strict=True)):
            token = input_ids[index : index + 1]
            if int(token) == self.config.backbone_eos_token_id:
                continue
            settings = DepthSampling(
                float(_scalar(info.get("breeze_depth_temperature"), 0.9)),
                int(_scalar(info.get("breeze_depth_top_k"), 50)),
                float(_scalar(info.get("breeze_depth_top_p"), 1.0)),
            )
            generator = self._depth_generators.get(request_id)
            if generator is None:
                generator = torch.Generator(device=input_ids.device)
                seed = _scalar(info.get("_omni_seed"), None)
                generator.seed() if seed is None else generator.manual_seed(int(seed))
                self._depth_generators[request_id] = generator
            frames[index : index + 1] = complete_frame(
                self.depth_decoder,
                token,
                hidden_states[index : index + 1],
                self.config.codec_codebook_size,
                settings,
                generator,
            )
            history = self._frame_history.setdefault(request_id, [])
            if len(history) >= 4096:
                self.on_requests_finished([request_id])
                raise ValueError("Breeze request exceeds the supported 4096 audio frames")
            history.append(frames[index : index + 1].detach().cpu().clone())
        return frames

    def on_requests_finished(self, request_ids: list[str], **_: Any) -> None:
        for request_id in request_ids:
            self._prompt_cache.pop(request_id, None)
            self._depth_generators.pop(request_id, None)
            self._frame_history.pop(request_id, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Codec belongs to stage 1. Every parameter of this stage must be loaded.
        source_names = []

        def selected_weights():
            for name, tensor in weights:
                if not name.startswith("codec_model."):
                    source_names.extend(self.hf_to_vllm_mapper.apply_list([name]))
                    yield name, tensor

        selected = selected_weights()
        loaded = AutoWeightsLoader(self).load_weights(selected, mapper=self.hf_to_vllm_mapper)
        validate_weight_coverage(self, loaded, source_names)
        return loaded


__all__ = ["BreezeTTSTalkerForConditionalGeneration"]
