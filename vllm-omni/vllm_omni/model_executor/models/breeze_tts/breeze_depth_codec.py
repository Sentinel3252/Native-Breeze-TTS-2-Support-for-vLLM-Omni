# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mimi waveform stage. Historical class/stage names remain compatible."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import MimiModel
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .codec_stream import MimiPrefixDecoder
from .weight_coverage import validate_weight_coverage


class BreezeDepthCodecDecoder(nn.Module):
    """Decode complete RVQ frames with history isolated by request ID."""

    requires_request_ids = True
    requires_raw_input_tokens = True
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.codec_model = MimiModel(self.config.codec_config).eval()
        self.sample_rate = int(self.config.codec_config.sampling_rate)
        # Non-streaming is the safe default. Streaming is explicitly enabled
        # in request metadata by the connector configuration.
        self._decoder = MimiPrefixDecoder(self.codec_model, self.config.num_codebooks)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.numel(), 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        request_ids: list[str] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        info = model_intermediate_buffer if model_intermediate_buffer is not None else runtime_additional_information
        if request_ids is None and info is None:
            # Runner memory profiling; never create synthetic request state.
            return OmniOutput(
                torch.empty((input_ids.numel(), 0), device=input_ids.device),
                {"audio": [torch.empty(0)], "sr": [torch.tensor(self.sample_rate)]},
            )
        if not request_ids or info is None or len(request_ids) != len(info):
            raise ValueError("Breeze Mimi requires explicit, aligned request IDs and payloads")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Duplicate request IDs in a Mimi batch")
        outputs = []
        for request_id, payload in zip(request_ids, info, strict=True):
            codes = payload.get("codes", {}).get("audio")
            if not isinstance(codes, torch.Tensor):
                raise ValueError("Breeze Mimi payload must contain complete codec frames")
            if codes.numel() == 0:
                codes = torch.empty((0, self.config.num_codebooks), device=input_ids.device, dtype=torch.long)
            meta = payload.get("meta", {})
            finished = bool(meta.get("stream_finished", meta.get("finished", True)))
            self._decoder.streaming = bool(meta.get("codec_streaming", False))
            outputs.append(self._decoder.push(str(request_id), codes.to(input_ids.device), finished=finished))
        # Lists are partitioned by the generation runner, preserving batch IDs.
        return OmniOutput(
            torch.empty((input_ids.numel(), 0), device=input_ids.device),
            {"audio": outputs, "sr": [torch.tensor(self.sample_rate) for _ in request_ids]},
        )

    def on_requests_finished(self, request_ids: list[str], **_: Any) -> None:
        self._decoder.discard(list(request_ids))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(
            (name, tensor) for name, tensor in weights if name.startswith("codec_model.")
        )
        # Mimi quantizer centroids/statistics are persistent buffers, and are
        # just as important as parameters for checkpoint coverage.
        validate_weight_coverage(self, loaded, loaded, persistent_buffers=True)
        return loaded


__all__ = ["BreezeDepthCodecDecoder"]
