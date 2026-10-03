# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""History-preserving Mimi decode; correctness-first prefix replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CodecHistory:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted: torch.Tensor | None = None


class MimiPrefixDecoder:
    """Replay the entire prefix, including transformer and convolution context.

    Mimi.decode's transformer cache alone does not preserve the upsampler and
    decoder convolution history. This implementation trades throughput for
    continuity. The held-back tail is released by an explicit terminal chunk.
    """

    def __init__(
        self,
        model: Any,
        num_codebooks: int,
        *,
        streaming: bool = False,
        holdback_frames: int = 2,
        max_frames: int = 4096,
    ) -> None:
        if holdback_frames < 0 or max_frames <= 0:
            raise ValueError("Invalid Mimi stream limits")
        if streaming and not getattr(model.config, "use_causal_conv", True):
            raise ValueError("Mimi streaming requires causal decoder convolutions")
        self.model = model
        self.num_codebooks = num_codebooks
        self.streaming = streaming
        self.holdback_frames = holdback_frames
        self.max_frames = max_frames
        self.states: dict[str, CodecHistory] = {}

    def discard(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.states.pop(str(request_id), None)

    @torch.inference_mode()
    def push(self, request_id: str, frames: torch.Tensor, *, finished: bool = False) -> torch.Tensor:
        if not request_id or frames.ndim != 2 or frames.shape[1] != self.num_codebooks:
            self.states.pop(request_id, None)
            raise ValueError("Mimi requires an explicit request ID and [frames, codebooks] codes")
        if frames.is_floating_point() or bool(((frames < 0) | (frames >= self.model.config.codebook_size)).any()):
            self.states.pop(request_id, None)
            raise ValueError("Mimi received an invalid or reserved codec token")
        state = self.states.setdefault(request_id, CodecHistory())
        try:
            if self.streaming and not getattr(self.model.config, "use_causal_conv", True):
                raise ValueError("Mimi streaming requires causal decoder convolutions")
            count = sum(chunk.shape[0] for chunk in state.frames) + frames.shape[0]
            if count > self.max_frames:
                raise ValueError("Mimi request exceeds max_frames")
            if frames.numel():
                state.frames.append(frames.detach().clone())
            if not state.frames or (not self.streaming and not finished):
                return torch.empty(0, dtype=torch.float32)
            codes = torch.cat(state.frames).transpose(0, 1).unsqueeze(0)
            output = self.model.decode(codes, return_dict=True)
            audio = output.audio_values.reshape(-1).float().cpu()
            if not bool(torch.isfinite(audio).all()):
                raise ValueError("Mimi produced non-finite audio")
            emitted = state.emitted.numel() if state.emitted is not None else 0
            if emitted and (
                audio.numel() < emitted or not torch.allclose(audio[:emitted], state.emitted, atol=2e-3, rtol=2e-3)
            ):
                raise RuntimeError("Mimi prefix changed after emission; disable codec_streaming for this checkpoint")
            samples_per_frame = self.model.config.sampling_rate / self.model.config.frame_rate
            end = (
                audio.numel()
                if finished
                else max(emitted, audio.numel() - int(self.holdback_frames * samples_per_frame))
            )
            delta = audio[emitted:end].clone()
            state.emitted = audio[:end].clone()
            return delta
        except Exception:
            self.states.pop(request_id, None)
            raise
        finally:
            if finished:
                self.states.pop(request_id, None)
