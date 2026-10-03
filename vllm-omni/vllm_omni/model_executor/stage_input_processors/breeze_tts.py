# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete-frame transport from Breeze AR/depth to Mimi."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch


def _frames(payload: Any) -> torch.Tensor | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("codes.audio")
    if value is None:
        value = payload.get("codes", {}).get("audio")
    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.numel() == 0 and value.ndim != 2:
        # Runner partial-prefill rows use a 1-D empty suppression marker.
        return None
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.is_floating_point():
        raise ValueError("Breeze transport requires integer [frames, codebooks] codes")
    if bool((value < 0).any()):
        raise ValueError("EOS/negative padding must be removed before Breeze transport")
    return value.detach().to("cpu", torch.long).contiguous()


def _payload(frames: torch.Tensor, *, finished: bool, streaming: bool) -> dict[str, Any]:
    # Receiver removes meta.finished while updating scheduler state. Carry a
    # separate stream_finished flag so Mimi can flush its held-back tail.
    return {
        "codes": {"audio": frames},
        "meta": {
            "finished": torch.tensor(finished),
            "stream_finished": torch.tensor(finished),
            "is_segment_finished": torch.tensor(finished),
            "codec_streaming": streaming,
        },
    }


def talker2depth_async_chunk(
    transfer_manager: Any, multimodal_output: dict[str, Any] | None, request: Any, is_finished: bool = False
) -> dict[str, Any] | None:
    request_id = str(request.external_req_id)
    finished = bool(is_finished or (getattr(request, "is_finished", lambda: False)()))
    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = defaultdict(list)
    if not hasattr(transfer_manager, "request_payload"):
        transfer_manager.request_payload = {}
    # These standard adapter maps are also cleared by cancellation/abort.
    buffers = transfer_manager.code_prompt_token_ids
    metadata = transfer_manager.request_payload
    frame = _frames(multimodal_output)
    if frame is not None:
        width = metadata.setdefault(request_id, {}).get("breeze_width", frame.shape[1])
        if frame.shape[1] != width:
            raise ValueError("Breeze codebook width changed within a request")
        metadata[request_id]["breeze_width"] = width
        buffers[request_id].extend(frame.split(1))
    cfg = getattr(getattr(transfer_manager, "connector", None), "config", {}) or {}
    cfg = cfg.get("extra", cfg)
    chunk_size = int(cfg.get("depth_chunk_frames", 8))
    if chunk_size <= 0:
        raise ValueError("depth_chunk_frames must be positive")
    pending = buffers[request_id]
    if not finished and len(pending) < chunk_size:
        return None
    count = len(pending) if finished else len(pending) // chunk_size * chunk_size
    width = metadata.get(request_id, {}).get("breeze_width", 0)
    emitted = torch.cat(pending[:count]) if count else torch.empty((0, width), dtype=torch.long)
    del pending[:count]
    if finished:
        buffers.pop(request_id, None)
        metadata.pop(request_id, None)
    return _payload(emitted, finished=finished, streaming=bool(cfg.get("codec_streaming", False)))


def talker2depth_full_payload(transfer_manager: Any, pooling_output: Any, request: Any) -> dict[str, Any]:
    """Worker-connector producer signature, matching the full-payload adapter."""
    frame = _frames(pooling_output)
    if frame is None:
        frame = torch.empty((0, 0), dtype=torch.long)
    return _payload(frame, finished=True, streaming=False)


def talker2depth_token_only(
    source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False
) -> list[dict[str, Any]]:
    """One scheduling placeholder per finished request; frames use connector."""
    return [{"prompt_token_ids": [0]} for output in source_outputs if output.finished]
