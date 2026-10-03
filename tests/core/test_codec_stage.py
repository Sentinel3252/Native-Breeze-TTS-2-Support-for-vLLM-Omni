"""Real batched Mimi wrapper output must remain attributable to requests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from breeze_core.codec_stream import MimiPrefixDecoder
from test_lifecycle import source_methods
from test_numerics import mimi


@dataclass
class Output:
    text_hidden_states: torch.Tensor
    multimodal_outputs: dict


def stage():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm-omni/vllm_omni/model_executor/models/breeze_tts/breeze_depth_codec.py"
    )
    methods = source_methods(
        path,
        "BreezeDepthCodecDecoder",
        {"forward", "on_requests_finished"},
        dict(torch=torch, Any=Any, OmniOutput=Output),
    )
    model = type("CodecStageCallbacks", (), methods)()
    model.codec_model = mimi()
    model.config = type("Config", (), {"num_codebooks": 4})()
    model.sample_rate = 160
    model._decoder = MimiPrefixDecoder(model.codec_model, 4)
    return model


def test_codec_stage_batch_separates_waveforms_and_terminal_flags():
    model = stage()
    a, b = torch.zeros((3, 4), dtype=torch.long), torch.ones((4, 4), dtype=torch.long)
    payloads = [
        {"codes": {"audio": a}, "meta": {"stream_finished": True}},
        {"codes": {"audio": b}, "meta": {"stream_finished": False}},
    ]
    result = model.forward(
        torch.zeros(2, dtype=torch.long),
        request_ids=["a", "b"],
        model_intermediate_buffer=payloads,
    )
    audio = result.multimodal_outputs["audio"]
    assert len(audio) == 2 and audio[1].numel() == 0
    expected = (
        model.codec_model.decode(a.T.unsqueeze(0)).audio_values.reshape(-1).float()
    )
    torch.testing.assert_close(audio[0], expected)
    assert set(model._decoder.states) == {"b"}
    model.on_requests_finished(["b"])
    assert not model._decoder.states


def test_codec_stage_profile_does_not_create_request_ids_and_missing_ids_fail():
    model = stage()
    model.forward(torch.zeros(2, dtype=torch.long))
    assert not model._decoder.states
    with pytest.raises(ValueError, match="explicit"):
        model.forward(
            torch.zeros(1, dtype=torch.long), runtime_additional_information=[{}]
        )
