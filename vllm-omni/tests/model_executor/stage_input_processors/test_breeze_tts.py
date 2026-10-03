# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.breeze_tts import (
    talker2depth_async_chunk,
    talker2depth_full_payload,
    talker2depth_token_only,
)


def manager():
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"depth_chunk_frames": 2, "codec_streaming": True}}),
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
    )


def request(name):
    return SimpleNamespace(external_req_id=name, request_id=name, is_finished=lambda: False)


def test_incremental_frames_request_isolation_and_tail_flush():
    transfer = manager()
    a = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])
    first = talker2depth_async_chunk(transfer, {"codes": {"audio": a}}, request("a"))
    assert first["codes"]["audio"].tolist() == a[:2].tolist()
    b = torch.tensor([[9, 0, 1, 2]])
    assert talker2depth_async_chunk(transfer, {"codes": {"audio": b}}, request("b")) is None
    last = talker2depth_async_chunk(transfer, None, request("a"), is_finished=True)
    assert last["codes"]["audio"].tolist() == a[2:].tolist()
    assert bool(last["meta"]["stream_finished"])
    assert "a" not in transfer.code_prompt_token_ids and "a" not in transfer.request_payload
    assert "b" in transfer.code_prompt_token_ids
    tail = talker2depth_async_chunk(transfer, None, request("b"), is_finished=True)
    assert tail["codes"]["audio"].tolist() == b.tolist()


def test_zero_frames_are_valid_and_empty_terminal_is_forwarded():
    transfer = manager()
    zero = torch.zeros((2, 4), dtype=torch.long)
    payload = talker2depth_async_chunk(transfer, {"codes": {"audio": zero}}, request("a"))
    assert torch.equal(payload["codes"]["audio"], zero)
    terminal = talker2depth_async_chunk(transfer, None, request("a"), is_finished=True)
    assert terminal["codes"]["audio"].shape == (0, 4)
    assert bool(terminal["meta"]["finished"])
    # Empty terminal chunks must schedule the generation stage once, otherwise
    # the receiver would finish without flushing Mimi's held-back tail.
    assert bool(terminal["meta"]["is_segment_finished"])
    assert not transfer.code_prompt_token_ids and not transfer.request_payload


def test_full_payload_and_sync_callbacks_use_runtime_contract():
    frames = torch.tensor([[0, 1, 2, 3]])
    payload = talker2depth_full_payload(manager(), {"codes.audio": frames}, request("a"))
    assert torch.equal(payload["codes"]["audio"], frames)
    assert bool(payload["meta"]["finished"]) and not payload["meta"]["codec_streaming"]
    outputs = [SimpleNamespace(finished=True), SimpleNamespace(finished=False)]
    assert talker2depth_token_only(outputs) == [{"prompt_token_ids": [0]}]


def test_partial_prefill_empty_runner_rows_do_not_emit_or_replay_audio():
    transfer = manager()
    assert (
        talker2depth_async_chunk(transfer, {"codes": {"audio": torch.empty(0, dtype=torch.long)}}, request("a")) is None
    )
    assert not transfer.code_prompt_token_ids["a"]


@pytest.mark.parametrize("codes", [torch.ones(4), torch.tensor([[1, -1]]), torch.ones((1, 4))])
def test_invalid_frames_fail_before_transfer(codes):
    with pytest.raises(ValueError):
        talker2depth_async_chunk(manager(), {"codes": {"audio": codes}}, request("a"))
