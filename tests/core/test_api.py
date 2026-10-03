"""Speech adapter contract: real prompt builder, isolated serving context."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from breeze_core.frame_decoder import DepthSampling
from breeze_core.text_conditioning import speech_prompt_ids

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "vllm-omni/vllm_omni/entrypoints/openai/tts_adapters/breeze_tts.py"
tree = ast.parse(path.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
cls.decorator_list = []
cls.bases = []
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        cls,
    ],
    type_ignores=[],
)
namespace = dict(
    copy=copy,
    DepthSampling=DepthSampling,
    speech_prompt_ids=speech_prompt_ids,
    tokens_input=lambda **kwargs: kwargs,
    PreparedRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    apply_max_new_tokens=lambda params, request: params,
)
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
Adapter = namespace[cls.name]


def request(**kwargs):
    values = dict(
        input="你好",
        ref_audio=None,
        ref_text=None,
        speaker_embedding=None,
        voice=None,
        task_type=None,
        extra_params={},
        seed=42,
        instructions=None,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input": " "}, "empty"),
        ({"ref_audio": "a.wav"}, "cloning"),
        ({"voice": "alice"}, "S0"),
        ({"extra_params": {"cfg_scale": 2}}, "cfg_scale"),
        ({"extra_params": {"depth_top_p": 0}}, "sampling"),
        ({"extra_params": {"unknown": 1}}, "Unsupported"),
    ],
)
def test_unsupported_requests_fail_explicitly(kwargs, message):
    assert message in Adapter().validate(request(**kwargs))


def test_codec_stops_are_explicit_and_sampling_list_is_not_mutated():
    adapter = Adapter()
    adapter.ctx = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=2051))
        )
    )
    params = [SimpleNamespace(stop_token_ids=[0], ignore_eos=False, seed=None)]
    result = adapter.apply_sampling_overrides(params, request())
    assert result[0].stop_token_ids == [2051] and result[0].ignore_eos is True
    assert result[0].seed == 42 and result[0].repetition_penalty == 1
    assert params[0].stop_token_ids == [0] and params[0].seed is None


def test_text_vocab_is_carried_outside_the_narrow_scheduler_vocab():
    import asyncio

    tokenizer = SimpleNamespace(
        encode=lambda *args, **kwargs: [12000, 23000],
        decode=lambda *args, **kwargs: "rendered",
    )
    adapter = Adapter()
    adapter.ctx = SimpleNamespace(server=SimpleNamespace(tokenizer=tokenizer))
    prepared = asyncio.run(adapter.build(request(), [], False))
    assert prepared.prompt["prompt_token_ids"] == [0, 0]
    assert prepared.prompt["additional_information"]["breeze_prompt_ids"] == [
        12000,
        23000,
    ]
    assert prepared.prompt["additional_information"]["breeze_text_segment_lengths"] == [
        2
    ]
