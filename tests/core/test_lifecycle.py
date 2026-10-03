"""Run real worker/model callbacks on CPU; CUDA execution is a separate gate.

AST extraction isolates methods from imports of the GPU-only vLLM runner. It
does not replace their logic or the depth model with mock tensor operations.
"""

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
from breeze_core.depth_decoder import BreezeDepthDecoderForCausalLM
from breeze_core.frame_decoder import DepthSampling, complete_frame, embed_audio_frame
from breeze_core.text_conditioning import TextConditioning
from test_numerics import depth_config, tiny_config

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "vllm-omni/vllm_omni/model_executor/models/breeze_tts"


def source_methods(path, class_name, methods, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    selected = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    compiled = ast.Module(body=selected, type_ignores=[])
    exec(compile(compiled, str(path), "exec"), namespace)
    return {name: namespace[name] for name in methods}


def talker():
    def scalar(value, default):
        if value is None:
            return default
        if isinstance(value, (tuple, list)):
            return value[0]
        return value

    namespace = dict(
        torch=torch,
        Any=Any,
        DepthSampling=DepthSampling,
        complete_frame=complete_frame,
        embed_audio_frame=embed_audio_frame,
        _scalar=scalar,
    )
    methods = source_methods(
        MODELS / "breeze_tts_talker.py",
        "BreezeTTSTalkerForConditionalGeneration",
        {"preprocess", "post_sample_talker_mtp", "on_requests_finished"},
        namespace,
    )
    cls = type("TalkerCallbacks", (torch.nn.Module,), methods)
    model = cls()
    model.config = tiny_config()
    model.conditioning = TextConditioning(model.config).eval()
    model.audio_embedding = torch.nn.Embedding(44, 24)
    model.depth_decoder = BreezeDepthDecoderForCausalLM(depth_config()).eval()
    model._prompt_cache, model._depth_generators = {}, {}
    model._frame_history = {}
    return model


def test_prefill_chunks_decode_alignment_and_cleanup():
    model = talker()
    ids = torch.tensor([1, 2, 3, 4])
    info = dict(
        request_id="a",
        _omni_prompt_len=4,
        _omni_is_prefill=True,
        breeze_prompt_ids=ids.tolist(),
        breeze_text_segment_lengths=[4],
    )
    _, first, _ = model.preprocess(
        torch.zeros(2, dtype=torch.long), **info, _omni_num_computed_tokens=0
    )
    _, second, _ = model.preprocess(
        torch.zeros(2, dtype=torch.long), **info, _omni_num_computed_tokens=2
    )
    torch.testing.assert_close(
        torch.cat((first, second)), model.conditioning.embed_text_tokens(ids)
    )
    hidden = torch.randn(1, 24)
    codes = model.post_sample_talker_mtp(
        input_ids=torch.tensor([0]),
        hidden_states=hidden,
        req_ids=["a"],
        req_infos=[{"breeze_depth_temperature": [0], "_omni_seed": 17}],
    )
    expected = complete_frame(
        model.depth_decoder, torch.tensor([0]), hidden, 8, DepthSampling(temperature=0)
    )
    torch.testing.assert_close(codes, expected)
    decode_info = dict(request_id="a", _omni_prompt_len=4, _omni_num_computed_tokens=4)
    _, feedback, update = model.preprocess(torch.tensor([0]), **decode_info)
    torch.testing.assert_close(
        feedback, embed_audio_frame(model.audio_embedding, codes, 11, 4)
    )
    assert not bool(update["breeze_text_mask"].any())
    with pytest.raises(ValueError, match="out of step"):
        model.preprocess(torch.tensor([1]), **decode_info)
    eos = model.post_sample_talker_mtp(
        input_ids=torch.tensor([11]),
        hidden_states=hidden,
        req_ids=["a"],
        req_infos=[{}],
    )
    assert bool((eos == -1).all())
    model.on_requests_finished(["a"])
    assert (
        not model._prompt_cache
        and not model._depth_generators
        and not model._frame_history
    )


def test_kv_preemption_replays_complete_frames_without_new_depth_sampling():
    model = talker()
    ids = torch.tensor([1, 2, 3])
    info = dict(
        request_id="a",
        _omni_prompt_len=3,
        breeze_prompt_ids=ids.tolist(),
        breeze_text_segment_lengths=[3],
    )
    model.preprocess(
        torch.zeros(3, dtype=torch.long), **info, _omni_num_computed_tokens=0
    )
    for token in (0, 2):
        model.post_sample_talker_mtp(
            input_ids=torch.tensor([token]),
            hidden_states=torch.randn(1, 24),
            req_ids=["a"],
            req_infos=[{"breeze_depth_temperature": [0]}],
        )
    history = torch.cat(model._frame_history["a"]).clone()
    _, replay, _ = model.preprocess(
        torch.tensor([0, 0, 0, 0, 2]), **info, _omni_num_computed_tokens=0
    )
    expected = torch.cat(
        (
            model.conditioning.embed_text_tokens(ids),
            embed_audio_frame(model.audio_embedding, history, 11, 4),
        )
    )
    torch.testing.assert_close(replay, expected)
    assert len(model._frame_history["a"]) == 2


def runner(model):
    namespace = dict(torch=torch, Any=Any)
    methods = source_methods(
        ROOT / "vllm-omni/vllm_omni/worker/gpu_ar_model_runner.py",
        "GPUARModelRunner",
        {"_run_post_sample_talker_mtp"},
        namespace,
    )
    cls = type("RunnerCallback", (), methods)
    instance = cls()
    instance.model = model
    instance.use_async_scheduling = False
    instance.model_intermediate_buffer = {
        "a": {"breeze_depth_temperature": [0]},
        "b": {},
    }
    instance._update_intermediate_buffer = lambda rid, update: (
        instance.model_intermediate_buffer[rid].update(update)
    )
    return instance


def test_sampling_hook_first_and_final_frame_eos_and_partial_prefill():
    model = talker()
    model.post_sample_all_requests = True
    model.post_sample_drop_negative_frames = True
    worker = runner(model)
    args = dict(
        req_ids=["a", "b"],
        sampled_token_ids=torch.tensor([[0], [11]]),
        invalid_req_indices=[],
        sample_hidden_states=torch.randn(2, 24),
        multimodal_outputs={},
    )
    output = worker._run_post_sample_talker_mtp(
        valid_sampled_token_ids=[[0], [11]], **args
    )
    assert output["codes"]["audio"][0].shape == (1, 4)
    assert output["codes"]["audio"][1].shape == (0, 4)
    assert worker.model_intermediate_buffer["a"]["codes"]["audio"].shape == (1, 4)
    output = worker._run_post_sample_talker_mtp(
        valid_sampled_token_ids=[[], [11]], **args
    )
    assert output["codes"]["audio"][0].numel() == 0


def test_sampling_hook_preserves_legacy_opt_in_and_rejects_speculation():
    model = talker()
    worker = runner(model)
    args = dict(
        req_ids=["a"],
        sampled_token_ids=torch.tensor([[0]]),
        invalid_req_indices=[],
        sample_hidden_states=torch.randn(1, 24),
        multimodal_outputs={"original": True},
    )
    assert worker._run_post_sample_talker_mtp(
        valid_sampled_token_ids=[[0]], **args
    ) == {"original": True}
    model.post_sample_all_requests = True
    with pytest.raises(ValueError, match="exactly one"):
        worker._run_post_sample_talker_mtp(valid_sampled_token_ids=[[0, 1]], **args)
