# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI speech request normalization for Breeze TTS 2."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.breeze_tts.frame_decoder import DepthSampling
from vllm_omni.model_executor.models.breeze_tts.text_conditioning import speech_prompt_ids

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class BreezeTTSAdapter(ARTTSAdapter):
    """Text/instruction speech with one speaker and CFG scale 1."""

    stage_keys = frozenset({"breeze_tts", "breeze_depth_codec"})
    name = "breeze"

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio or request.ref_text or request.speaker_embedding:
            return "Breeze native voice cloning is not implemented"
        if request.voice not in (None, "S0", "s0", "default"):
            return "Breeze native speech currently supports only the S0 speaker"
        if request.task_type is not None:
            return "Use text and optional instructions for Breeze; task_type is unsupported"
        extra = request.extra_params or {}
        allowed = {"depth_temperature", "depth_top_k", "depth_top_p", "cfg_scale"}
        if set(extra) - allowed:
            return f"Unsupported Breeze parameters: {', '.join(sorted(set(extra) - allowed))}"
        try:
            if float(extra.get("cfg_scale", 1.0)) != 1.0:
                return "Breeze native execution currently requires cfg_scale=1"
            DepthSampling(
                float(extra.get("depth_temperature", 0.9)),
                int(extra.get("depth_top_k", 50)),
                float(extra.get("depth_top_p", 1.0)),
            )
        except (ValueError, TypeError):
            return "Invalid Breeze depth sampling parameters"
        return None

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        error = self.validate(request)
        if error or has_inline_ref_audio:
            raise ValueError(error or "Breeze native voice cloning is not implemented")
        ids = speech_prompt_ids(self.ctx.server.tokenizer, request.input, request.instructions)
        extra = request.extra_params or {}
        params = {
            "breeze_prompt_ids": ids,
            "breeze_text_segment_lengths": [len(ids)],
            "breeze_depth_temperature": [float(extra.get("depth_temperature", 0.9))],
            "breeze_depth_top_k": [int(extra.get("depth_top_k", 50))],
            "breeze_depth_top_p": [float(extra.get("depth_top_p", 1.0))],
        }
        # Scheduler token IDs share the narrow codec vocabulary. Real text
        # IDs travel separately and are encoded by the checkpoint text model.
        prompt = tokens_input(prompt_token_ids=[0] * len(ids))
        prompt["additional_information"] = params
        return PreparedRequest(prompt=prompt, tts_params=params, model_type=self.name)

    def apply_sampling_overrides(
        self, sampling_params_list: list, request: OpenAICreateSpeechRequest, **_: object
    ) -> list:
        params = copy.deepcopy(apply_max_new_tokens(sampling_params_list, request))
        config = self.ctx.engine_client.model_config.hf_config
        params[0].stop_token_ids = [int(config.vocab_size)]
        # The text tokenizer's EOS may overlap valid codec IDs. Stop only on
        # the explicitly listed backbone EOS, which lives outside codec IDs.
        params[0].ignore_eos = True
        params[0].detokenize = False
        params[0].presence_penalty = 0.0
        params[0].frequency_penalty = 0.0
        params[0].repetition_penalty = 1.0
        if request.seed is not None:
            params[0].seed = int(request.seed)
        return params
