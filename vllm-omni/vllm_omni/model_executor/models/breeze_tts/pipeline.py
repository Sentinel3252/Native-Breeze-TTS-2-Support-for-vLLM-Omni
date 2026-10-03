# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze TTS 2 pipeline: native AR + frame depth -> Mimi waveform decoder."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.breeze_tts"

BREEZE_TTS_PIPELINE = PipelineConfig(
    model_type="breeze",
    default_deploy_config_name="breeze_tts.yaml",
    model_arch="BreezeTTSTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="breeze_tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2depth_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2depth_full_payload",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="breeze_depth_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="BreezeDepthCodecDecoder",
            retains_state_across_chunks=True,
            sync_process_input_func=f"{_PROC}.talker2depth_token_only",
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)

__all__ = ["BREEZE_TTS_PIPELINE"]
