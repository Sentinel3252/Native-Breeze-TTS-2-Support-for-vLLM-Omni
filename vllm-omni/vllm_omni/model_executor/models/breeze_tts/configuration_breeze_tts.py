# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint configuration and the native paged-attention backbone view."""

from __future__ import annotations

from typing import Any

from transformers import LlamaConfig, PretrainedConfig, Qwen3Config

from .configuration_breeze_base import BreezeConfig


def config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, PretrainedConfig):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("Nested Breeze configurations must be dictionaries or PretrainedConfig objects")


class BreezeTTSConfig(BreezeConfig):
    """Retain all official fields; expose only the AR transformer to vLLM."""

    model_type = "breeze"

    def __init__(
        self,
        text_encoder_config: Any = None,
        backbone_config: Any = None,
        backbone_model_type: str = "breeze",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.text_encoder_config = None if text_encoder_config is None else config_dict(text_encoder_config)
        self.backbone_config = None if backbone_config is None else config_dict(backbone_config)
        self.backbone_model_type = backbone_model_type
        if backbone_config is not None:
            for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "rope_theta",
                "rope_scaling",
                "max_position_embeddings",
                "hidden_act",
                "attention_bias",
                "mlp_bias",
            ):
                if name in self.backbone_config:
                    setattr(self, name, self.backbone_config[name])
        self.backbone_eos_token_id = self.vocab_size
        self.eos_token_id = self.backbone_eos_token_id
        self.codec_codebook_size = int(self.codec_config.codebook_size)
        if not 0 < self.codec_codebook_size <= self.vocab_size:
            raise ValueError("codec codebook_size must fit inside the Breeze audio vocabulary")
        if int(self.depth_decoder_config.num_codebooks) != self.num_codebooks:
            raise ValueError("Depth decoder and Breeze must use the same num_codebooks")
        if int(self.depth_decoder_config.vocab_size) != self.vocab_size:
            raise ValueError("Depth decoder and Breeze must use the same audio vocabulary")
        if int(self.depth_decoder_config.backbone_hidden_size) != self.hidden_size:
            raise ValueError("Depth backbone_hidden_size must match the temporal AR backbone")
        if self.num_codebooks > int(self.codec_config.num_quantizers):
            raise ValueError("Breeze num_codebooks exceeds Mimi's configured quantizers")
        if int(self.codec_config.audio_channels) != 1:
            raise ValueError("Breeze native speech currently requires a mono Mimi codec")

    def get_text_config(self, **_: Any) -> PretrainedConfig:
        """vLLM uses this view for AR dimensions, RoPE and logits metadata."""
        # Transformers 5 validates token IDs from inside PretrainedConfig's
        # constructor, before the Breeze-specific architecture fields exist.
        if not hasattr(self, "backbone_model_type") or not hasattr(self, "vocab_size"):
            return self
        if self.backbone_model_type not in {"breeze", "llama3", "qwen3"}:
            raise ValueError(f"Unsupported Breeze backbone: {self.backbone_model_type}")
        if self.backbone_model_type != "breeze" and self.backbone_config is None:
            raise ValueError("Native execution requires an embedded backbone_config")
        raw = dict(self.backbone_config or {})
        raw.pop("model_type", None)
        raw.pop("tie_word_embeddings", None)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "max_position_embeddings",
            "hidden_act",
            "attention_bias",
            "mlp_bias",
        ):
            raw[name] = getattr(self, name)
        raw.update(vocab_size=self.vocab_size + 1, eos_token_id=self.vocab_size, pad_token_id=None)
        if raw.get("rope_parameters") is None:
            raw["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                **(self.rope_scaling or {}),
            }
        cls = Qwen3Config if self.backbone_model_type == "qwen3" else LlamaConfig
        view = cls(tie_word_embeddings=False, **raw)
        if not hasattr(view, "rope_parameters"):
            view.rope_parameters = {"rope_type": "default", "rope_theta": self.rope_theta, **(self.rope_scaling or {})}
        return view


__all__ = ["BreezeTTSConfig", "config_dict"]
