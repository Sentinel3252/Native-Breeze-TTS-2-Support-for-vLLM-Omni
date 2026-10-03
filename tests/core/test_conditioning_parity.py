"""Use the official segment/projection methods as an independent oracle."""

from pathlib import Path
from typing import Any

import pytest
import torch
from breeze_core.text_conditioning import TextConditioning
from test_lifecycle import source_methods
from test_numerics import tiny_config
from torch import nn


@pytest.mark.parametrize("fuse_first", [True, False])
@pytest.mark.parametrize("feature_indices", [[-1], [0, -1]])
def test_text_projection_matches_official_segments_and_layer_selection(
    fuse_first, feature_indices
):
    config = tiny_config(
        text_encoder_config={
            "model_type": "t5gemma2_text",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "sliding_window": 4,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        },
        text_encoder_proj_type="breeze_dimfusion",
        text_encoder_dimfusion_fuse_first_layer=fuse_first,
        text_encoder_feature_layer_idx=feature_indices,
    )
    actual = TextConditioning(config).eval()
    path = Path(__file__).resolve().parents[2] / "breeze-tts/models/breeze.py"
    methods = source_methods(
        path,
        "BreezeForConditionalGeneration",
        {
            "_batched_text_encoder_forward",
            "_project_segments",
            "convert_input_ids_to_embeds",
        },
        dict(torch=torch, nn=nn, Any=Any),
    )
    oracle = type("OfficialConditioningMethods", (), methods)()
    for name in (
        "text_encoder",
        "text_encoder_proj",
        "text_encoder_layer_projs",
        "embed_text_tokens",
        "config",
    ):
        setattr(oracle, name, getattr(actual, name))
    oracle.text_encoder_feature_layer_idx = tuple(feature_indices)
    oracle.text_encoder_dimfusion_fuse_first_layer = fuse_first
    oracle.text_encoder_dimfusion_layer_start_idx = 1
    oracle.text_encoder_dimfusion_layer_end_idx = None
    ids = torch.tensor([3, 4, 5, 6, 7, 8, 9])
    embeds, layers = actual(ids, [2, 5])
    with torch.inference_mode():
        expected, expected_layers = oracle.convert_input_ids_to_embeds(
            input_ids=ids.unsqueeze(0),
            text_ids_mask=torch.ones((1, ids.numel()), dtype=torch.bool),
            text_ids_len=torch.tensor([2, 5]),
        )
    torch.testing.assert_close(embeds, expected.squeeze(0), atol=1e-5, rtol=1e-5)
    for result, reference in zip(layers, expected_layers, strict=True):
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
