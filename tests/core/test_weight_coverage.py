import pytest
import torch
from breeze_core.weight_coverage import validate_weight_coverage
from torch import nn


def test_missing_shards_cannot_hide_behind_fused_parameter_names():
    model = nn.Module()
    model.qkv_proj = nn.Linear(4, 12, bias=False)
    model.gate_up_proj = nn.Linear(4, 8, bias=False)
    loaded = {"qkv_proj.weight", "gate_up_proj.weight"}
    with pytest.raises(ValueError, match="k_proj"):
        validate_weight_coverage(model, loaded, {"q_proj.weight", "gate_proj.weight"})
    validate_weight_coverage(
        model,
        loaded,
        {
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
        },
    )


def test_tied_alias_is_accepted_but_persistent_codec_buffers_are_required():
    model = nn.Module()
    model.backbone = nn.Embedding(8, 4)
    model.depth = model.backbone
    model.register_buffer("codec_centroids", torch.ones(8, 4))
    model.register_buffer("rope", torch.ones(4), persistent=False)
    validate_weight_coverage(model, {"depth.weight"}, {"depth.weight"})
    with pytest.raises(ValueError, match="codec_centroids"):
        validate_weight_coverage(
            model, {"depth.weight"}, {"depth.weight"}, persistent_buffers=True
        )
    validate_weight_coverage(
        model,
        {"depth.weight", "codec_centroids"},
        {"depth.weight"},
        persistent_buffers=True,
    )
