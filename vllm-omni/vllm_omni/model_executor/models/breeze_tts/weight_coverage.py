# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coverage checks including tied parameters and fused native projections."""

from collections.abc import Iterable

from torch import nn


def validate_weight_coverage(
    module: nn.Module, loaded: set[str], source_names: Iterable[str], *, persistent_buffers: bool = False
) -> None:
    aliases = {name: id(param) for name, param in module.named_parameters(remove_duplicate=False)}
    loaded_ids = {aliases[name] for name in loaded if name in aliases}
    missing = [name for name, param in module.named_parameters() if id(param) not in loaded_ids]
    sources = set(source_names)
    # A fused qkv/gate-up tensor can be reported loaded after just ONE shard.
    # Require every original checkpoint projection before accepting coverage.
    for name in aliases:
        if "qkv_proj." in name and name not in sources:
            missing.extend(
                name.replace("qkv_proj.", part + ".")
                for part in ("q_proj", "k_proj", "v_proj")
                if name.replace("qkv_proj.", part + ".") not in sources
            )
        if "gate_up_proj." in name and name not in sources:
            missing.extend(
                name.replace("gate_up_proj.", part + ".")
                for part in ("gate_proj", "up_proj")
                if name.replace("gate_up_proj.", part + ".") not in sources
            )
    if persistent_buffers:
        missing.extend(set(module.state_dict()) - loaded - set(aliases))
    if missing:
        raise ValueError(f"Breeze checkpoint is missing required weights/buffers: {sorted(set(missing))[:12]}")
