"""Load tensor-only core modules without importing the CUDA vLLM package."""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "vllm-omni/vllm_omni/model_executor/models/breeze_tts"
package = types.ModuleType("breeze_core")
package.__path__ = [str(CORE)]
sys.modules["breeze_core"] = package

reference = types.ModuleType("breeze_reference")
reference.__path__ = [str(ROOT / "breeze-tts/models")]
sys.modules["breeze_reference"] = reference

# The official config wrapper globally registers T5Gemma names which are
# already built into Transformers 5. Depth uses the identical base config;
# isolate that dependency without changing the reference model's math.
import importlib

sys.modules["breeze_reference.breeze_config"] = importlib.import_module(
    "breeze_reference.breeze_base_config"
)
