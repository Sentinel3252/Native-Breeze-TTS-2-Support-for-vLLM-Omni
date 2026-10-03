"""Execute the same transport tests without requiring vLLM on CPU hosts."""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
module_name = "vllm_omni.model_executor.stage_input_processors.breeze_tts"
for name in (
    "vllm_omni",
    "vllm_omni.model_executor",
    "vllm_omni.model_executor.stage_input_processors",
):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
spec = importlib.util.spec_from_file_location(
    module_name,
    ROOT / "vllm-omni/vllm_omni/model_executor/stage_input_processors/breeze_tts.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
spec = importlib.util.spec_from_file_location(
    "breeze_transport_tests",
    ROOT / "vllm-omni/tests/model_executor/stage_input_processors/test_breeze_tts.py",
)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
for name, value in vars(test_module).items():
    if name.startswith("test_"):
        globals()[name] = value
