# Getting started

[Project overview](../README.md) · [Learning path (中文)](learning-guide.md) · [API](api.md) · [Validation](validation.md)

The standalone tooling runs on Windows, macOS, or Linux with Python 3.10+.
The native inference integration targets Linux/CUDA and is not yet a validated
checkpoint deployment. No minimum GPU memory figure has been established for it.

## Check the source checkout

From the project root:

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
```

These commands need only Python's standard library. They check the project
files and standalone tools, not weight loading or model execution.

## Prepare the native runtime

Use a fresh Linux environment. The included upstream installation snapshot
specifies vLLM `0.28.0`; this is a development dependency starting point, not
a tested Breeze compatibility matrix. Review the included
[CUDA installation guide](../vllm-omni/docs/getting_started/installation/gpu/cuda.inc.md)
and [runtime requirements](../vllm-omni/requirements/common.txt) before installing.
The integration has no independent dependency lock yet.

```bash
# Run from the project root on Linux.
python3.12 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm==0.28.0'
VLLM_OMNI_VERSION_OVERRIDE=0.28.0.dev0 VLLM_OMNI_TARGET_DEVICE=cuda \
  python -m pip install -e ./vllm-omni
python -m pip check
```

The version override gives this local development install valid package
metadata even when the source export has no upstream Git tags. It is not a
project release number or evidence of compatibility.

CUDA/PyTorch binary compatibility depends on the selected wheels and driver.
For a different CUDA build, follow the [official vLLM installation instructions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).
Do not replace this source install with `pip install vllm-omni`: an upstream
wheel does not contain this checkout's unmerged integration.

## Prepare a checkpoint

Obtain Breeze TTS 2 from the [official model repository](https://huggingface.co/BreezeBlue/breeze-tts-2)
under its model-material terms. Keep the checkpoint outside versioned source,
for example under an ignored `checkpoints/` directory. Record its immutable
revision alongside the runtime versions used for validation.

Before expecting successful synthesis, verify these integration boundaries:

1. The HF config resolves to `model_type=breeze` and the native config retains
   all backbone, depth, codec, and special-token settings needed by the checkpoint.
2. Every required stage parameter loads with the expected shape. Report
   unexpected/missing tensors; do not bypass failures by accepting random weights.
3. The tokenizer and text encoder match the checkpoint. The adapter implements
   the official S0/text/instruction template; voice cloning and multi-branch
   CFG are explicitly unsupported.
4. Stop tokens and logits masks match the checkpoint. The reference model sets
   the backbone stop ID to `vocab_size`. Codec value zero remains valid;
   the tokenizer's text EOS must not end audio generation.
5. Stage-0 hidden states align with the sampled codebook-0 frame. Test the
   decode-step handoff and final frame, rather than inferring correctness from shapes.

These are full-checkpoint validation gates. CPU core tests pass, but they do
not establish support for a named checkpoint or CUDA serving.

## Launch the development server

After preparing the runtime and a compatible checkpoint:

```bash
vllm serve /path/to/breeze-tts-2 --omni \
  --stage-configs-path vllm-omni/vllm_omni/deploy/breeze_tts.yaml \
  --host 127.0.0.1 --port 8000
```

The [deployment YAML](../vllm-omni/vllm_omni/deploy/breeze_tts.yaml) enables
async chunks with an eight-frame bridge. Its sequence counts, context lengths,
and memory fractions are development defaults, not capacity measurements.
Do not assume a GPU count or memory budget from those settings; validate the
upstream stage/device allocation on your hardware before tuning concurrency.

The serving layer can provide `/health` and `/v1/models` checks. A healthy
server alone does not establish valid speech output. Follow [validation](validation.md)
and make a minimal request with [the example client](../examples/speech_client.py).

## Run the official reference separately

The official inference tree currently pins Transformers `4.57.3`, while the
included native tree requires `>=5.10.1,<5.15`. They must use separate environments:

```bash
python3.12 -m venv .venv-reference
source .venv-reference/bin/activate
python -m pip install --upgrade pip
python -m pip install -r breeze-tts/requirements.txt
```

Use the commands and prompt templates in the
[reference README](../breeze-tts/README.md). Keep checkpoint revision, input,
sampling settings, and seed recorded when comparing outputs.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| Unknown architecture/model type | Confirm the modified source is installed and the Breeze registry entries are present |
| Missing or mismatched weights | Compare checkpoint tensor names and shapes with both stage loaders |
| Text accepted but speech incorrect | Check prompt conditioning and code/hidden-state alignment against the reference |
| Stop behavior differs | Compare backbone EOS, codec EOS, and masked vocabulary IDs |
| Clicks or discontinuities | Compare full decode with prefix replay; disable `codec_streaming` when a checkpoint has unstable prefixes |
| Import/dependency conflicts | Separate reference and native environments; run `python -m pip check` |
| GPU out of memory | Inspect stage placement and allocation; current YAML is unvalidated |

Include the exact failure and environment in an issue. Avoid reporting a
checkpoint as supported until the corresponding validation gates pass.
