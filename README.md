# Breeze TTS 2 for vLLM-Omni

This project adds Breeze TTS 2 support to vLLM-Omni and is intended for learning.
It includes the official inference code alongside the modified framework source,
so you can compare the two and see what the integration changes.

[中文](README.zh-CN.md) · [Learning guide (中文)](docs/learning-guide.md) · [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Roadmap](docs/roadmap.md)

The official reference is in `breeze-tts/`; the integration is in `vllm-omni/`.
The learning guide below explains the changes and how a request runs through them.

## Current progress

The current tag is `v0.1.0-dev`. The core integration code is largely complete.
Model registration, text/instruction conditioning, native autoregressive (AR)
execution, depth completion, complete-frame feedback, stage transport, Mimi audio
decoding, and the `/v1/audio/speech` adapter are implemented.

So far, 35 core CPU tests have passed under both Transformers 4.57.3 and 5.10.1,
along with 18 client and packaging tests. See the [validation record](docs/core-validation.md).
These tests use tiny models and CPU callback harnesses. Real-checkpoint loading,
CUDA numerical parity, speech quality, and concurrent serving have not been
verified yet. There are no performance measurements or stable releases.

| Area | Current state |
| --- | --- |
| Core text/instruction speech pipeline | Implemented; real-model end-to-end validation pending |
| Incremental audio | Mimi retains history and re-decodes the full prefix; tiny-model continuity tests pass, efficiency work remains |
| Request isolation and cleanup | CPU tests cover these; real serving cancellation, failure, and concurrency checks remain |
| Voice cloning, multi-speaker conditioning, CFG | Not supported yet |
| Execution configuration | Currently requires TP/PP=1, eager mode, and disabled prefix caching |

Use `git show v0.1.0-dev` to inspect this version and compare it with later changes.

## Next steps

| Planned version | Work |
| --- | --- |
| `0.1.0` | Verify real weights, compare AR/depth outputs with the reference, check Chinese/English speech and serving behavior, and record the Linux/CUDA environment |
| `0.2.0` | Cache Mimi decoder state, batch depth decoding, limit retained request state, handle failures and cancellation, and measure performance |
| `0.3.0` | Add reference audio/text, speaker conditioning, multi-branch CFG, and the API support and tests for them |
| `0.4.0` | Check CUDA graphs, prefix caching, TP/PP, and quantization separately; add checkpoint preflight and startup diagnostics |

There are no fixed dates for these changes. See the [roadmap](docs/roadmap.md)
for details and the [next-task document](docs/next-task-prompt.md) for the streaming work.

## Learning path

The [learning guide (中文)](docs/learning-guide.md) covers Breeze generation,
the framework integration, request execution, and validation. The final article
describes the steps for adapting another model: reading the reference code,
choosing interfaces, loading weights, and handling request state.

The articles assume Python, PyTorch, and Transformer basics. Reading them
requires no installation or GPU. They link to the relevant source; setup
commands and validation procedures are in separate guides.

## Quick start

Run basic checks from the project root with Python 3.10+:

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python examples/speech_client.py --help
```

For model CPU tests, use a separate Python 3.12 environment. No vLLM installation
or checkpoint download is needed:

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

For inference, follow [getting started](docs/getting-started.md) to prepare the
runtime and checkpoint, and install this project's modified `vllm-omni/` source.
Then use the [client example](examples/speech_client.py) and [API guide](docs/api.md).
The launch configuration still needs runtime validation. The checks are listed
in the [validation guide](docs/validation.md).

## Repository and documentation

| Path | Purpose |
| --- | --- |
| `breeze-tts/` | Official reference for understanding and comparing model behavior |
| `vllm-omni/` | Framework source with the native Breeze integration |
| `docs/learning-guide.md`, `docs/learning/` | Learning guide, integration explanations, and source references in Chinese |
| `docs/` | Setup, architecture, API, validation, roadmap, [change history](docs/changelog.md), and [releases](docs/releasing.md) |
| `tests/` | Tooling and CPU numerical, transport, and lifecycle tests |
| `examples/`, `scripts/` | Speech client, source checks, and source packaging |
| `project-manifest.json` | Upstream base revisions and integration file inventory |

Learning documentation, reproducible reports, and integration improvements are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License and attribution

Project source and documentation use [Apache-2.0](LICENSE). Component notices are
in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Breeze model materials have a
separate [research and non-commercial license](breeze-tts/MODEL_LICENSE).
Source releases do not contain model weights or generated audio.

Built on [vLLM-Omni](https://github.com/vllm-project/vllm-omni),
[vLLM](https://github.com/vllm-project/vllm), and
[BreezeBlue](https://github.com/breezeblue-ai/breeze-tts).
This is an independent integration project.
