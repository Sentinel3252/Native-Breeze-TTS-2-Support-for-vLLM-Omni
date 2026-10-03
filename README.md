# Breeze TTS 2 for vLLM-Omni

Native autoregressive execution and a two-stage speech pipeline for Breeze TTS 2.

[中文](README.zh-CN.md) · [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md)

This project integrates Breeze TTS 2 into vLLM-Omni. Stage 0 runs native
AR attention and completes each audio frame with the reference depth decoder;
Stage 1 synthesizes the complete frames with Mimi. The integration lives in the
included `vllm-omni/` source tree.

**Current development version: `0.1.0-dev` — Core implementation.** This is an
independent open-source model integration, with no stable release yet.
Model registration, stage transport,
depth decoding, text/instruction conditioning and history-preserving Mimi
are implemented with CPU regression tests. Real-checkpoint loading, native
AR parity, full-model audio continuity and concurrent serving still
require Linux/CUDA validation. There is no published performance result or
validated deployment configuration yet.

The initial scope is text/instruction speech synthesis through vLLM-Omni.
Efficient streaming, voice cloning and advanced execution are planned extensions.
The version identifies the development baseline; it does not indicate a
published Git tag or a verified runtime release.

## What is included

- Native Qwen3/llama-like AR attention, complete-frame feedback and strict weight coverage.
- A two-stage pipeline with incremental, request-isolated complete-codebook transport.
- Reference depth attention with frame-local KV cache and request-local sampling.
- Official text/instruction templates, text encoder projections and DimFusion.
- Mimi prefix replay with terminal-tail flushing and request cleanup hooks.
- A `/v1/audio/speech` adapter and async-chunk deployment configuration.
- A dependency-free speech client, local checks, contribution templates, and source packaging.

The code uses vLLM's scheduler and KV-cache execution path. This describes the
implementation; it does not establish checkpoint correctness or throughput.

## Support status

| Area | Current state |
| --- | --- |
| Native AR execution | Implemented; checkpoint loading and numerical parity pending |
| Complete-frame transport | Async and full-payload callbacks implemented; CPU regression tests |
| Text/instruction speech API | Implemented; end-to-end synthesis pending |
| Incremental audio | Full-prefix replay preserves history; tiny Mimi continuity tests pass |
| Text encoder conditioning and prompt templates | Implemented; CPU projection parity tests against official methods |
| Voice cloning, multi-speaker conditioning, CFG | Not supported; unsupported requests fail explicitly |
| Concurrency and cancellation | Request isolation/cleanup tested on CPU; real serving validation pending |
| Parallelism and graph capture | Initial path requires TP/PP=1, eager execution and disabled prefix caching |
| Performance | No measured latency, memory requirement, or throughput claim |

See the [validation guide](docs/validation.md) for the checks needed to promote
a capability to supported. Features of the original Breeze model are not
automatically features of this integration.

## Versions and planned milestones

| Version | Scope | Status / completion criteria |
| --- | --- | --- |
| `0.1.0-dev` | Core implementation: text conditioning, native AR, depth completion, full-frame feedback, Mimi prefix replay and speech API | Current baseline; CPU checks pass, real-checkpoint Linux/CUDA validation pending |
| `0.1.0` | Validated basic model support | Load a real checkpoint, synthesize intelligible audio, verify streaming and serving lifecycle, document a reproducible environment |
| `0.2.0` | Efficient streaming and service operation | True stateful Mimi decoding, batched depth decoding, request isolation, bounded state and failure/cancellation cleanup |
| `0.3.0` | Voice cloning and CFG | Reference audio/text and speaker conditioning, multi-branch CFG, reference parity and API coverage |
| `0.4.0` | Advanced execution and developer tooling | Validate execution optimizations individually; improve checkpoint preflight, startup diagnostics and integration tooling |

Future versions are plans, with no release dates or support guarantees. Developer
tools and regressions should improve throughout the series. Advanced features
remain disabled until their own checks pass; implementing one does not imply
support for every parallelism, quantization or graph-capture configuration.

Preserve versions as Git tags and source archives with checksums and validation
records. Keep one maintained source tree. The current workspace has no root Git
repository, so archive snapshots come first; tags belong in the published source
repository. See the [roadmap](docs/roadmap.md), [change history](docs/changelog.md)
and [release guide](docs/releasing.md). A reusable
[next-task prompt](docs/next-task-prompt.md) describes the `0.2.0` work.

## Getting started

For documentation, source checks, and client tests, Python 3.10+ is sufficient:

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
python examples/speech_client.py --help
```

Tensor/model tests run without vLLM or a downloaded checkpoint in a separate
Python 3.12 environment:

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

The recorded baseline has **35 core tests passing with both Transformers 4.57.3
and 5.10.1**, plus **18 standalone client/packaging tests**. These use tiny models
and CPU callback harnesses. See [CPU test details](tests/core/README.md) and
[current validation](docs/core-validation.md) for the scope of this evidence.

Inference development targets **Linux with an NVIDIA CUDA GPU**. Install the
modified source tree, rather than an unmodified `vllm-omni` wheel. The
[setup guide](docs/getting-started.md) covers separate runtime environments,
checkpoint preparation, and the remaining compatibility gates.

After preparing a compatible checkpoint and installing the runtime, the
development launch command is:

```bash
# Run from this project's root. Replace the path with your local checkpoint.
vllm serve /path/to/breeze-tts-2 --omni \
  --stage-configs-path vllm-omni/vllm_omni/deploy/breeze_tts.yaml \
  --host 127.0.0.1 --port 8000
```

This is a starting point for checkpoint validation, not a verified quick-start
deployment. Once the server produces valid audio:

```bash
python examples/speech_client.py \
  --model /path/to/breeze-tts-2 \
  --text "Hello from Breeze TTS." \
  --output outputs/hello.wav
```

The client accepts Chinese text, an optional API key through `BREEZE_API_KEY`,
and `--stream` for raw PCM streaming. See [API usage](docs/api.md) for the
request contract and timing limitations.

## Architecture

```text
POST /v1/audio/speech
        │
        ▼
Text-only adapter → marked text token IDs
        │
        ▼
Stage 0: native AR → sample codebook 0 → depth completion → full RVQ frame
        │
        ▼
Request-keyed bridge → incremental frame payloads
        │
        ▼
Stage 1: full codec history → Mimi prefix replay → per-request audio
```

The default bridge forwards eight frames at a time and flushes the terminal
tail. Mimi replays the full prefix to preserve all decoder context. Streaming
holds back a short tail; a final chunk flushes it. This correctness baseline
has quadratic total decode work. See the [architecture guide](docs/architecture.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `vllm-omni/` | Upstream source snapshot with the native Breeze integration |
| `breeze-tts/` | Official inference source used as the reference implementation |
| `docs/` | Setup, API, design, validation, version roadmap, change history and release guidance |
| `examples/` | Standalone client for an already-running server |
| `scripts/` | Source checks and clean release archive generation |
| `tests/` | Standalone tooling and CPU tensor/callback regression tests |
| `project-manifest.json` | Upstream base commits and integration file inventory |

The local workspace contains two nested Git checkouts. Use the
[release guide](docs/releasing.md) to export a complete source archive without
nested Git metadata; a plain `git add` of these directories can omit their contents.

## Contributing

Real-checkpoint AR parity, audio quality and concurrent serving validation are
the current priorities. Documentation fixes and reproducible
failure reports are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and
the [roadmap](docs/roadmap.md). Submit integration issues to the repository
hosting this project; use the upstream trackers for confirmed upstream defects.

## License and attribution

Project source and documentation are provided under [Apache-2.0](LICENSE),
subject to the notices and component licenses in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Breeze model materials have a separate [research and non-commercial license](breeze-tts/MODEL_LICENSE);
this project's code license does not grant rights to those materials.
Model weights and generated audio are not included in source releases.

Built on [vLLM-Omni](https://github.com/vllm-project/vllm-omni),
[vLLM](https://github.com/vllm-project/vllm), and
[BreezeBlue's reference implementation](https://github.com/breezeblue-ai/breeze-tts).
This is an independent integration project, with no claim of upstream endorsement.
