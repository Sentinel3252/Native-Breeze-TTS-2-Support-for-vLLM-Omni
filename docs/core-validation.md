# Core implementation validation record

Local validation date: 2026-10-03. This records CPU implementation checks,
not pretrained-checkpoint support or serving performance.

## Environment

- Windows, Python 3.12.14, PyTorch 2.9.1+cpu.
- Runtime compatibility: Transformers 5.10.1.
- Official reference environment: Transformers 4.57.3.
- Reference source: `breeze-tts` at `ca632ce6c4d05f7985da4eab29b1a5d445b43f7b`.
- Models use tiny configurations and randomly initialized weights; no model
  download or pretrained checkpoint was used.

Local results: **35 core tests passed under both Transformers versions**;
the **18 standalone client/packaging tests passed**. Source/documentation
checks passed for the updated integration inventory.

## Checks and scope

The core suite checks official depth logits and complete greedy codebooks,
including different backbone/audio/depth projection sizes. Transformers 4
compares against the live official model. Transformers 5 compares against the
same official output saved in a random-weight fixture. Float32 tensor checks
use PyTorch's closeness assertions; greedy code IDs must match exactly.

Text conditioning is compared with the official segment encoding/projection
methods, including selected features, cropping to backbone layer count and
first-layer DimFusion. Those methods share the loaded test encoder/projection
weights so the comparison tests processing semantics directly.

Tiny real Mimi tests concatenate incremental waveform deltas and compare them
with complete decode, including an empty final flush. Stream code holds back
two frames and rejects prefixes whose emitted samples change beyond tolerance
`atol=rtol=2e-3`. Full-checkpoint tolerances still need measurement.

Worker/adapter callbacks run from their actual source in CPU harnesses. Tests
cover same-step sampling, valid codec zero, EOS suppression, partial prefill,
KV-preemption replay, request isolation, complete-frame transport, terminal
boundaries and cleanup. Weight coverage rejects missing fused shards and
persistent codec buffers while accepting valid tied aliases.

## Remaining gates

The machine does not provide a Linux/CUDA vLLM execution environment for this
validation. Native paged-attention outputs, actual checkpoint loading, audible
speech, scheduler cancellation and concurrent OpenAI serving remain untested.
The default YAML is a development configuration. TP/PP, quantization,
speculative decoding, prefix caching and graph capture are restricted.

Mimi prefix replay preserves context through complete re-decoding. Its total
work across chunks is quadratic; this record makes no latency, throughput or
minimum-memory claim. Voice cloning and multi-branch CFG remain unsupported.
