# Validation and performance reporting

[Learning path (中文)](learning-guide.md) · [Correctness rationale (中文)](learning/04-correctness.md) · [CPU validation record](core-validation.md)

Support claims should be tied to a checkpoint revision, runtime environment,
and reproducible result. The current integration has not passed the GPU gates
below. No latency, throughput, or minimum-memory result is published.

## Checks available without a GPU

```bash
python scripts/check_project.py
python -m unittest discover -s tests -v
```

The project check validates the integration inventory, local documentation
links, JSON, and Python syntax without importing the runtime. Standalone tests
exercise the HTTP client's success/failure paths and source-packaging rules.
The standalone CI job runs these checks without importing vLLM or a model.
The separate CPU core job below executes tiny tensor models.

## CPU core numerical and callback tests

In a separate Python 3.12 environment:

```bash
python -m pip install -r tests/core/requirements.txt
python -m pytest tests/core -q
```

This suite runs tiny real depth/text/Mimi models and isolates worker/adapter
callbacks from GPU imports. It covers official depth logits and greedy frames,
text projection parity, complete feedback, KV-preemption replay, codec zero/EOS,
request RNG, waveform continuity, terminal tails, API rejection and weight
coverage. CI runs it with Transformers 4.57.3 and 5.10.1.
See [test details](../tests/core/README.md) and the [local record](core-validation.md).
These tests do not validate CUDA paged attention or a pretrained checkpoint.

## Native bridge regression

In a prepared native environment, run from `vllm-omni/`:

```bash
python -m pytest tests/model_executor/stage_input_processors/test_breeze_tts.py -v
```

This test exercises real tensor transport for interleaved request buffers and
terminal-tail flushing. It needs the runtime dependencies loaded by upstream
test fixtures. It does not establish checkpoint support or audio correctness.

## Checkpoint and correctness gates

| Gate | Required evidence |
| --- | --- |
| Weight coverage | Every required parameter loaded with the expected shape; explicit treatment of unused/reference-only tensors |
| Prompt and conditioning | Tokenizer, special tokens, text encoder features, reference format, and CFG behavior matched or clearly scoped |
| Stage-0 parity | Prefill/decode hidden states, logits, allowed vocabulary, and stopping behavior compared with reference |
| Frame alignment | Complete depth frame uses the hidden state from the same AR sampling step; first/final and token-limit frames are preserved |
| Depth parity | Complete codebooks compared under equivalent sampling; greedy and stochastic paths distinguished |
| Audio smoke | Nonempty finite waveform, sample rate, duration, and intelligibility checked on short English/Chinese inputs |
| Codec continuity | Complete decode compared with split frames/chunks; boundary discontinuities and duration drift measured |
| Serving isolation | Interleaved long/short requests, EOS tails, cancellation, failure cleanup, and per-request output attribution |

Use fixed seeds where sampling applies and state numeric tolerances. A matching
seed alone does not guarantee identical RNG consumption across implementations.
Run the reference in a separate environment and record both dependency sets.

## Serving validation

Begin with one short request. Once correctness passes, exercise concurrency
1, 2, 4, and 8, varying request lengths and chunk sizes. Include cancellation
before the first chunk, cancellation mid-stream, early EOS, token-limit stops,
and an injected failure. Verify no request inherits another request's buffered
frames, hidden states, or waveform, and no state survives termination.

## Performance measurements

Only compare paths with equivalent checkpoint, input, conditioning, and output
quality. Distinguish cold startup from warmed inference. Report:

- TTFA: time from request submission to the first playable audio samples,
  including an explicit sample threshold and measurement location.
- End-to-end latency: request submission to final audio completion; include p50/p95.
- RTF: end-to-end latency / generated audio duration, with a consistent definition.
- Throughput: completed requests/s and generated audio seconds/wall-clock second.
- Resources: observed peak GPU memory, stage placement, and GPU utilization.
- Failures: timeouts, cancellations, invalid/empty audio, and failed-request rate.

The example client's first-body-byte metric is diagnostic transport timing.
It must not be relabeled as TTFA. Keep failed requests in the report rather
than excluding them from a favorable average.

## Report format

Attach a report with these fields to a validation PR or issue:

```text
Source revision / source archive SHA-256:
Checkpoint ID and immutable revision:
OS, GPU model/count, driver, CUDA, Python:
vLLM / vLLM-Omni / PyTorch / Transformers versions:
Launch command and stage configuration:
Inputs, reference conditioning, seed and sampling settings:
Warmup, measured repetitions, concurrency, chunk size:
Correctness tolerances and results:
Latency distributions, audio durations, throughput, peak memory:
Failures and unverified cases:
```

Share only audio you are permitted to disclose under the applicable terms.
Change the README support table only when the corresponding evidence is reviewable.
