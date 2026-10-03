# Version roadmap

The project provides native Breeze TTS 2 model integration for vLLM-Omni.
The current development baseline is **`0.1.0-dev` — Core implementation**.
Future versions below are planned scopes, not published releases or delivery dates.
The [validation guide](validation.md) defines the evidence needed to claim support.

## `0.1.0-dev`: current core implementation

Implemented: text/instruction conditioning and DimFusion, native AR integration,
reference depth completion, full-frame feedback, request-keyed transport,
history-preserving Mimi prefix replay, speech API and strict weight coverage.
The baseline also includes source packaging, documentation and CPU regressions.

The [validation record](core-validation.md) reports 35 core tests passing in
each of two Transformers environments and 18 standalone tooling tests. It uses
tiny random-weight models and CPU callback harnesses. Actual checkpoint loading,
native CUDA execution, intelligible speech and real serving remain unverified.
Mimi prefix replay is a correctness baseline with quadratic total decoding work.

## `0.1.0`: validated basic model support

Keep the initial feature scope and fix issues revealed by real execution:

- Load a complete checkpoint with configuration and weight coverage checks.
- Compare text prefill, native AR logits, complete frames and depth outputs
  against the official implementation in a recorded Linux/CUDA environment.
- Generate intelligible short Chinese/English speech through `/v1/audio/speech`.
- Check EOS and token-limit termination, preserving the first/final frames.
- Validate incremental audio, terminal tails, consecutive requests, concurrency
  and cancellation without output misattribution or retained request state.
- Publish reproducible dependencies, checkpoint revision, hardware and commands.

Completion means a user can reproduce basic synthesis in the documented
environment. Performance optimization, voice cloning and multi-card support
are not prerequisites for this milestone. CPU-only evidence is insufficient.

## `0.2.0`: efficient streaming and service operation

This milestone has three code workstreams:

- **True streaming Mimi decoding:** carry transformer, convolution and
  upsampler state per request, replacing repeated full-prefix work. Match
  full decoding across arbitrary chunk boundaries and final-tail flushing.
  Retain prefix replay as a reference/fallback with explicit configuration.
- **Batched depth decoding:** decode compatible active requests together while
  preserving frame-local KV state, request-local RNG, different stopping times
  and complete-frame feedback. Test interleaving and batch regrouping.
- **Service reliability:** bound retained request state, isolate request errors,
  release state on completion/cancellation/failure and improve diagnostics.
  Define overflow/error behavior instead of silently dropping frames or audio.

Completion requires numerical/continuity regressions and real serving lifecycle
checks. Measure performance in a recorded environment before making speed or
memory claims. Do not rename prefix replay as true streaming or a per-request
Python loop as batched decoding. See the [next-task prompt](next-task-prompt.md).

## `0.3.0`: voice cloning and CFG

- Add reference audio/text and speaker conditioning following official semantics.
- Add multi-branch classifier-free guidance (CFG), including branch alignment,
  frame feedback and request state cleanup.
- Extend the speech API with explicit validation, limits and compatibility notes.
- Compare conditioning and guidance outputs against the reference and validate
  real-checkpoint synthesis with and without the new features.

Completion requires verified model behavior and usable API examples. Original
model capabilities are not automatically supported by this integration.

## `0.4.0`: advanced execution and developer tooling

- Evaluate CUDA graph capture, conditioning-aware prefix caching, TP/PP and
  quantization as separate capabilities with explicit configuration gates.
- Preserve actual text/speaker/reference conditioning in any cache identity;
  scheduler placeholder IDs alone are not a valid key.
- Add checkpoint preflight, startup diagnostics, environment reporting and
  repeatable runtime integration checks.
- Document supported combinations and migration steps; leave unverified
  execution paths disabled. Narrow or split this milestone if required by
  checkpoint or upstream constraints.

Developer tools, documentation and regression coverage should also improve in
earlier versions whenever necessary; they are not postponed wholesale to `0.4.0`.
Submit focused upstream changes once tests and checkpoint evidence are available.

## Version preservation

Maintain one source tree with annotated Git tags, checksum-protected source
archives, change history and per-version validation records. The current root
has no Git repository: preserve an archive before subsequent core changes and
create tags after importing the exported source into the published repository.
Do not create parallel `v0.1/`, `v0.2/` code directories. The
[release guide](releasing.md) describes export and version updates.
