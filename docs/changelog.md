# Change history

This file records implemented changes. Future scopes are in the
[version roadmap](roadmap.md); planned versions have not been released.

## `0.1.0-dev` — Core implementation (development baseline)

Baseline recorded on 2026-10-03. No formal release or Git tag is claimed.

- Integrate native AR with text/instruction conditioning, text encoder
  projections and DimFusion.
- Complete each audio frame with reference depth attention and request-local
  sampling; feed all codebooks back into the temporal backbone.
- Handle explicit audio EOS, first/final frames and KV-preemption replay.
- Transport complete codebooks between stages with request isolation.
- Decode Mimi using complete-prefix replay, incremental waveform deltas and
  terminal-tail flushing; clean up request state.
- Add configuration/weight coverage guards, a speech API adapter and a
  development deployment configuration.
- Provide bilingual READMEs, setup/API/design documentation, contribution
  guidance, standalone speech client, source checks and source packaging.
- Record 35 CPU core tests passing with Transformers 4.57.3 and 5.10.1,
  and 18 standalone tooling tests passing. See [validation](core-validation.md).

Known limits: real-checkpoint CUDA execution and full serving are unverified;
Mimi replay has quadratic total decoding work; depth decoding runs per request.
Voice cloning and multi-branch CFG are unsupported. Initial execution requires
TP/PP=1, eager mode and disabled prefix caching; advanced paths remain restricted.
