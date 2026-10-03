# Breeze integration status

Core code now implements prompt/text conditioning, complete-frame depth
generation, same-step AR feedback, codec vocabulary/EOS handling, request-local
sampling, complete-frame transport and history-preserving Mimi decode.

CPU tests compare depth results with the official reference and exercise text
projection, frame alignment, stream continuity, payloads and cleanup. Native
callbacks execute in CPU harnesses; these tests do not run CUDA paged attention.

Remaining gates are a named checkpoint's full weight load, AR numerical parity,
intelligible audio and Linux/CUDA serving at concurrency 1/2/4/8. Verify empty
terminal chunks, cancellation and failure cleanup through the real scheduler.
After those pass, optimize Mimi state caching and depth batching, measure
performance, then extend voice/reference conditioning and CFG.

See the integration project's root `docs/validation.md` and `docs/roadmap.md`.
