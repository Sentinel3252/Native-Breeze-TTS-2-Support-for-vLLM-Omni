# Breeze TTS 2 native runtime

Stage 0 runs temporal AR with native vLLM attention. Its post-sampling hook
completes the entire RVQ frame from the current sampled codebook-0 token and
that step's hidden state. The complete frame is summed into the next AR input
and forwarded to Stage 1 for Mimi synthesis. Historical `breeze_depth_codec`
stage/class names are retained for configuration compatibility.

Text conditioning supports official speaker/instruction templates, segment
encoding and DimFusion projections. Depth uses the reference Breeze attention
and a frame-local KV cache. Codec zero is valid; backbone EOS is `vocab_size`.

Mimi keeps request history through full-prefix replay, with an optional held
tail for streaming. This preserves convolution context but has quadratic work
across chunks. Full-payload transport and empty terminal flushing are implemented.

The initial path requires eager execution, synchronous AR scheduling, disabled
prefix caching, TP/PP size 1 and no quantization/speculative decoding. Complete
checkpoints and CUDA serving remain unverified. The integration project's root
`docs/architecture.md` and `docs/validation.md` describe the design and gates.
