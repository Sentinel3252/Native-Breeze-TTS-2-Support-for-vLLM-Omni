# Speech API usage

[Setup](getting-started.md) · [Example client](../examples/speech_client.py)

The integration registers an adapter for the vLLM-Omni speech endpoint.
It implements text and optional instruction conditioning using the official
speaker/template format. Full-checkpoint synthesis remains unverified.

## Non-streaming request

The `model` value must match the name advertised by your server. When serving
a local checkpoint without an alias, use the same checkpoint path:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/breeze-tts-2","input":"Hello from Breeze.","response_format":"wav","max_new_tokens":256}' \
  --output hello.wav
```

The standalone client validates a nonempty PCM WAV before publishing the
output file and reports its sample rate, channel count, and duration:

```bash
python examples/speech_client.py \
  --model /path/to/breeze-tts-2 --text "你好，Breeze。" \
  --output outputs/hello.wav
```

Existing output files are preserved unless `--force` is specified. Failed
HTTP responses and invalid audio do not leave a final output file.
Set `BREEZE_API_KEY` in your environment if the server requires a bearer token;
the client does not print it or write it to its report.

## Streaming request

The shared speech API defaults to SSE for `stream=true`. For raw bytes, choose
`stream_format=audio` explicitly. The example client selects raw PCM:

```bash
python examples/speech_client.py \
  --model /path/to/breeze-tts-2 --text "A short streaming request." \
  --stream --output outputs/hello.pcm
```

Equivalent request body:

```json
{
  "model": "/path/to/breeze-tts-2",
  "input": "A short streaming request.",
  "response_format": "pcm",
  "stream": true,
  "stream_format": "audio",
  "max_new_tokens": 256
}
```

Raw PCM has no file header. Obtain sample rate/channel count from the validated
codec configuration before playback. The client saves signed 16-bit PCM bytes
as supplied by the speech API; it does not guess a sample rate or wrap them in WAV.
Even-sized, nonempty PCM proves only basic transport validity, not speech quality.

## Parameters and limits

| Parameter | Integration behavior |
| --- | --- |
| `input` | Nonempty text; official `[S0]` template and tokenizer special tokens |
| `model` | Selects the server's advertised model |
| `max_new_tokens` | Applied through the shared AR token limit helper; low limits may truncate audio |
| `response_format`, `stream`, `stream_format` | Handled by shared speech response transport |
| `instructions` | Encoded inside `<ins_bos>...<ins_eos>` before the speech text |
| `voice` | Omit or use `S0`, `s0`, or `default`; other voices are rejected |
| `seed` | Seeds AR sampling and an independent per-request depth generator |
| `extra_params` | Supports `depth_temperature`, `depth_top_k`, `depth_top_p`, and `cfg_scale=1`; other keys are rejected |
| `language` | No separate language-control token; language follows the text/checkpoint |
| `ref_audio`, `ref_text`, `speaker_embedding`, `task_type` | Rejected until reference conditioning is implemented |

Depth defaults are temperature `0.9`, top-k `50`, and top-p `1.0`. Use
`"extra_params":{"depth_temperature":0}` for greedy depth comparison.
Only the backbone EOS at `vocab_size` stops generation; codec value `0`
remains valid. Multi-branch CFG is not implemented.

The shared protocol contains fields for other TTS models. Schema acceptance
does not imply Breeze implements their semantics. The standalone client sends
only the minimal supported request and transport fields.

## Timing

The client prints JSON with `elapsed_s`, `first_body_byte_s`, and `bytes`.
For WAV it also reports `audio_duration_s` and `rtf` (elapsed time divided by
audio duration). `first_body_byte_s` is measured at the HTTP client; headers,
container bytes, and buffering can precede playable audio. It is **not a TTFA
benchmark**. Model/codec correctness and streaming continuity need independent
validation before performance comparisons; see [validation](validation.md).
