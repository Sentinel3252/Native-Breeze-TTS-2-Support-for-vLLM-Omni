# Third-party notices

This project includes source from the following upstream repositories.
Existing copyright notices and component licenses remain applicable.

| Component | Source | Base revision | License location |
| --- | --- | --- | --- |
| vLLM-Omni | [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) | `58cb8de68bfef2993440b568907db60f448956b6` | [vllm-omni/LICENSE](vllm-omni/LICENSE) |
| Breeze reference inference | [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) | `ca632ce6c4d05f7985da4eab29b1a5d445b43f7b` | [breeze-tts/LICENSE](breeze-tts/LICENSE) |

The revisions identify the source bases, not the complete modified trees.
[project-manifest.json](project-manifest.json) lists the Breeze integration
overlay. Release archives contain the working-tree versions of source files,
including the integration changes.

Both upstream repositories license their main source under Apache-2.0.
The native `configuration_breeze_base.py`, `depth_decoder.py`, and
`t5gemma2_encoder.py` adapt the locally included Breeze reference implementation.
The original Sesame/Hugging Face notices are retained where present; the
T5Gemma 2 compatibility source is attributed to BreezeBlue. CPU oracle data
contains only tiny models initialized with random weights, not pretrained
model materials.
Individual vendored components and assets may carry additional notices;
retain those files when redistributing the source.

Breeze TTS 2 model weights, model-specific tokenizer/codec artifacts, and other
model materials are separate from the source-code license. The locally included
[MODEL_LICENSE](breeze-tts/MODEL_LICENSE) identifies its research and
non-commercial terms. Consult the [official model repository](https://huggingface.co/BreezeBlue/breeze-tts-2)
for the terms attached to the checkpoint you download. No model materials or
generated recordings are bundled by this project's source packager.

vLLM, Transformers, PyTorch, and other installed dependencies are not
redistributed by the source packager. Their own licenses apply.
