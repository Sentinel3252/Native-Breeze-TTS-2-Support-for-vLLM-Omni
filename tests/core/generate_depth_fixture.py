"""Generate a tiny, random-weight oracle using the unmodified official model.

Run in the Transformers 4.57.3 reference environment. This fixture contains no
pretrained model material. It is needed because the official 4.x model cannot
be instantiated under Transformers 5's RoPE API.
"""

import json
from pathlib import Path

import conftest  # noqa: F401
import torch
from breeze_reference.breeze import BreezeDepthDecoderForCausalLM
from breeze_reference.breeze_base_config import BreezeDepthDecoderConfig

torch.manual_seed(12)
config = BreezeDepthDecoderConfig(
    num_codebooks=4,
    vocab_size=11,
    backbone_hidden_size=24,
    audio_embed_size=16,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=5,
)
config._attn_implementation = "eager"
model = BreezeDepthDecoderForCausalLM(config).eval()
hidden = torch.randn(2, 24)
tokens = torch.tensor([[0, 0], [0, 6]])
with torch.inference_mode():
    logits = model(
        input_ids=tokens, backbone_last_hidden_state=hidden, use_cache=False
    ).logits
    full = tokens.clone()
    for _ in range(3):
        output = model(
            input_ids=full, backbone_last_hidden_state=hidden, use_cache=False
        )
        full = torch.cat(
            (full, output.logits[:, -1, :8].argmax(-1, keepdim=True)), dim=1
        )
fixture = {
    "origin": "Official Breeze ca632ce6; Transformers 4.57.3; torch CPU float32; random seed 12",
    "weights": {name: tensor.tolist() for name, tensor in model.state_dict().items()},
    "hidden": hidden.tolist(),
    "tokens": tokens.tolist(),
    "logits": logits.tolist(),
    "codes": full[:, 1:].tolist(),
}
path = Path(__file__).with_name("fixtures") / "depth_reference.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
print(f"Wrote reference fixture: {path.name}")
