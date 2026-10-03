"""Real CPU tensors and tiny models, with official Breeze as the oracle."""

import json
from pathlib import Path

import pytest
import torch
import transformers
from breeze_core.codec_stream import MimiPrefixDecoder
from breeze_core.configuration_breeze_base import BreezeDepthDecoderConfig
from breeze_core.configuration_breeze_tts import BreezeTTSConfig
from breeze_core.depth_decoder import BreezeDepthDecoderForCausalLM
from breeze_core.frame_decoder import DepthSampling, complete_frame, embed_audio_frame
from breeze_core.text_conditioning import (
    TextConditioning,
    fuse_text_features,
    speech_prompt_ids,
)
from transformers import MimiConfig, MimiModel

if transformers.__version__.startswith("4."):
    from breeze_reference.breeze import BreezeDepthDecoderForCausalLM as ReferenceDepth
    from breeze_reference.breeze_config import (
        BreezeDepthDecoderConfig as ReferenceDepthConfig,
    )


def depth_config(**overrides):
    values = dict(
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
    values.update(overrides)
    config = BreezeDepthDecoderConfig(**values)
    config._attn_implementation = "eager"
    return config


def test_depth_cached_logits_and_complete_frame_match_official():
    torch.manual_seed(12)
    config = depth_config()
    decoder = BreezeDepthDecoderForCausalLM(config).eval()
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/depth_reference.json").read_text()
    )
    decoder.load_state_dict(
        {name: torch.tensor(value) for name, value in fixture["weights"].items()},
        strict=True,
    )
    hidden = torch.tensor(fixture["hidden"])
    tokens = torch.tensor(fixture["tokens"])
    actual = decoder(
        input_ids=tokens,
        backbone_last_hidden_state=hidden,
        use_cache=True,
        cache_position=torch.arange(2),
    )
    torch.testing.assert_close(actual.logits, torch.tensor(fixture["logits"]))
    codes = complete_frame(
        decoder, tokens[:, 1], hidden, 8, DepthSampling(temperature=0)
    )
    torch.testing.assert_close(codes, torch.tensor(fixture["codes"]))
    if transformers.__version__.startswith("4."):
        reference_config = ReferenceDepthConfig(**config.to_dict())
        reference_config._attn_implementation = "eager"
        reference = ReferenceDepth(reference_config).eval()
        reference.load_state_dict(decoder.state_dict(), strict=True)
        expected = reference(
            input_ids=tokens,
            backbone_last_hidden_state=hidden,
            use_cache=True,
            cache_position=torch.arange(2),
        )
        torch.testing.assert_close(actual.logits, expected.logits)
    assert codes.shape == (2, 4) and codes.min() >= 0 and codes.max() < 8


def test_depth_rng_is_request_local_and_repeatable():
    decoder = BreezeDepthDecoderForCausalLM(depth_config()).eval()
    hidden = torch.ones(1, 24)
    one = complete_frame(
        decoder,
        torch.tensor([0]),
        hidden,
        8,
        generator=torch.Generator().manual_seed(42),
    )
    torch.rand(100)
    two = complete_frame(
        decoder,
        torch.tensor([0]),
        hidden,
        8,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(one, two)
    with pytest.raises(ValueError, match="reserved"):
        complete_frame(decoder, torch.tensor([11]), hidden, 8)


def test_all_codebooks_feed_back_including_zero():
    embedding = torch.nn.Embedding(4 * 11, 8)
    codes = torch.tensor([[0, 1, 2, 3]])
    actual = embed_audio_frame(embedding, codes, 11, 4)
    expected = sum(embedding(codes[:, index] + index * 11) for index in range(4))
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="complete"):
        embed_audio_frame(embedding, codes[:, :1], 11, 4)


def tiny_config(**kwargs):
    return BreezeTTSConfig(
        num_codebooks=4,
        vocab_size=11,
        text_vocab_size=32,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        depth_decoder_config=depth_config().to_dict(),
        codec_config={"model_type": "mimi", "codebook_size": 8, "num_quantizers": 4},
        **kwargs,
    )


def test_config_roundtrip_and_audio_eos():
    config = tiny_config()
    clone = BreezeTTSConfig(**config.to_dict())
    assert clone.codec_codebook_size == 8 and clone.eos_token_id == 11
    assert clone.codebook_eos_token_id == 0
    assert clone.get_text_config().vocab_size == 12
    assert clone.get_text_config().rope_parameters["rope_theta"] == clone.rope_theta
    with pytest.raises(ValueError, match="num_codebooks"):
        BreezeTTSConfig(num_codebooks=2)


def test_text_segments_independent_and_dimfusion_shapes():
    config = tiny_config(
        text_encoder_config={
            "model_type": "t5gemma2_text",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "sliding_window": 4,
            "layer_types": ["sliding_attention", "full_attention"],
        },
        text_encoder_proj_type="breeze_dimfusion",
        text_encoder_dimfusion_fuse_first_layer=True,
    )
    module = TextConditioning(config).eval()
    ids = torch.tensor([3, 4, 5, 6, 7])
    embeds, layers = module(ids, [2, 3])
    first, first_layers = module(ids[:2], [2])
    second, second_layers = module(ids[2:], [3])
    torch.testing.assert_close(embeds, torch.cat((first, second)))
    assert len(layers) == 2 and embeds.shape == (5, 24)
    for i in range(2):
        torch.testing.assert_close(
            layers[i], torch.cat((first_layers[i], second_layers[i]))
        )
    hidden = torch.randn(6, 24)
    fused = fuse_text_features(hidden, layers[1], torch.tensor([True] * 5 + [False]))
    torch.testing.assert_close(fused[:5, 12:], layers[1])
    torch.testing.assert_close(fused[:, :12], hidden[:, :12])
    torch.testing.assert_close(fused[5], hidden[5])
    with pytest.raises(ValueError, match="exactly"):
        module(ids, [4])


def mimi():
    config = MimiConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_filters=4,
        num_residual_layers=1,
        upsampling_ratios=[4, 4],
        upsample_groups=1,
        sampling_rate=160,
        frame_rate=5,
        codebook_size=8,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        num_quantizers=4,
        num_semantic_quantizers=1,
        sliding_window=8,
    )
    config._attn_implementation = "eager"
    return MimiModel(config).eval()


def test_mimi_split_decode_matches_whole_and_flushes_empty_terminal():
    torch.manual_seed(42)
    model = mimi()
    frames = torch.randint(0, 8, (8, 4))
    full = model.decode(frames.T.unsqueeze(0)).audio_values.reshape(-1).float()
    stream = MimiPrefixDecoder(model, 4, streaming=True)
    chunks = [
        stream.push("a", frames[:3]),
        stream.push("a", frames[3:6]),
        stream.push("a", frames[6:]),
        stream.push("a", frames[:0], finished=True),
    ]
    torch.testing.assert_close(torch.cat(chunks), full)
    assert "a" not in stream.states


def test_mimi_interleaved_requests_are_isolated_and_abort_cleans_up():
    model = mimi()
    stream = MimiPrefixDecoder(model, 4)
    a, b = torch.zeros((2, 4), dtype=torch.long), torch.ones((3, 4), dtype=torch.long)
    assert stream.push("a", a).numel() == 0
    assert stream.push("b", b).numel() == 0
    actual = stream.push("a", a[:0], finished=True)
    expected = model.decode(a.T.unsqueeze(0)).audio_values.reshape(-1).float()
    torch.testing.assert_close(actual, expected)
    assert set(stream.states) == {"b"}
    stream.discard(["b"])
    assert not stream.states
    assert stream.push("empty", a[:0], finished=True).numel() == 0
    assert not stream.states
    with pytest.raises(ValueError, match="invalid"):
        stream.push("bad", torch.full((1, 4), 8))


def test_prompt_template_uses_specials_and_speaker_prefix():
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens):
            self.calls.append((text, add_special_tokens))
            return [2, 3]

        def decode(self, ids, skip_special_tokens):
            assert not skip_special_tokens
            return "rendered"

    tokenizer = Tokenizer()
    assert speech_prompt_ids(tokenizer, "你好", "平静") == [2, 3]
    assert tokenizer.calls == [
        ("[S0]<ins_bos>平静<ins_eos>你好", True),
        ("rendered", False),
    ]
