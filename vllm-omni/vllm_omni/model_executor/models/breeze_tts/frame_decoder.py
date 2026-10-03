# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete one RVQ frame with the official depth attention and projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .depth_decoder import BreezeDepthDecoderForCausalLM


@dataclass(frozen=True)
class DepthSampling:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.temperature < float("inf") or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid Breeze depth sampling parameters")


def sample_depth(logits: torch.Tensor, settings: DepthSampling, generator: torch.Generator | None) -> torch.Tensor:
    if settings.temperature == 0:
        return logits.argmax(-1, keepdim=True)
    scores = logits.float() / settings.temperature
    if settings.top_k:
        threshold = scores.topk(min(settings.top_k, scores.shape[-1]), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if settings.top_p < 1:
        sorted_scores, indices = scores.sort(descending=True, dim=-1)
        remove = sorted_scores.softmax(-1).cumsum(-1) > settings.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scores = scores.masked_fill(torch.zeros_like(remove).scatter(1, indices, remove), float("-inf"))
    return torch.multinomial(scores.softmax(-1), 1, generator=generator)


@torch.inference_mode()
def complete_frame(
    decoder: BreezeDepthDecoderForCausalLM,
    first_code: torch.Tensor,
    backbone_hidden: torch.Tensor,
    codebook_size: int,
    settings: DepthSampling = DepthSampling(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return [batch, codebooks]; depth KV state lives for exactly one frame."""
    first = first_code.reshape(-1, 1).long()
    if backbone_hidden.shape != (first.shape[0], decoder.config.backbone_hidden_size):
        raise ValueError("One matching backbone hidden state is required per audio frame")
    if not 0 < codebook_size <= decoder.config.vocab_size or bool(((first < 0) | (first >= codebook_size)).any()):
        raise ValueError("First codebook contains a reserved token or EOS")
    if decoder.config.num_codebooks == 1:
        return first
    tokens = torch.cat((torch.zeros_like(first), first), dim=1)
    output = decoder(
        input_ids=tokens,
        backbone_last_hidden_state=backbone_hidden,
        cache_position=torch.arange(2, device=first.device),
        use_cache=True,
        logits_to_keep=1,
    )
    generated = [first]
    for index in range(decoder.config.num_codebooks - 1):
        token = sample_depth(output.logits[:, -1, :codebook_size], settings, generator)
        generated.append(token)
        if index + 1 < decoder.config.num_codebooks - 1:
            output = decoder(
                input_ids=token,
                past_key_values=output.past_key_values,
                use_cache=True,
                cache_position=torch.tensor([index + 2], device=first.device),
                logits_to_keep=1,
            )
    return torch.cat(generated, dim=1)


def embed_audio_frame(
    embedding: torch.nn.Embedding, codes: torch.Tensor, vocab_size: int, num_codebooks: int
) -> torch.Tensor:
    if codes.ndim != 2 or codes.shape[1] != num_codebooks:
        raise ValueError("AR feedback requires a complete RVQ frame")
    if bool(((codes < 0) | (codes >= vocab_size)).any()):
        raise ValueError("Audio frame contains invalid embedding indices")
    offsets = torch.arange(num_codebooks, device=codes.device) * vocab_size
    return embedding(codes.long() + offsets).sum(dim=1)
