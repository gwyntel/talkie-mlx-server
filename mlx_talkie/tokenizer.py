"""Talkie tokenizer helpers for MLX inference."""

from __future__ import annotations

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

_PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

BASE_VOCAB_SIZE = 65536
IT_VOCAB_SIZE = BASE_VOCAB_SIZE + 4

BASE_SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
}

IT_SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
    "<|end|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>": BASE_VOCAB_SIZE + 3,
}


def build_tokenizer(vocab_path: str | Path, style: str = "it") -> tiktoken.Encoding:
    """Build the Talkie tiktoken tokenizer from ``vocab.txt``."""
    mergeable_ranks = load_tiktoken_bpe(str(vocab_path))
    mergeable_ranks = {
        k: v for k, v in mergeable_ranks.items() if v < BASE_VOCAB_SIZE - 1
    }

    if style == "it":
        special_tokens = dict(IT_SPECIAL_TOKENS)
        name = "talkie-it"
    else:
        special_tokens = dict(BASE_SPECIAL_TOKENS)
        name = "talkie-base"

    return tiktoken.Encoding(
        name=name,
        pat_str=_PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
