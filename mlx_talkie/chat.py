"""Chat-template helpers for Talkie instruction-tuned models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


def format_chat(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>{msg.content}<|end|>")
        elif msg.role == "user":
            parts.append(f"<|user|>{msg.content}<|end|>")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>{msg.content}<|end|>")
        else:
            raise ValueError(f"unsupported role: {msg.role}")
    parts.append("<|assistant|>")
    return "".join(parts)


def format_prompt(prompt: str) -> str:
    return f"<|user|>{prompt}<|end|><|assistant|>"


STOP_STRINGS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
    "<|endoftext|>",
)


def truncate_at_stop(text: str) -> tuple[str, bool]:
    positions = [text.find(s) for s in STOP_STRINGS if s in text]
    if not positions:
        return text, False
    stop_at = min(positions)
    return text[:stop_at], True
