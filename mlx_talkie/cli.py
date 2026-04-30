"""Small CLI for MLX Talkie inference."""

from __future__ import annotations

import argparse

from .chat import Message, format_prompt
from .generate import GenerationConfig, Talkie


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Talkie 13B on MLX")
    parser.add_argument("prompt", nargs="?", help="prompt/user message")
    parser.add_argument("--model-dir", default="mlx/talkie-1930-13b-it-mlx")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw", action="store_true", help="do not wrap prompt in chat template")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt or input("User: ")
    cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    talkie = Talkie(args.model_dir)
    text_prompt = prompt if args.raw else format_prompt(prompt)

    if args.stream:
        for piece in talkie.stream(text_prompt, cfg):
            print(piece, end="", flush=True)
        print()
    else:
        print(talkie.generate(text_prompt, cfg))


if __name__ == "__main__":
    main()
