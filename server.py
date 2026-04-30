#!/usr/bin/env python3
"""
OpenAI-compatible API server for Talkie 1930 13B IT (8-bit quantized).

Uses mlx_lm.load() which natively handles quantized weights (QuantizedLinear),
tiktoken tokenizer via TokenizerWrapper, and chat template rendering.

Endpoints:
    GET  /v1/models            - list models
    POST /v1/chat/completions  - OpenAI chat format
    POST /v1/completions       - raw text completion

Usage:
    python server.py [--model-dir ./models/talkie-1930-13b-it-mlx-8bit] [--port 8080] [--host 127.0.0.1]

Memory: ~15 GB at 2K ctx, ~18 GB at 4K ctx on M5 Pro 48GB.
Context: Set via max_seq_len in model config.json (default 4096, trained at 2048).
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

# ---------------------------------------------------------------------------
# Globals (set in main())
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_model_dir = None
_model_id = "talkie-1930-13b-it-mlx-8bit"

MAX_TOKENS_DEFAULT = 200
TEMPERATURE_DEFAULT = 0.7
TOP_P_DEFAULT = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix="chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _chat_completion(body: dict) -> dict:
    messages_raw = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)

    # Render chat template via tokenizer
    prompt = _tokenizer.apply_chat_template(messages_raw, tokenize=False)

    sampler = make_sampler(temp=temperature, top_p=top_p)
    text = generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens,
                    sampler=sampler, verbose=False)

    return {
        "id": _make_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _completion(body: dict) -> dict:
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)

    sampler = make_sampler(temp=temperature, top_p=top_p)
    text = generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens,
                    sampler=sampler, verbose=False)

    return {
        "id": _make_id("cmpl"),
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": _model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "talkie-lm",
                "max_context": _model.args.max_seq_len,
            }
        ],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json_response(200, _models())
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON"})
            return

        t0 = time.time()
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            result = _chat_completion(body)
        elif self.path in ("/v1/completions", "/completions"):
            result = _completion(body)
        else:
            self._json_response(404, {"error": "not found"})
            return

        elapsed = time.time() - t0
        mem = mx.get_peak_memory() / 1e9
        content = result["choices"][0].get("message", result["choices"][0]).get(
            "content", result["choices"][0].get("text", ""))
        print(f"[talkie-server] {self.path} {elapsed:.1f}s, {len(content)} chars, {mem:.1f} GB peak")

        self._json_response(200, result)

    def log_message(self, fmt, *args):
        pass  # suppress default logging


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _model, _tokenizer, _model_dir, _model_id

    parser = argparse.ArgumentParser(description="Talkie 8-bit OpenAI-compatible server")
    parser.add_argument(
        "--model-dir",
        default="./models/talkie-1930-13b-it-mlx-8bit",
        help="Path to the Talkie MLX model directory (default: ./models/talkie-1930-13b-it-mlx-8bit)",
    )
    parser.add_argument("--model-id", default="talkie-1930-13b-it-mlx-8bit",
                        help="Model identifier returned by the API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    _model_dir = str(Path(args.model_dir).resolve())
    _model_id = args.model_id

    print(f"[talkie-server] Loading model from {_model_dir} ...")
    _model, _tokenizer = load(_model_dir)
    print(f"[talkie-server] Loaded. Peak: {mx.get_peak_memory() / 1e9:.1f} GB")
    print(f"[talkie-server] Context window: {_model.args.max_seq_len} tokens")

    # Detect quantization
    for name, mod in _model.named_modules():
        if "QuantizedLinear" in type(mod).__name__:
            print(f"[talkie-server] Quantization: {mod.bits}-bit (group_size={mod.group_size}, mode={mod.mode})")
            break
    else:
        print("[talkie-server] Quantization: none (BF16/F16)")

    print(f"[talkie-server] Serving on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[talkie-server] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
