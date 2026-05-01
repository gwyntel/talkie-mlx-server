#!/usr/bin/env python3
"""
OpenAI-compatible API server for Talkie 1930 13B IT (8-bit quantized).

Uses mlx_lm.load() which natively handles quantized weights (QuantizedLinear),
tiktoken tokenizer via TokenizerWrapper, and chat template rendering.

Supports both streaming (SSE) and non-streaming responses.

Endpoints:
    GET  /v1/models            - list models
    POST /v1/chat/completions  - OpenAI chat format (streaming + non-streaming)
    POST /v1/completions       - raw text completion

Usage:
    python server.py [--model-dir ./models/talkie-1930-13b-it-mlx-8bit] [--port 8080] [--host 127.0.0.1]
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

MAX_TOKENS_DEFAULT = 1024
TEMPERATURE_DEFAULT = 0.7
TOP_P_DEFAULT = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(prefix="chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# EOS suppression: Talkie IT tends to emit <|end|> (token 65536) after just
# 1-2 sentences, truncating responses. We bias down EOS logits for the first
# MIN_RESPONSE_TOKENS tokens so the model naturally writes longer replies.
# ---------------------------------------------------------------------------
_EOS_IDS = {65535, 65536, 65537, 65539}
_EOS_BIAS = -100.0
_MIN_RESPONSE_TOKENS = 80


def _make_eos_processor():
    """Return a logits processor that suppresses EOS tokens for the first
    _MIN_RESPONSE_TOKENS generated tokens. Uses a closure to track count.
    """
    count = [0]  # mutable via closure

    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        count[0] += 1
        if count[0] <= _MIN_RESPONSE_TOKENS:
            for eos_id in _EOS_IDS:
                logits[eos_id] = logits[eos_id] + _EOS_BIAS
        return logits

    return processor


def _generate_with_eos_suppress(prompt: str, max_tokens: int,
                                 temperature: float, top_p: float) -> str:
    """Generate with EOS suppression via logits_processors.

    Talkie IT emits <|end|> prematurely (after 10-20 tokens of output),
    producing very short 1-2 sentence responses. By biasing the EOS token
    logits down by -100 for the first 80 tokens, the model naturally produces
    multi-sentence, multi-paragraph Edwardian prose.
    """
    sampler = make_sampler(temp=temperature, top_p=top_p)
    processor = _make_eos_processor()
    return generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens,
                    sampler=sampler, logits_processors=[processor], verbose=False)


def _generate_tokens(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Generate full response, then yield it word-by-word for SSE streaming.

    Uses EOS suppression to get longer responses from Talkie IT.
    Newlines are preserved by splitting on word boundaries while keeping
    line breaks as separate chunks.
    """
    text = _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)

    # Split preserving newlines: each line break becomes its own chunk
    parts = []
    for paragraph in text.split("\n"):
        if paragraph:
            words = paragraph.split(" ")
            for i, word in enumerate(words):
                parts.append(word if i == 0 else " " + word)
        parts.append("\n")  # Preserve line breaks
    # Remove trailing newline if the original didn't end with one
    if parts and parts[-1] == "\n" and not text.endswith("\n"):
        parts.pop()
    # Also remove leading newline if empty first paragraph
    if parts and parts[0] == "\n":
        parts.pop(0)

    for part in parts:
        yield part


def _generate_full(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate full response (non-streaming) with EOS suppression."""
    return _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)


def _chat_completion_stream(body: dict):
    """Yield SSE chunks for a chat completion request."""
    messages_raw = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)
    completion_id = _make_id()

    prompt = _tokenizer.apply_chat_template(messages_raw, tokenize=False)

    # Generate full text, then stream it word-by-word
    sampler = make_sampler(temp=temperature, top_p=top_p)
    full_text = generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens,
                         sampler=sampler, verbose=False)

    # Stream tokens in chunks (word by word for readability)
    words = full_text.split(' ')
    for i, word in enumerate(words):
        token_text = word if i == 0 else ' ' + word
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with finish_reason
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _chat_completion(body: dict) -> dict:
    messages_raw = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)

    prompt = _tokenizer.apply_chat_template(messages_raw, tokenize=False)
    text = _generate_full(prompt, max_tokens, temperature, top_p)

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

    text = _generate_full(prompt, max_tokens, temperature, top_p)

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

    def _sse_response(self, generator):
        """Stream SSE chunks from a generator."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for chunk in generator:
            self.wfile.write(chunk.encode())
            self.wfile.flush()

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
            stream = body.get("stream", False)
            if stream:
                self._sse_response(_chat_completion_stream(body))
                elapsed = time.time() - t0
                mem = mx.get_peak_memory() / 1e9
                print(f"[talkie-server] chat/completions (stream) {elapsed:.1f}s, {mem:.1f} GB peak")
            else:
                result = _chat_completion(body)
                elapsed = time.time() - t0
                mem = mx.get_peak_memory() / 1e9
                n = len(result["choices"][0]["message"]["content"])
                print(f"[talkie-server] chat/completions {elapsed:.1f}s, {n} chars, {mem:.1f} GB peak")
                self._json_response(200, result)

        elif self.path in ("/v1/completions", "/completions"):
            result = _completion(body)
            elapsed = time.time() - t0
            n = len(result["choices"][0]["text"])
            mem = mx.get_peak_memory() / 1e9
            print(f"[talkie-server] completions {elapsed:.1f}s, {n} chars, {mem:.1f} GB peak")
            self._json_response(200, result)

        else:
            self._json_response(404, {"error": "not found"})

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
        default=".",
        help="Path to the Talkie MLX model directory (default: current directory)",
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

    print(f"[talkie-server] SSE streaming: enabled")
    print(f"[talkie-server] Serving on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[talkie-server] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
