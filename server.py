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

DEBUG LOGGING:
    Set LOG_LEVEL=DEBUG env var or --log-level flag for verbose request/response logging.
    Logs to stderr by default; redirect to file via your launcher (launchd, systemd, etc).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

# ---------------------------------------------------------------------------
# Logging setup — configured before anything else so all paths log
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("talkie-server")
REQUEST_LOG = logging.getLogger("talkie-server.requests")

# ---------------------------------------------------------------------------
# Globals (set in main())
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_model_dir = None
_model_id = "talkie-1930-13b-it-mlx-8bit"

MAX_TOKENS_DEFAULT=1024
TEMPERATURE_DEFAULT = 0.7
TOP_P_DEFAULT = 0.95

# ---------------------------------------------------------------------------
# EOS suppression: Talkie IT tends to emit <|end|> (token 65536) after
# just 1-2 sentences, truncating responses. We bias down EOS logits for
# the first MIN_RESPONSE_TOKENS tokens so the model writes longer replies.
# ---------------------------------------------------------------------------
_EOS_IDS = {65535, 65536, 65537, 65539}
_EOS_BIAS = -100.0
_MIN_RESPONSE_TOKENS=80


def _make_eos_processor():
    """Return a logits processor that suppresses EOS tokens for the first
    _MIN_RESPONSE_TOKENS generated tokens. Uses a closure to track count.

    Each call returns a FRESH processor (don't reuse across requests).
    """
    count = [0]

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

    Uses mlx_lm.generate() which handles prefix-caching, KV cache,
    and speculative decoding correctly — unlike a hand-rolled loop.
    """
    sampler = make_sampler(temp=temperature, top_p=top_p)
    processor = _make_eos_processor()
    t0 = time.perf_counter()
    result = generate(_model, _tokenizer, prompt=prompt, max_tokens=max_tokens,
                      sampler=sampler, logits_processors=[processor], verbose=False)
    elapsed = time.perf_counter() - t0
    gen_len = len(result)
    logger.info(
        f"[GENERATE] {elapsed:.2f}s, {gen_len} chars, "
        f"max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, "
        f"prompt[:80]={repr(prompt[:80])}"
    )
    REQUEST_LOG.debug(f"[GENERATE] full_output={repr(result)}")
    return result


def _generate_tokens(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Generate full response, then yield it word-by-word for SSE streaming.

    Yields strings (words or \n) that should be concatenated to reconstruct
    the original generated text character-for-character.
    """
    text = _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)
    logger.info(f"[STREAM] generated {len(text)} chars, will yield ~{text.count(' ') + text.count(chr(10))} chunks")

    # Split preserving newlines so Discord sees paragraph breaks too
    parts = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line:
            words = line.split(" ")
            for j, word in enumerate(words):
                parts.append(word if j == 0 else " " + word)
        if i < len(lines) - 1:
            parts.append("\n")

    REQUEST_LOG.debug(f"[STREAM] chunks={parts}")
    for part in parts:
        yield part


def _generate_full(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate full response (non-streaming) with EOS suppression."""
    return _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)


# ---------------------------------------------------------------------------
# OpenAI-compatible response builders
# ---------------------------------------------------------------------------

def _make_id(prefix="chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _chat_completion_stream(body: dict):
    """Yield SSE chunks for a chat completion request.

    CRITICAL FIX: previously called generate() directly without EOS
    suppression, so streaming responses were truncated to 1-2 sentences
    while non-streaming was fine. Now uses _generate_with_eos_suppress
    consistently.
    """
    req_id = _make_id()
    messages_raw = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)

    prompt = _tokenizer.apply_chat_template(messages_raw, tokenize=False)
    prompt_tokens = len(_tokenizer.encode(prompt))

    REQUEST_LOG.debug(
        f"[CHAT_STREAM] req_id={req_id} prompt_tokens={prompt_tokens} "
        f"max_tokens={max_tokens} temp={temperature} top_p={top_p} "
        f"messages={json.dumps(messages_raw, ensure_ascii=False)[:500]}"
    )
    REQUEST_LOG.info(
        f"[CHAT_STREAM] req_id={req_id} starting generation "
        f"prompt_tokens={prompt_tokens} max_tokens={max_tokens}"
    )

    try:
        full_text = _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)
    except Exception:
        logger.exception(f"[CHAT_STREAM] req_id={req_id} generation FAILED")
        REQUEST_LOG.error(f"[CHAT_STREAM] req_id={req_id} generation FAILED: {traceback.format_exc()}")
        raise

    REQUEST_LOG.info(
        f"[CHAT_STREAM] req_id={req_id} generation done, "
        f"output_len={len(full_text)} output[:120]={repr(full_text[:120])}"
    )
    REQUEST_LOG.debug(f"[CHAT_STREAM] req_id={req_id} FULL_OUTPUT={repr(full_text)}")

    # Stream word-by-word
    parts = []
    lines = full_text.split("\n")
    for i, line in enumerate(lines):
        if line:
            words = line.split(" ")
            for j, word in enumerate(words):
                parts.append(word if j == 0 else " " + word)
        if i < len(lines) - 1:
            parts.append("\n")

    chunks_sent = 0
    chars_sent = 0
    for part in parts:
        token_text = part
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None,
            }],
        }
        chunk_str = f"data: {json.dumps(chunk)}\n\n"
        yield chunk_str
        chunks_sent += 1
        chars_sent += len(token_text)

    # Final chunk
    final = {
        "id": req_id,
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

    REQUEST_LOG.info(
        f"[CHAT_STREAM] req_id={req_id} DONE chunks_sent={chunks_sent} "
        f"chars_sent={chars_sent} total_output={len(full_text)}"
    )


def _chat_completion(body: dict) -> dict:
    req_id = _make_id()
    messages_raw = body.get("messages", [])
    max_tokens = body.get("max_tokens", MAX_TOKENS_DEFAULT)
    temperature = body.get("temperature", TEMPERATURE_DEFAULT)
    top_p = body.get("top_p", TOP_P_DEFAULT)

    prompt = _tokenizer.apply_chat_template(messages_raw, tokenize=False)

    REQUEST_LOG.debug(
        f"[CHAT] req_id={req_id} max_tokens={max_tokens} temp={temperature} "
        f"messages={json.dumps(messages_raw, ensure_ascii=False)[:500]}"
    )

    try:
        text = _generate_full(prompt, max_tokens, temperature, top_p)
    except Exception:
        logger.exception(f"[CHAT] req_id={req_id} generation FAILED")
        raise

    REQUEST_LOG.info(
        f"[CHAT] req_id={req_id} output_len={len(text)} "
        f"output[:200]={repr(text[:200])}"
    )
    REQUEST_LOG.debug(f"[CHAT] req_id={req_id} FULL_OUTPUT={repr(text)}")

    return {
        "id": req_id,
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
            "prompt_tokens": len(_tokenizer.encode(prompt)),
            "completion_tokens": len(_tokenizer.encode(text)),
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
            "prompt_tokens": len(_tokenizer.encode(prompt)),
            "completion_tokens": len(_tokenizer.encode(text)),
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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as e:
            REQUEST_LOG.warning(f"[RESPONSE] client disconnected before JSON body: {e}")

    def _sse_response(self, generator):
        """Stream SSE chunks from a generator. Handle BrokenPipe gracefully."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        chunks_sent = 0
        try:
            for chunk in generator:
                try:
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
                    chunks_sent += 1
                except (BrokenPipeError, ConnectionResetError) as e:
                    REQUEST_LOG.warning(
                        f"[SSE] client disconnected after {chunks_sent} chunks: {e}"
                    )
                    break
        except Exception:
            logger.exception(f"[SSE] generator raised exception after {chunks_sent} chunks")
            raise

    def do_GET(self):
        client = self.client_address[0] if self.client_address else "unknown"
        REQUEST_LOG.info(f"[GET] {self.path} from {client}")
        if self.path in ("/v1/models", "/models"):
            self._json_response(200, _models())
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        client = self.client_address[0] if self.client_address else "unknown"
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            REQUEST_LOG.warning(f"[POST] invalid JSON from {client}")
            self._json_response(400, {"error": "invalid JSON"})
            return

        t0 = time.time()
        REQUEST_LOG.info(
            f"[POST] {self.path} from {client} "
            f"stream={body.get('stream', False)} "
            f"max_tokens={body.get('max_tokens', MAX_TOKENS_DEFAULT)}"
        )
        REQUEST_LOG.debug(
            f"[POST_BODY] {self.path} body={json.dumps(body, ensure_ascii=False)[:2000]}"
        )

        try:
            if self.path in ("/v1/chat/completions", "/chat/completions"):
                stream = body.get("stream", False)
                if stream:
                    try:
                        self._sse_response(_chat_completion_stream(body))
                    except Exception:
                        # SSE errors can't send a normal JSON error (headers already sent),
                        # just log it and let the connection close.
                        logger.exception("[POST] SSE stream failed")
                        REQUEST_LOG.error(f"[POST] SSE stream failed: {traceback.format_exc()}")
                    finally:
                        elapsed = time.time() - t0
                        mem = mx.get_peak_memory() / 1e9
                        logger.info(
                            f"[POST] chat/completions (stream) total_time={elapsed:.2f}s "
                            f"peak_mem={mem:.1f}GB"
                        )
                else:
                    result = _chat_completion(body)
                    elapsed = time.time() - t0
                    mem = mx.get_peak_memory() / 1e9
                    n = len(result["choices"][0]["message"]["content"])
                    logger.info(
                        f"[POST] chat/completions time={elapsed:.2f}s chars={n} "
                        f"peak_mem={mem:.1f}GB"
                    )
                    self._json_response(200, result)

            elif self.path in ("/v1/completions", "/completions"):
                result = _completion(body)
                elapsed = time.time() - t0
                n = len(result["choices"][0]["text"])
                mem = mx.get_peak_memory() / 1e9
                logger.info(
                    f"[POST] completions time={elapsed:.2f}s chars={n} "
                    f"peak_mem={mem:.1f}GB"
                )
                self._json_response(200, result)

            else:
                REQUEST_LOG.warning(f"[POST] unknown path {self.path}")
                self._json_response(404, {"error": "not found"})

        except Exception:
            logger.exception(f"[POST] UNHANDLED EXCEPTION on {self.path}")
            REQUEST_LOG.error(
                f"[POST] UNHANDLED EXCEPTION on {self.path}: {traceback.format_exc()}"
            )
            self._json_response(500, {"error": "internal server error"})

    def log_message(self, fmt, *args):
        pass  # we use our own logging


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
    parser.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"],
                        default=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        help="Logging level (also set via LOG_LEVEL env var)")
    args = parser.parse_args()

    # Reconfigure logging level if flag differs from env
    if args.log_level != LOG_LEVEL:
        logging.getLogger("talkie-server").setLevel(getattr(logging, args.log_level))
        logging.getLogger("talkie-server.requests").setLevel(getattr(logging, args.log_level))

    _model_dir = str(Path(args.model_dir).resolve())
    _model_id = args.model_id

    logger.info(f"Loading model from {_model_dir} ...")
    logger.info(f"Log level: {args.log_level}")
    _model, _tokenizer = load(_model_dir)
    logger.info(f"Loaded. Peak: {mx.get_peak_memory() / 1e9:.1f} GB")
    logger.info(f"Context window: {_model.args.max_seq_len} tokens")
    logger.info(f"EOS token ids: {_tokenizer.eos_token_ids}")
    logger.info(f"EOS bias: {_EOS_BIAS} for first {_MIN_RESPONSE_TOKENS} tokens")

    # Detect quantization
    for name, mod in _model.named_modules():
        if "QuantizedLinear" in type(mod).__name__:
            logger.info(f"Quantization: {mod.bits}-bit (group_size={mod.group_size}, mode={mod.mode})")
            break
    else:
        logger.info("Quantization: none (BF16/F16)")

    logger.info(f"SSE streaming: enabled")
    logger.info(f"Serving on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
