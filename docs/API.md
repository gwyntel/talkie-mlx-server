# TalkieLM API Reference

Complete reference for the Talkie MLX Server's OpenAI-compatible API. The server runs at `http://127.0.0.1:8080` by default.

---

## Base URL

```
http://127.0.0.1:8080/v1
```

The server accepts requests with or without the `/v1` prefix — both `/v1/chat/completions` and `/chat/completions` work.

**Authentication:** None required. The server binds to localhost only. Set `api_key: local-no-key-needed` in your OpenAI client config.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |

---

## GET /v1/models

List the available model(s). The server serves a single model — the one loaded at startup.

### Request

```bash
curl http://127.0.0.1:8080/v1/models
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "talkie-1930-13b-it-mlx-8bit",
      "object": "model",
      "created": 1746123456,
      "owned_by": "talkie-lm"
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | Always `"list"` |
| `data` | array | List of model objects |
| `data[].id` | string | Model identifier (configurable via `--model-id` flag) |
| `data[].object` | string | Always `"model"` |
| `data[].created` | integer | Unix timestamp |
| `data[].owned_by` | string | Always `"talkie-lm"` |

---

## POST /v1/chat/completions

Generate a chat completion using the OpenAI messages format. Supports both streaming (SSE) and non-streaming responses.

### Request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "messages": [
      {"role": "system", "content": "You are a helpful Edwardian-era assistant."},
      {"role": "user", "content": "What is the wireless telegraph?"}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": false
  }'
```

### Request Body

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | — | Model ID (any value accepted; server uses loaded model) |
| `messages` | array | — | **Required.** Array of message objects |
| `messages[].role` | string | — | `"system"`, `"user"`, or `"assistant"` |
| `messages[].content` | string | — | Message text |
| `max_tokens` | integer | 1024 | Maximum tokens in the generated response |
| `temperature` | float | 0.7 | Sampling temperature (0.0–2.0) |
| `top_p` | float | 0.95 | Nucleus sampling threshold |
| `stream` | boolean | false | If `true`, return SSE stream instead of JSON |

### Non-Streaming Response

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1746123456,
  "model": "talkie-1930-13b-it-mlx-8bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I say, the wireless telegraph is perhaps the most marvellous invention of our age..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 87,
    "total_tokens": 0
  }
}
```

### Non-Streaming Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion ID (`chatcmpl-` prefix + 12 hex chars) |
| `object` | string | Always `"chat.completion"` |
| `created` | integer | Unix timestamp |
| `model` | string | Model identifier |
| `choices` | array | List of choices (always 1) |
| `choices[].index` | integer | Always `0` |
| `choices[].message.role` | string | Always `"assistant"` |
| `choices[].message.content` | string | Generated text |
| `choices[].finish_reason` | string | Always `"stop"` |
| `usage.prompt_tokens` | integer | Tokens in the prompt (after chat template) |
| `usage.completion_tokens` | integer | Tokens in the completion |
| `usage.total_tokens` | integer | Always `0` (not computed) |

### Streaming Response

When `stream: true`, the server returns `Content-Type: text/event-stream` with SSE chunks:

```
data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1746123456,"model":"talkie-1930-13b-it-mlx-8bit","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1746123456,"model":"talkie-1930-13b-it-mlx-8bit","choices":[{"index":0,"delta":{"content":" must"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1746123456,"model":"talkie-1930-13b-it-mlx-8bit","choices":[{"index":0,"delta":{"content":" say,"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-a1b2c3d4e5f6","object":"chat.completion.chunk","created":1746123456,"model":"talkie-1930-13b-it-mlx-8bit","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Streaming Chunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Same ID for all chunks in the stream |
| `object` | string | Always `"chat.completion.chunk"` |
| `choices[].delta.content` | string | New text for this chunk (word-level granularity) |
| `choices[].delta` | object | Empty `{}` on the final chunk |
| `choices[].finish_reason` | string \| null | `null` during generation, `"stop"` on final chunk |

**Important:** The `delta.content` field contains ONLY the new text for this chunk. Do not accumulate with previous content — this would cause duplication. Append each `delta.content` to build the full response.

### Streaming Behavior

The server generates the full response first, then streams it word-by-word:

1. **Generation phase:** The model generates the complete response (with EOS suppression for the first 80 tokens)
2. **Streaming phase:** The complete text is split into word-level chunks and streamed as SSE events
3. **Final chunk:** An empty delta with `finish_reason: "stop"` signals completion
4. **Termination:** `data: [DONE]\n\n` ends the stream

This means total latency is equal to generation time — the streaming doesn't reduce time-to-first-token. However, it provides progressive display in Discord.

### Python Example (Streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-no-key-needed")

response = client.chat.completions.create(
    model="talkie-1930-13b-it-mlx-8bit",
    messages=[
        {"role": "system", "content": "You are an Edwardian-era conversationalist."},
        {"role": "user", "content": "What do you think of aeroplanes?"}
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].finish_reason:
        print()  # newline at end
```

### Python Example (Non-Streaming)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-no-key-needed")

response = client.chat.completions.create(
    model="talkie-1930-13b-it-mlx-8bit",
    messages=[
        {"role": "user", "content": "Tell me about the motor-car."}
    ],
    max_tokens=500,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

---

## POST /v1/completions

Generate a text completion from a raw prompt (no chat template). The chat template is NOT applied — the prompt is passed directly to the model.

### Request

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "prompt": "The wireless telegraph is perhaps the most",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

### Request Body

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | — | Model ID (any value accepted) |
| `prompt` | string | — | **Required.** Raw text prompt |
| `max_tokens` | integer | 1024 | Maximum tokens in the completion |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.95 | Nucleus sampling threshold |

### Response

```json
{
  "id": "cmpl-a1b2c3d4e5f6",
  "object": "text_completion",
  "created": 1746123456,
  "model": "talkie-1930-13b-it-mlx-8bit",
  "choices": [
    {
      "index": 0,
      "text": " remarkable invention of the modern age...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 27,
    "total_tokens": 0
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Completion ID with `cmpl-` prefix |
| `object` | string | Always `"text_completion"` |
| `choices[].text` | string | Generated text (no chat template applied) |
| `choices[].finish_reason` | string | Always `"stop"` |

**Note:** The `/v1/completions` endpoint does NOT stream. Only `/v1/chat/completions` supports streaming.

---

## EOS Suppression

All generation (both endpoints, both streaming and non-streaming) applies EOS suppression automatically:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Suppressed tokens | `{65535, 65536, 65537, 65539}` | All model EOS token IDs |
| Bias | -100.0 | Added to EOS token logits |
| Duration | First 80 generated tokens | After 80 tokens, EOS is allowed naturally |

This is not configurable via the API. It is hardcoded in `server.py` and cannot be turned off.

**Why this exists:** The Talkie IT model tends to emit `<|end|>` (token 65536) after just 1–2 sentences, dramatically truncating responses. The -100 bias effectively makes EOS impossible for the first 80 tokens, forcing the model to write at least a paragraph.

---

## Error Responses

### 400 Bad Request

```json
{"error": "invalid JSON"}
```

Returned when the request body is not valid JSON.

### 404 Not Found

```json
{"error": "not found"}
```

Returned for unknown endpoints or unsupported HTTP methods.

### 500 Internal Server Error

```json
{"error": "internal server error"}
```

Returned when generation fails (model error, OOM, etc.). Check server logs for details.

**Note:** SSE streaming errors cannot return a JSON error response because headers have already been sent. The connection simply closes. Check server logs for the exception details.

---

## Chat Template

The Talkie IT model uses custom chat template tokens:

| Token | Role | Description |
|-------|------|-------------|
| `segmentor` | System turn | Prefix for system messages |
| `uctor` | User turn | Prefix for user messages |
| `ECTOR` | Assistant turn | Prefix for assistant messages |
| `<|end|>` | End of turn | Suffix for all turns |

**Example rendered template:**

```
_segmentorYou are an Edwardian-era conversationalist...<|end|>uctorWhat is the wireless telegraph?<|end|>ECTOR
```

The server handles this automatically via `_tokenizer.apply_chat_template(messages, tokenize=False)`. You do not need to format these tokens manually — just pass standard OpenAI-format messages.

**For raw completions** (`/v1/completions`), the chat template is NOT applied. You must format the prompt manually if you want chat-style behavior.

---

## Rate Limits & Concurrency

The server is **single-threaded** (Python `http.server`). It processes one request at a time. While a generation is in progress, other requests will block.

**Practical implications:**
- Only one Discord user can get a response at a time
- If multiple people @mention the bot simultaneously, requests are serialized
- The server does NOT implement rate limiting, queuing, or request prioritization

**Typical generation times (M5 Pro 48 GB, 8-bit):**

| Response length | Generation time |
|-----------------|----------------|
| ~50 tokens | ~3 seconds |
| ~100 tokens | ~6 seconds |
| ~500 tokens | ~30 seconds |
| ~1000 tokens | ~60 seconds |

---

## Compatibility Notes

### OpenAI SDK (Python)

Fully compatible with the `openai` Python package:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-no-key-needed",
)
```

### Parameters Not Supported

The following OpenAI API parameters are **not supported** and will be silently ignored:

| Parameter | Notes |
|-----------|-------|
| `n` | Only 1 choice is generated |
| `stop` | Not supported (EOS suppression handles stopping) |
| `presence_penalty` | Not implemented |
| `frequency_penalty` | Not implemented |
| `logit_bias` | Not implemented (EOS bias is hardcoded) |
| `user` | Not used |
| `seed` | Not implemented (no deterministic sampling) |
| `response_format` | Not implemented |
| `tools` / `functions` | Not supported (Talkie is not trained for tool use) |
| `logprobs` | Not implemented |

### Differences from Standard OpenAI API

| Feature | OpenAI Standard | Talkie Server |
|----------|----------------|---------------|
| Multiple models | Yes | No — single model loaded at startup |
| Streaming granularity | Token-level | Word-level (batch-then-stream) |
| `usage.total_tokens` | Sum of prompt + completion | Always `0` |
| SSE for `/v1/completions` | Supported | Not supported (non-streaming only) |
| Model parameter | Selects model | Accepted but ignored (uses loaded model) |
