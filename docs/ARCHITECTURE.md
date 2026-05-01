# TalkieLM Architecture

System design, data flow, and component-level documentation for the TalkieLM Discord bot.

---

## High-Level Architecture

TalkieLM is a **dual-service system** running two Python processes on an Apple Silicon Mac:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Apple Silicon Mac                             │
│                                                                      │
│  ┌────────────────────┐         ┌─────────────────────────────────┐ │
│  │   llmcord.py        │         │   server.py                     │ │
│  │   (Discord Bridge)  │         │   (MLX Inference Server)       │ │
│  │                     │         │                                 │ │
│  │  • Discord.py bot   │  HTTP   │  • OpenAI-compatible API        │ │
│  │  • Message chain    │────────►│  • /v1/chat/completions         │ │
│  │    builder          │ (SSE   │  • /v1/completions               │ │
│  │  • Mention↔era      │ stream) │  • /v1/models                   │ │
│  │    translator       │         │                                 │ │
│  │  • Streaming embed  │         │  ┌───────────────────────────┐ │ │
│  │    renderer          │         │  │ Talkie 1930 13B IT        │ │ │
│  │  • Context mgmt     │         │  │ (8-bit MLX quantized)     │ │ │
│  │  • Permission check │         │  │ ~15 GB RAM, ~17 tok/s    │ │ │
│  │                     │         │  │ 4096 token context        │ │ │
│  └────────────────────┘         │  └───────────────────────────┘ │ │
│                                  └─────────────────────────────────┘ │
│                                                                      │
│  Launchd manages both processes:                                     │
│  • com.gwyntel.talkie-server.plist → server.py                     │
│  • com.gwyntel.llmcord.plist       → llmcord.py                    │
└──────────────────────────────────────────────────────────────────────┘
         │
         │ Discord Gateway (WebSocket)
         ▼
┌────────────────────┐
│    Discord         │
│  • Guild channels  │
│  • Direct messages │
│  • Reply threads   │
└────────────────────┘
```

---

## Component Details

### 1. Talkie MLX Inference Server (`server.py`)

A lightweight HTTP server that loads the Talkie model and serves OpenAI-compatible inference.

**Key characteristics:**
- Built on Python's `http.server` (not FastAPI/Flask — intentional for minimal deps)
- Loads model at startup via `mlx_lm.load()` (~5 seconds)
- Handles both streaming (SSE) and non-streaming responses
- Runs on `127.0.0.1:8080` by default (localhost only, no auth needed)

**Critical subsystem — EOS suppression:**

The Talkie IT model tends to emit `<|end|>` (token 65536) after just 1–2 sentences, dramatically truncating responses. The server applies a **logits processor** that biases EOS tokens by -100 for the first 80 generated tokens:

```python
_EOS_IDS = {65535, 65536, 65537, 65539}
_EOS_BIAS = -100.0
_MIN_RESPONSE_TOKENS = 80

def _make_eos_processor():
    count = [0]
    def processor(tokens, logits):
        count[0] += 1
        if count[0] <= _MIN_RESPONSE_TOKENS:
            for eos_id in _EOS_IDS:
                logits[eos_id] += _EOS_BIAS
        return logits
    return processor
```

This is the single most important quality-of-service feature. Without it, every response is 1–2 sentences long.

**Streaming architecture:**

The server uses **batch-then-stream** — it generates the complete response, then streams it word-by-word as SSE chunks. True token-by-token streaming from MLX with the Talkie custom model is complex; this approach gives the streaming UX while keeping the implementation simple.

```
Request arrives
     │
     ▼
Generate FULL response via mlx_lm.generate() + EOS suppression
     │
     ▼
Split into word-level chunks (preserving newlines)
     │
     ▼
Stream as SSE: data: {"choices":[{"delta":{"content":"word"}}]}\n\n
     │
     ▼
Final chunk: finish_reason="stop", then data: [DONE]
```

**Startup sequence:**

```
server.py main()
  ├── Parse CLI args (--model-dir, --port, --host, --log-level)
  ├── Configure logging (stderr + level from env/flag)
  ├── mlx_lm.load(model_dir) → (_model, _tokenizer)
  │   ├── Reads config.json for model architecture
  │   ├── Loads 8-bit quantized weights via QuantizedLinear
  │   └── Wraps tiktoken tokenizer via TokenizerWrapper
  ├── Log model info (context window, quantization, EOS tokens)
  └── HTTPServer((host, port), Handler).serve_forever()
```

### 2. llmcord Discord Bridge (`llmcord.py`)

A Discord bot that proxies messages to the inference server and streams responses back. Based on [jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord) with Talkie-specific enhancements.

**Key responsibilities:**

1. **Message gating** — Decides which messages to respond to
2. **Context building** — Constructs the OpenAI-format messages array
3. **Mention translation** — Converts Discord `<@ID>` to era-appropriate format
4. **Streaming rendering** — Updates Discord embeds in real-time as tokens arrive
5. **Context management** — Slash commands for showing/clearing/managing context

**Message flow (detailed):**

```
Discord message arrives (on_message event)
     │
     ▼
┌─────────────────────┐
│ Gating: Should the  │──No──► return (ignore message)
│ bot respond?        │
└────────┬────────────┘
         │ Yes
         ▼
┌─────────────────────┐
│ Permission check:   │──Blocked──► return
│ user/channel/role   │
└────────┬────────────┘
         │ Allowed
         ▼
┌─────────────────────┐
│ Build message chain │──► Walk reply chain up to max_messages (10)
│ (MsgNode traversal) │──► Each node cached in msg_nodes dict
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Translate mentions  │──► Discord <@123> → "Correspondent No. 123:"
│ for model input     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Add system prompt   │──► {date} and {time} placeholders filled
│                     │──► Appended as role="system" message
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Call OpenAI API     │──► AsyncOpenAI.chat.completions.create(stream=True)
│ (SSE streaming)     │──► Base URL: http://127.0.0.1:8080/v1
└────────┬────────────┘
         │
         ▼  (SSE chunks arriving)
┌─────────────────────┐
│ Stream to Discord   │──► Accumulate delta.content from each chunk
│ embed               │──► Edit embed every 1 second (EDIT_DELAY_SECONDS)
│                     │──► Convert "Correspondent No. N:" back to <@N> pings
│                     │──► Orange embed + ⚪ = still generating
│                     │──► Green embed = complete
│                     │──► Split into multiple messages if >4096 chars
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Cache response      │──► Store in msg_nodes for future context
│                     │──► Evict oldest nodes if cache > 500
└─────────────────────┘
```

### 3. Message Node Cache (`MsgNode`)

llmcord maintains an in-memory cache of processed messages as `MsgNode` objects:

```python
@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"
    text: Optional[str] = None                   # Processed message text
    images: list[dict] = field(default_factory=list)  # Base64 encoded images
    has_bad_attachments: bool = False             # Unsupported file types
    fetch_parent_failed: bool = False            # Parent message couldn't be fetched
    parent_msg: Optional[discord.Message] = None # Next message in the reply chain
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Prevents concurrent processing
```

**Key behaviors:**
- Cache keyed by Discord message ID
- Lock per node prevents concurrent processing of the same message
- Oldest nodes evicted when cache exceeds 500 entries
- Text includes message content + embed text + text attachments
- User messages are prefixed with `"Correspondent No. {author_id}: "`

### 4. Mention Translation System

Discord uses `<@123456789>` for user pings, but the Talkie tokenizer doesn't have these tokens in its vocabulary. A bidirectional translation system solves this:

```
┌──────────────────┐                              ┌──────────────────┐
│  Discord format  │                              │   Model input    │
│  <@123456789>    │  ──llmcord preprocess──►     │ Correspondent    │
│                  │                              │ No. 123456789:   │
│  <@987654321>    │                              │ Correspondent    │
│                  │                              │ No. 987654321:   │
└──────────────────┘                              └──────────────────┘
                                                          │
                                                    Model generates
                                                    "Correspondent No. 123456789:"
                                                          │
                                                          ▼
┌──────────────────┐                              ┌──────────────────┐
│  Discord format  │  ◄──llmcord postprocess──   │   Model output   │
│  <@123456789>    │                              │ Correspondent    │
│                  │                              │ No. 123456789:   │
└──────────────────┘                              └──────────────────┘
```

**Preprocessing** (input to model):
- User message text: `discord_bot.user.mention` prefix stripped
- User ID prepended: `f"Correspondent No. {author_id}: {text}"`

**Postprocessing** (output from model):
- Regex: `r"Correspondent No\.\s*(\d+)(:)"` → `r"<@\1>:"`
- Applied by `era_to_discord()` function

### 5. Mutable Context System

The `/context reopen` command creates a time-limited window where the bot responds to @mentions in a channel without requiring a reply chain:

```python
t_mutable_context = {}  # channel_id → {opener_id, opened_at, ttl_seconds}
MUTABLE_CONTEXT_TTL = 900  # 15 minutes
```

**Flow:**
1. User runs `/context reopen` → channel ID recorded with timestamp
2. On subsequent messages, if channel has an active mutable context and the bot is @mentioned, it responds
3. After 15 minutes, the entry expires and is cleaned up on next message check
4. `/context clear` does not clear mutable context — it only clears the message cache

### 6. Launchd Persistence

Both services are managed by macOS launchd for reliability:

| Service | Plist | Logs |
|---------|-------|------|
| Inference server | `com.gwyntel.talkie-server.plist` | `~/Library/Logs/talkie-server.log` / `.err` |
| Discord bridge | `com.gwyntel.llmcord.plist` | `~/Library/Logs/llmcord.log` / `.err` |

**Key plist settings:**
- `RunAtLoad: true` — Start when user logs in
- `KeepAlive: true` — Auto-restart on crash
- Uses venv Python directly (`/Users/gwyn/mlx-env/bin/python3`) — no activation needed

**Startup ordering:** llmcord will fail to connect if the inference server hasn't loaded the model yet (~5 seconds). With `KeepAlive: true`, launchd keeps restarting llmcord until it succeeds.

---

## Data Flow: Complete Request Lifecycle

This traces a single user message through the entire system:

```
t=0.0s   User sends: "@Talkie What are your thoughts on the wireless telegraph?"
         │
t=0.0s   Discord delivers message via Gateway → llmcord on_message()
         │
t=0.0s   Gating: bot_mentioned=True → should_respond=True
         │
t=0.0s   Permissions: user not blocked, channel allowed → proceed
         │
t=0.0s   Context building:
         ├── Current message: "What are your thoughts on the wireless telegraph?"
         │   → MsgNode created, text="Correspondent No. USERID: What are your thoughts..."
         │   → Reply parent fetched (if any) → walk up chain
         │   → Collect up to 10 messages
         │
t=0.1s   System prompt appended with current date/time
         │
t=0.1s   OpenAI API call: stream=True
         ├── POST http://127.0.0.1:8080/v1/chat/completions
         ├── Body: {model, messages, stream: true}
         │
t=0.1s   server.py receives request
         ├── Validates JSON body
         ├── Applies chat template: _tokenizer.apply_chat_template(messages)
         ├── Counts prompt tokens
         │
t=0.1s   Generation begins (with EOS suppression):
         ├── mlx_lm.generate() with logits_processors=[eos_processor]
         ├── EOS bias -100 on tokens {65535-65539} for first 80 tokens
         │   (prevents model from stopping after "I think...")
         │
t=3.0s   Generation complete (~50 tokens at 17 tok/s)
         │
t=3.0s   SSE streaming begins (word-by-word from pre-generated text):
         ├── data: {"choices":[{"delta":{"content":"I"}}]}
         ├── data: {"choices":[{"delta":{"content":" must"}}]}
         ├── data: {"choices":[{"delta":{"content":" say,"}}]}
         ├── ...
         ├── data: {"choices":[{"finish_reason":"stop"}]}
         ├── data: [DONE]
         │
t=3.0s   llmcord receives chunks:
         ├── Each delta.content appended to response_contents
         ├── era_to_discord() converts "Correspondent No. N:" → "<@N>:"
         ├── Embed edited every 1 second (or on chunk boundary)
         ├── ⚪ indicator while streaming
         │
t=5.0s   Streaming complete
         ├── Final embed: green color, no ⚪
         ├── Response cached in msg_nodes
         ├── Oldest nodes evicted if cache > 500
```

---

## Model Architecture (Talkie 1930 13B IT)

Talkie uses a custom GPT variant, **not** based on Llama/Qwen/Mistral. This is why standard tooling doesn't work out of the box.

| Characteristic | Value |
|---|---|
| Parameters | 13B |
| Architecture | Custom GPT variant (`TalkieForCausalLM`) |
| Vocab size | 65,540 |
| Layers | 40 |
| Hidden dim | 5,120 |
| Attention heads | 40 |
| Head dim | 128 |
| RoPE base | 1,000,000 |
| Training context | 2,048 tokens |
| Extended context | 4,096 tokens (2x, reliable) |

**Unique architectural features:**
- Per-head scalar gains on queries (`TalkieHeadGain`)
- Scalar weight gain on `lm_head` (`TalkieWeightGain`)
- Per-layer activation gains at `(2 × n_layer)^−0.5`
- Embedding skip connections per layer
- QK-Norm (RMS norm on Q and K after RoPE)
- Custom tiktoken tokenizer (not HuggingFace `tokenizers`)

**Chat template tokens:**

| Token | Role |
|-------|------|
| `segmentor` | System turn prefix |
| `uctor` | User turn prefix |
| `ECTOR` | Assistant turn prefix |
| `<|end|>` | End of turn |

**EOS tokens:** The model uses multiple end-of-sequence tokens: `{65535, 65536, 65537, 65539}`. All are suppressed during the first 80 generated tokens to prevent premature stopping.

---

## Configuration Architecture

### server.py Configuration

| Source | Parameters |
|--------|-----------|
| CLI flags | `--model-dir`, `--port`, `--host`, `--model-id`, `--log-level` |
| Environment | `LOG_LEVEL` (overridden by `--log-level`) |
| Model config.json | `max_seq_len`, quantization settings |
| Hardcoded | EOS bias, temperature/top_p defaults, max_tokens default |

### llmcord Configuration (`config.yaml`)

```yaml
# Discord connection
bot_token: "..."           # Discord bot token
client_id: "..."           # Discord app client ID
status_message: "..."      # Bot status text

# Context limits
max_text: 3000             # Characters per message (truncates)
max_images: 0              # Images per message (0 = text only)
max_messages: 10           # Messages in reply chain

# Display
use_plain_responses: false # Embed (false) vs plain text (true)
allow_dms: true            # Respond in DMs

# Permissions
permissions:
  users: {admin_ids, allowed_ids, blocked_ids}
  roles: {allowed_ids, blocked_ids}
  channels: {allowed_ids, blocked_ids}

# Inference
providers:
  talkie:
    base_url: http://127.0.0.1:8080/v1
    api_key: local-no-key-needed

models:
  talkie/talkie-1930-13b-it-mlx-8bit: {}

# Persona
system_prompt: |
  You are an Edwardian-era conversationalist...
  Today's date is {date}. The current time is {time}.
```
