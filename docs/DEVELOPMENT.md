# TalkieLM Development Guide

Setting up a development environment, understanding the codebase, testing, and contributing to the TalkieLM Discord bot project.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Apple Silicon Mac | M1+ | Required for MLX inference |
| macOS | 15.0+ (Sequoia) | Required for MLX |
| Python | 3.13+ | 3.14 tested |
| Git | Any | For cloning and contributing |
| Discord server | — | With bot management permissions |

---

## Development Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
```

Keep this environment active for all subsequent steps.

### 2. Install core dependencies

```bash
# MLX with Talkie support (PR branch)
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
pip install tiktoken huggingface-hub

# Discord bridge dependencies
pip install discord.py httpx openai pyyaml

# Development tools
pip install ruff mypy
```

### 3. Clone the repository

```bash
git clone https://github.com/gwyntel/talkie-mlx-server.git ~/Projects/talkie-mlx-server
cd ~/Projects/talkie-mlx-server
```

### 4. Download model weights

If you don't already have them:

```bash
# Download BF16 weights
hf download zimengxiong/talkie-1930-13b-it-mlx \
  --local-dir ~/.omlx/models/talkie-1930-13b-it-mlx

# Quantize to 8-bit (takes 2-3 minutes)
python3 -c "
from mlx_lm import load
from mlx_lm.utils import quantize_model, save_model
import json, shutil
from pathlib import Path

src = Path('$HOME/.omlx/models/talkie-1930-13b-it-mlx')
out = Path('$HOME/.omlx/models/talkie-1930-13b-it-mlx-8bit')

model, tokenizer = load(str(src))
model, config = quantize_model(model, {}, bits=8, group_size=64)
save_model(str(out), model, donate_model=True)

with open(src / 'config.json') as f:
    orig = json.load(f)
orig.update(config)
orig['max_seq_len'] = 4096
orig['original_max_seq_len'] = 2048
with open(out / 'config.json', 'w') as f:
    json.dump(orig, f, indent=2)

for name in ['vocab.txt', 'tokenizer.json']:
    if (src / name).exists():
        shutil.copy2(src / name, out / name)

if (src / 'mlx_talkie').exists():
    shutil.copytree(src / 'mlx_talkie', out / 'mlx_talkie', dirs_exist_ok=True)

print('Done!')
"
```

### 5. Set up llmcord (if working on the bridge)

```bash
# The patched llmcord is in the repo
cd ~/Projects/talkie-mlx-server

# Or if working on a separate clone:
git clone https://github.com/jakobdylanc/llmcord.git ~/Projects/llmcord
# Then apply Talkie-specific patches
```

---

## Project Structure

```
talkie-mlx-server/
├── server.py              # Inference server (OpenAI-compatible API)
├── llmcord.py             # Discord bridge (patched llmcord)
├── config.yaml            # Live configuration (not committed — contains bot token)
├── config.yaml.example    # Template configuration
├── config.example.json    # Model config example (quantization settings)
├── LICENSE
├── README.md              # Project overview
├── DISCORD_GUIDE.md       # Full Discord setup guide
├── docs/                  # This documentation suite
│   ├── README.md
│   ├── USER_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── TROUBLESHOOTING.md
│   ├── DEVELOPMENT.md
│   └── API.md
└── mlx_talkie/            # Talkie model runtime for MLX
    ├── __init__.py
    ├── chat.py            # Chat template formatting
    ├── cli.py             # CLI entry point
    ├── generate.py        # Text generation helpers
    ├── model.py           # Model architecture definition
    └── tokenizer.py       # Tokenizer wrapper
```

**Key files for development:**

| File | Language | Purpose | Lines |
|------|----------|---------|-------|
| `server.py` | Python | OpenAI-compatible HTTP server, EOS suppression, SSE streaming | ~514 |
| `llmcord.py` | Python | Discord bot, message chain builder, mention translator | ~659 |
| `config.yaml` | YAML | Live config (bot token, model, system prompt) | ~65 |
| `mlx_talkie/chat.py` | Python | Chat template with Talkie special tokens | ~48 |

---

## Running in Development

### Starting the inference server (foreground)

```bash
source ~/mlx-env/bin/activate
python server.py \
  --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit \
  --port 8080 \
  --log-level DEBUG
```

The `--log-level DEBUG` flag enables verbose logging of every request, including full request bodies and generated text. Only use this during active debugging — it's extremely chatty.

### Starting the Discord bridge (foreground)

```bash
source ~/mlx-env/bin/activate
cd ~/Projects/talkie-mlx-server  # or ~/Projects/llmcord
python llmcord.py
```

### Stopping launchd services (if running)

If you have launchd services loaded, stop them before running foreground:

```bash
launchctl unload ~/Library/LaunchAgents/com.gwyntel.talkie-server.plist
launchctl unload ~/Library/LaunchAgents/com.gwyntel.llmcord.plist
```

### Quick smoke test

```bash
# Test the inference server directly
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "messages": [{"role": "user", "content": "What is the wireless telegraph?"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
  }' | python3 -m json.tool
```

---

## Key Implementation Details

### EOS Suppression System

The most critical quality-of-life feature. Without it, every response is 1–2 sentences.

**How it works:**
1. `server.py` defines `_EOS_IDS = {65535, 65536, 65537, 65539}` — all EOS token IDs
2. A logits processor factory `_make_eos_processor()` returns a fresh processor per request
3. The processor adds -100 bias to all EOS token logits for the first 80 generated tokens
4. After 80 tokens, EOS is allowed naturally — the model decides when to stop

**Important:** The processor uses a closure with `count = [0]` (mutable list) because the processor function is called once per generated token. A new processor must be created for each request — don't reuse.

**Testing changes:** If you modify the EOS system, verify both streaming and non-streaming paths. The bug where streaming bypassed EOS suppression was real — always ensure both paths call `_generate_with_eos_suppress()`.

### Mention Translation

The Talkie tokenizer doesn't understand Discord's `<@ID>` mention format — those tokens are out-of-vocabulary. The bridge translates in both directions:

**Preprocessing (Discord → model):**
```python
# In llmcord.py, when building user messages:
curr_node.text = f"Correspondent No. {curr_msg.author.id}: {curr_node.text}"
```

**Postprocessing (model → Discord):**
```python
# Regex in llmcord.py
_CORRESPONDENT_PATTERN = re.compile(r"Correspondent No\.\s*(\d+)(:)")
def era_to_discord(text: str) -> str:
    return _CORRESPONDENT_PATTERN.sub(r"<@\1>:", text)
```

If you modify either direction, test that the regex matches what the model actually produces. The model sometimes generates slight variations.

### SSE Streaming in server.py

The server uses **batch-then-stream** — the model generates the entire response, then streams it word-by-word:

```python
def _chat_completion_stream(body: dict):
    # 1. Generate full text (with EOS suppression)
    full_text = _generate_with_eos_suppress(prompt, max_tokens, temperature, top_p)
    
    # 2. Split into word-level chunks
    parts = []
    lines = full_text.split("\n")
    for i, line in enumerate(lines):
        if line:
            words = line.split(" ")
            for j, word in enumerate(words):
                parts.append(word if j == 0 else " " + word)
        if i < len(lines) - 1:
            parts.append("\n")
    
    # 3. Yield as SSE chunks
    for part in parts:
        chunk = {"id": req_id, "choices": [{"delta": {"content": part}}]}
        yield f"data: {json.dumps(chunk)}\n\n"
```

This gives the UX of streaming but doesn't reduce total latency. True token-by-token streaming from MLX would require integrating with MLX's evaluation loop directly, which is complex for the Talkie custom model.

### Message Node Cache

llmcord caches processed messages in `msg_nodes = {}` keyed by Discord message ID. Key behaviors:

- **Lazy population:** Nodes are created on first access via `msg_nodes.setdefault(msg.id, MsgNode())`
- **Lock per node:** Prevents concurrent processing of the same message
- **Eviction:** Oldest nodes (by message ID) are evicted when the cache exceeds 500 entries
- **No persistence:** Cache is lost on restart — it rebuilds from Discord message history

### Context Building Flow

The reply chain is walked backwards from the triggering message:

```python
while curr_msg is not None and len(messages) < max_messages:
    curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())
    async with curr_node.lock:
        if curr_node.text is None:
            # Process message content, attachments, embeds
            # Prepend "Correspondent No. {author_id}: " for user messages
            # Determine parent message (reply, previous in channel, thread start)
        # Append to messages list
        curr_msg = curr_node.parent_msg  # Walk up the chain
```

**Parent determination logic:**
1. If message has a Discord reply reference → use referenced message
2. If in a public thread with no reply → use thread starter message
3. If previous message in channel is from expected author (bot in DMs, same user in guilds) → auto-chain
4. If parent fetch fails → `fetch_parent_failed = True`, chain stops

---

## Testing

### Manual testing checklist

Since there's no automated test suite, use this manual checklist:

**Server tests:**
- [ ] Server starts and loads model without errors
- [ ] `GET /v1/models` returns the model ID
- [ ] `POST /v1/chat/completions` (non-streaming) returns a complete response
- [ ] `POST /v1/chat/completions` (streaming) returns SSE chunks and `[DONE]`
- [ ] Responses are more than 2 sentences (EOS suppression working)
- [ ] Server logs show EOS bias configuration on startup
- [ ] Server handles `BrokenPipeError` gracefully (client disconnect mid-stream)

**Bridge tests:**
- [ ] Bot appears online in Discord
- [ ] Bot responds to @mentions in guild channels
- [ ] Bot responds to replies without @mention
- [ ] Bot responds in DMs (if `allow_dms: true`)
- [ ] Bot ignores its own messages
- [ ] `Correspondent No. N:` in bot output becomes `<@N>:` (proper Discord ping)
- [ ] `/context show` displays reply chain and token estimate
- [ ] `/context clear` resets the message cache
- [ ] `/context reopen` allows multiple users to join via @mention
- [ ] `/context status` shows correct server health
- [ ] Streaming indicator (⚪) appears during generation
- [ ] Embed color changes from orange to green on completion

### Testing with curl

```bash
# Non-streaming chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "messages": [
      {"role": "system", "content": "You are a helpful Edwardian-era assistant."},
      {"role": "user", "content": "Tell me about aeroplanes."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# Streaming chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# List models
curl http://127.0.0.1:8080/v1/models

# Text completion
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie-1930-13b-it-mlx-8bit",
    "prompt": "The wireless telegraph is",
    "max_tokens": 100
  }'
```

---

## Contributing

### Code Style

- Python 3.11+ features are fine (match statements, type hints, etc.)
- Use `logging` module — the project has "obsessive" logging as a design principle
- Docstrings on all public functions
- Type hints on function signatures
- 4-space indentation, 120-char line limit (soft)

### Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test manually (see checklist above)
5. Commit with a descriptive message
6. Push and open a Pull Request

### Important: Don't commit secrets

The `config.yaml` file contains the Discord bot token. It should **never** be committed to the repository. The `.gitignore` should include:

```
config.yaml
*.err
*.log
```

Use `config.yaml.example` as the template for new installations.

### Areas That Need Improvement

- **Automated tests** — There's currently no test suite. Even basic pytest fixtures would be valuable.
- **True token-by-token streaming** — The batch-then-stream approach works but adds latency. Integrating with MLX's evaluation loop for true streaming would improve the UX.
- **Context window management** — Currently manual (`/context clear`). Automatic context window management (summarize-then-continue) would be a major improvement.
- **Multiple model support** — The bridge supports it via config, but the server is hardcoded to one model.
- **Structured logging** — The current `logging` calls could be replaced with JSON-structured logs for easier parsing.

---

## Debugging Tips

### Server-side debugging

**Enable DEBUG logging:**
```bash
python server.py --model-dir ~/.omlx/... --log-level DEBUG
```

This logs:
- Full request bodies (up to 2000 chars)
- Full generated text for each request
- SSE chunk contents
- Memory usage per request

**Common server errors:**
- `matmul shape mismatch` → Wrong quantized weight format. Make sure you're using 8-bit, not 4-bit.
- `Token id X out of range` → Tokenizer/model mismatch. Ensure the tokenizer files match the model.
- `OOM` → Not enough memory. Reduce context or close other apps.

### Bridge-side debugging

**Enable DEBUG logging** by editing `llmcord.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

This logs:
- Every SSE chunk received from the server
- The complete message chain built from the reply chain
- Each step of the streaming display update
- Any mismatch between assembled parts and segments

**Common bridge errors:**
- `Forbidden 403` → Bot lacks permissions in the channel
- `ConnectionRefused` → Inference server isn't running
- `[MISMATCH] parts != segments` → Bug in streaming assembly — this should never happen

### Using the logs with launchd

```bash
# Follow live server logs
tail -f ~/Library/Logs/talkie-server.log

# Follow live bridge logs
tail -f ~/Library/Logs/llmcord.log

# Check for errors (note: .err may contain normal startup messages too)
cat ~/Library/Logs/talkie-server.err | tail -20
cat ~/Library/Logs/llmcord.err | tail -20

# Search for specific patterns
grep "FAILED\|ERROR\|Traceback" ~/Library/Logs/talkie-server.log
grep "STREAM_ERROR\|MISMATCH\|Traceback" ~/Library/Logs/llmcord.log
```
