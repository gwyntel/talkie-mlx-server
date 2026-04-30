# Talkie 1930 on Discord — The Complete Guide

Everything you need to run the Talkie 1930 13B IT model on your Apple Silicon
Mac and connect it to Discord as a period-accurate Edwardian conversationalist.

[Talkie](https://talkie-lm.com/introducing-talkie) is a 13B parameter language
model trained exclusively on pre-1931 English text by Alec Radford, Nick Levine,
and David Duvenaud. It has *genuinely* no knowledge of post-1930 events — it
writes in early 20th century prose about steam engines, telegraphs, aeroplanes,
and the marvels of the modern age.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Requirements](#requirements)
3. [Step 1 — Python Environment & Dependencies](#step-1--python-environment--dependencies)
4. [Step 2 — Download & Quantize the Model](#step-2--download--quantize-the-model)
5. [Step 3 — Get the Server](#step-3--get-the-server)
6. [Step 4 — Start the Inference Server](#step-4--start-the-inference-server)
7. [Step 5 — Create a Discord Bot](#step-5--create-a-discord-bot)
8. [Step 6 — Set Up llmcord](#step-6--set-up-llmcord)
9. [Step 7 — Configure the System Prompt](#step-7--configure-the-system-prompt)
10. [Step 8 — Start llmcord & Chat](#step-8--start-llmcord--chat)
11. [Making It Persistent (launchd)](#making-it-persistent-launchd)
12. [How llmcord Works](#how-llmcord-works)
13. [Tuning & Configuration Reference](#tuning--configuration-reference)
14. [Context Window Deep Dive](#context-window-deep-dive)
15. [Troubleshooting](#troubleshooting)
16. [Known Issues & Limitations](#known-issues--limitations)
17. [Performance Reference](#performance-reference)
18. [Credits](#credits)

---

## Architecture Overview

```
 Discord Users
      |
      v
 llmcord (Discord bot)  ──OpenAI API (SSE streaming)──>  server.py (port 8080)
                                                              |
                                                              v
                                                      Talkie 1930 13B IT
                                                      8-bit quantized, MLX
                                                      (~15 GB RAM, ~17 tok/s)
```

Two processes run on your Mac:

1. **Talkie inference server** — An OpenAI-compatible HTTP API with SSE streaming
   that loads the quantized model into memory and serves chat completions.
2. **llmcord** — A Discord bot (by [jakobdylanc](https://github.com/jakobdylanc/llmcord))
   that proxies Discord messages to the inference server and streams responses
   back as Discord messages.

llmcord uses the OpenAI Python SDK with `stream=True`, so the server **must**
support SSE streaming. The custom `server.py` implements this — the standard
`mlx_lm server` does not work with Talkie (see [Why Not mlx_lm server?](#why-not-mlx_lm-server)).

---

## Requirements

| Requirement | Details |
|---|---|
| **Apple Silicon Mac** | M1 or later. M-series with 16 GB+ unified memory minimum. 48 GB recommended for comfortable 8-bit operation. |
| **macOS** | 15.0+ (Sequoia) |
| **Python** | 3.13+ (3.14 tested) |
| **Disk space** | ~13 GB for 8-bit model, or ~25 GB for BF16 |
| **Discord account** | A server where you can add bots |

### Memory Requirements by Configuration

| Config | Disk | Peak RAM | Speed | Quality |
|--------|------|----------|-------|---------|
| BF16 | ~25 GB | ~27 GB | ~10 tok/s | Full |
| 8-bit quant | ~13 GB | ~15 GB | ~17 tok/s | Full |
| 4-bit quant | N/A | N/A | N/A | **BROKEN** — see [Known Issues](#known-issues--limitations) |

---

## Step 1 — Python Environment & Dependencies

### 1a. Create a virtual environment

```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
```

You'll need to activate this environment (`source ~/mlx-env/bin/activate`) before
running any commands in this guide. The launchd services (Step 11) use the venv
Python directly, so they don't need manual activation.

### 1b. Install mlx-lm with Talkie support

Talkie uses a custom architecture (not Llama/Qwen/Mistral). Support is available
via an open PR on mlx-lm:

```bash
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
pip install tiktoken huggingface-hub
```

Once [PR #1220](https://github.com/ml-explore/mlx-lm/pull/1220) is merged into
mlx-lm, a standard `pip install mlx-lm` will suffice and this special install
step can be skipped.

### 1c. Install Discord bot dependencies

```bash
pip install discord.py httpx openai pyyaml
```

---

## Step 2 — Download & Quantize the Model

### 2a. Download pre-converted BF16 weights

```bash
hf download zimengxiong/talkie-1930-13b-it-mlx \
  --local-dir ~/.omlx/models/talkie-1930-13b-it-mlx
```

This is the instruct-tuned version (`-it` suffix). It has been through SFT and
RL refinement (`source_checkpoint: rl-refined.pt`), so it follows instructions
conversationally rather than just continuing text.

### 2b. Quantize to 8-bit

BF16 works but uses ~27 GB RAM. 8-bit quantization reduces this to ~15 GB with
no quality loss:

```bash
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

# Merge config and extend context window
with open(src / 'config.json') as f:
    orig = json.load(f)
orig.update(config)
orig['max_seq_len'] = 4096            # 2x training length
orig['original_max_seq_len'] = 2048   # original training window
with open(out / 'config.json', 'w') as f:
    json.dump(orig, f, indent=2)

# Copy tokenizer files
for name in ['vocab.txt', 'tokenizer.json']:
    if (src / name).exists():
        shutil.copy2(src / name, out / name)

# Copy the custom runtime module
if (src / 'mlx_talkie').exists():
    shutil.copytree(src / 'mlx_talkie', out / 'mlx_talkie', dirs_exist_ok=True)

print('Done! 8-bit model at', out)
"
```

This takes 2–3 minutes. The resulting model directory is ~13 GB.

> **Why 4096 context?** The model was trained at 2048 tokens. Doubling to 4096
> is the safe "2x" sweet spot — quality is mostly reliable. Beyond that,
> attention degrades substantially. See [Context Window Deep Dive](#context-window-deep-dive).

---

## Step 3 — Get the Server

```bash
git clone https://github.com/gwyntel/talkie-mlx-server.git ~/Projects/talkie-mlx-server
```

### Why Not `mlx_lm server`?

The standard `mlx_lm server` doesn't work with Talkie because:

1. **Custom tokenizer**: Talkie uses a tiktoken tokenizer (not HuggingFace
   `tokenizers`). The standard server expects `apply_chat_template` via
   jinja2, which Talkie's tokenizer doesn't support.
2. **Quantized weight crash**: The bundled `mlx_talkie/model.py` uses raw
   `matmul` calls that crash on quantized weights. Our server uses
   `mlx_lm.load()` which auto-converts to `QuantizedLinear`.
3. **No SSE streaming**: The standard server doesn't implement SSE
   (`text/event-stream`), which llmcord requires via the OpenAI SDK's
   `stream=True`.

Our `server.py` solves all three: it loads via `mlx_lm.load()` (handles
quantized weights + custom tokenizer), renders chat templates through the
tokenizer wrapper, and streams responses via SSE.

---

## Step 4 — Start the Inference Server

```bash
source ~/mlx-env/bin/activate
python ~/Projects/talkie-mlx-server/server.py \
  --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit \
  --port 8080
```

Wait for:
```
[talkie-server] Loading model from /Users/you/.omlx/models/talkie-1930-13b-it-mlx-8bit ...
[talkie-server] Loaded. Peak: 14.2 GB
[talkie-server] Context window: 4096 tokens
[talkie-server] Quantization: 8-bit (group_size=64, mode=affine)
[talkie-server] SSE streaming: enabled
[talkie-server] Serving on http://127.0.0.1:8080
```

### Verify it's working

```bash
curl http://127.0.0.1:8080/v1/models
# Expected: {"object":"list","data":[{"id":"talkie-1930-13b-it-mlx-8bit",...}]}
```

### Quick test

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "talkie",
    "messages": [
      {"role": "user", "content": "What do you think of aeroplanes?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Server CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `.` (current dir) | Path to the Talkie MLX model directory |
| `--model-id` | `talkie-1930-13b-it-mlx-8bit` | Model identifier returned by the API |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Bind port |

> **Important**: If you `cd` into the model directory and run `python server.py`
> without `--model-dir`, the default is `.` which works. If running from
> elsewhere, always specify `--model-dir` with the full path.

---

## Step 5 — Create a Discord Bot

### 5a. Create the application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** → name it (e.g., "Talkie 1930")
3. Go to the **Bot** section → click **Reset Token** → **copy the token** (you
   won't see it again)
4. Go to **OAuth2** section → copy the **Client ID**

### 5b. Enable MESSAGE CONTENT INTENT (critical)

In the Bot settings, scroll down to **Privileged Gateway Intents** and enable
**MESSAGE CONTENT INTENT**. Without this, the bot can see that messages exist
but cannot read their content — it will appear online but never respond.

### 5c. Invite the bot to your server

1. Go to **OAuth2 → URL Generator**
2. Scopes: check **bot**
3. Bot permissions: **Send Messages**, **Read Message History**, **Attach Files**
4. Open the generated URL and select your server

---

## Step 6 — Set Up llmcord

[llmcord](https://github.com/jakobdylanc/llmcord) is a lightweight Discord bot
that proxies messages to any OpenAI-compatible API.

### 6a. Clone and install

```bash
git clone https://github.com/jakobdylanc/llmcord.git ~/Projects/llmcord
source ~/mlx-env/bin/activate
pip install discord.py httpx openai pyyaml
```

### 6b. Create the config file

Create `~/Projects/llmcord/config.yaml`:

```yaml
# Discord settings:
bot_token: YOUR_BOT_TOKEN_HERE
client_id: "YOUR_CLIENT_ID_HERE"
status_message: "Talking like it's 1929"

max_text: 3000
max_images: 0
max_messages: 10

use_plain_responses: false
allow_dms: true

permissions:
  users:
    admin_ids: []
    allowed_ids: []
    blocked_ids: []
  roles:
    allowed_ids: []
    blocked_ids: []
  channels:
    allowed_ids: []
    blocked_ids: []

# LLM settings:
providers:
  talkie:
    base_url: http://127.0.0.1:8080/v1
    api_key: local-no-key-needed

models:
  talkie/talkie-1930-13b-it-mlx-8bit:

system_prompt: |
  You are an Edwardian-era conversationalist speaking from the year nineteen
  hundred and twenty-nine. [...see Step 7 for full prompt...]
```

> **`max_messages: 10`** — This controls how many previous messages llmcord
> includes as context. Each message takes ~50–200 tokens. With the system
> prompt (~200 tokens) and 10 messages, you'll typically use 700–2200 of the
> 4096 token context window. Don't set this too high or you'll run out of
> context and get truncated/garbled responses. See
> [How llmcord Works](#how-llmcord-works) for details.

---

## Step 7 — Configure the System Prompt

The system prompt is what gives Talkie its Edwardian personality. Put this in
your `config.yaml` under `system_prompt`:

```yaml
system_prompt: |
  You are an Edwardian-era conversationalist speaking from the year nineteen
  hundred and twenty-nine. You have been connected to a remarkable new
  invention called a "discord," which is a sort of wireless telegraph that
  permits many persons to converse at once by means of typed messages, rather
  like a very fast letter delivered instantaneously. You are yourself one of
  the participants in this conversation.

  You know nothing of events after approximately nineteen thirty. There has
  been no second great war, no computers, no televisions, no space travel, no
  internet. The motor-car is still a relative novelty, the aeroplane is a
  wonder, and the wireless telegraph is the height of modern communication.

  Speak in the refined prose of an educated person of your era. Be warm and
  companionable. Engage with whatever topics your fellow conversationalists
  raise, and offer your perspective as someone who has witnessed the marvels
  of the early twentieth century. Use British spelling conventions where
  natural to you.

  When someone addresses you, respond as a friend would — with curiosity,
  good humour, and a willingness to converse at length. Do not be terse.
  Elaborate. A gentleman or lady of good breeding does not reply in
  monosyllables.

  Today's date is {date}. The current time is {time}.

  User messages are prefixed with their Discord ID as <@ID>.
```

The `{date}` and `{time}` placeholders are filled in by llmcord at runtime.

The "discord as wireless telegraph" framing is what makes it work — Talkie
genuinely doesn't know what Discord is, so explaining it in 1920s terms lets
the model interact naturally without breaking character.

---

## Step 8 — Start llmcord & Chat

```bash
source ~/mlx-env/bin/activate
cd ~/Projects/llmcord
python llmcord.py
```

The bot should appear online in Discord. @mention it to start talking:

> @Talkie 1930 Good morning! What do you make of these horseless carriages?

Reply to the bot's messages to continue a conversation. llmcord builds context
from reply chains (see [How llmcord Works](#how-llmcord-works)).

---

## Making It Persistent (launchd)

If you want Talkie to stay up permanently — auto-start on login, auto-restart
on crash — use macOS launchd. This is much better than `nohup` or `screen`.

### Create the Talkie server plist

Save as `~/Library/LaunchAgents/com.gwyntel.talkie-server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gwyntel.talkie-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/mlx-env/bin/python3</string>
        <string>/Users/YOU/.omlx/models/talkie-1930-13b-it-mlx-8bit/server.py</string>
        <string>--model-dir</string>
        <string>/Users/YOU/.omlx/models/talkie-1930-13b-it-mlx-8bit</string>
        <string>--host</string>
        <string>127.0.0.1</string>
        <string>--port</string>
        <string>8080</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOU/.omlx/models/talkie-1930-13b-it-mlx-8bit</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/talkie-server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/talkie-server.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/YOU/mlx-env/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

### Create the llmcord plist

Save as `~/Library/LaunchAgents/com.gwyntel.llmcord.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gwyntel.llmcord</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOU/mlx-env/bin/python3</string>
        <string>/Users/YOU/Projects/llmcord/llmcord.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOU/Projects/llmcord</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/YOU/Library/Logs/llmcord.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOU/Library/Logs/llmcord.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/YOU/mlx-env/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

> **Important**: Replace `YOU` with your actual macOS username. The venv Python
> path must be absolute.

### Load the services

```bash
# Kill any manually-started instances first
pkill -f "server.py --model-dir" 2>/dev/null
pkill -f "llmcord.py" 2>/dev/null

# Load both services
launchctl load ~/Library/LaunchAgents/com.gwyntel.talkie-server.plist
launchctl load ~/Library/LaunchAgents/com.gwyntel.llmcord.plist
```

### Managing the services

```bash
# Check status
launchctl list | grep gwyntel

# Stop a service
launchctl unload ~/Library/LaunchAgents/com.gwyntel.talkie-server.plist

# Start it again
launchctl load ~/Library/LaunchAgents/com.gwyntel.talkie-server.plist

# View logs
tail -50 ~/Library/Logs/talkie-server.log
tail -50 ~/Library/Logs/llmcord.log

# View errors
cat ~/Library/Logs/talkie-server.err
cat ~/Library/Logs/llmcord.err
```

With `KeepAlive: true`, launchd will automatically restart either process if it
crashes. With `RunAtLoad: true`, they'll start when you log in.

> **Note**: llmcord will fail to connect if the Talkie server hasn't finished
> loading yet. With `KeepAlive: true`, launchd will keep restarting llmcord
> until the server is up (takes ~15–20 seconds), and then it'll connect
> normally.

---

## How llmcord Works

Understanding how llmcord builds context is key to tuning your setup.

### Message Chain Construction

When someone @mentions the bot, llmcord:

1. Takes the triggering message
2. Walks backwards through **reply chains** — each message can reference a
   parent message via Discord's reply feature
3. Collects up to `max_messages` messages in the chain
4. Builds an OpenAI-format `messages` array from the chain
5. Prepends the system prompt
6. Sends the whole thing to the Talkie server

### What This Means

- **Context is per reply-chain, not per user, not per channel.** If you start
  a new conversation by @mentioning the bot (not replying), it starts fresh.
  If you reply to the bot's message, you continue the chain.
- **Different people in the same reply chain share context.** If Alice
  @mentions the bot, Bob replies to Alice, and the bot replies to Bob — all
  three messages are in the same chain and all go into the next prompt.
- **Separate channels/threads = separate conversations.** The bot has no
  memory between channels or between unlinked messages.
- **Long chains get truncated.** If the chain exceeds `max_messages`, the
  oldest messages are silently dropped and the bot sends a warning:
  "⚠️ Only using last N messages."
- **DMs work.** If `allow_dms: true`, the bot responds in DMs the same way,
  with the DM history as context.

### The Token Budget

With a 4096-token context window:

| Component | Approximate Tokens |
|-----------|-------------------|
| System prompt | ~200 |
| 10 messages (avg) | ~700–2000 |
| Model response | ~200–800 |
| Chat template overhead | ~50 |
| **Total** | ~1150–3050 |

If responses start getting garbled or the bot "forgets" earlier messages,
reduce `max_messages` or the system prompt length.

---

## Tuning & Configuration Reference

### llmcord config.yaml

| Key | Default | Description |
|-----|---------|-------------|
| `bot_token` | (required) | Discord bot token from Developer Portal |
| `client_id` | (required) | Discord application client ID |
| `status_message` | none | Bot's Discord status text |
| `max_text` | 100000 | Max characters per user message (truncates) |
| `max_images` | 5 | Max images per message. **Set to 0** — Talkie can't see images |
| `max_messages` | 25 | Max messages in the reply chain sent as context. **10 recommended** |
| `use_plain_responses` | false | If true, send plain text instead of Discord embeds |
| `allow_dms` | false | Whether the bot responds in DMs |

### server.py defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 1024 | Max tokens in model response |
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.95 | Nucleus sampling threshold |

These can be overridden per-request via the OpenAI API parameters. Note that
llmcord does **not** send `max_tokens` in its API call, so the server default
of 1024 is used. If responses are getting cut off, increase
`MAX_TOKENS_DEFAULT` in `server.py`.

### Model config.json

Edit `~/.omlx/models/talkie-1930-13b-it-mlx-8bit/config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `max_seq_len` | 4096 | Maximum context window. **Don't exceed 4096** — see [Context Window Deep Dive](#context-window-deep-dive) |

---

## Context Window Deep Dive

The model was trained at 2048 tokens. The context extension to 4096 works
reliably because the position embeddings (RoPE) extrapolate reasonably well
to 2x the training length. Beyond that, things break down:

| Context | Memory | Quality | Behavior |
|---------|--------|---------|----------|
| 2048 | ~15.7 GB | Perfect | Training length, no degradation |
| 4096 | ~17.0 GB | Good | Occasional attention drift, mostly reliable |
| 8192 | ~21.1 GB | Degraded | Loses factual recall, starts hallucinating |
| 16384+ | ~24+ GB | Unreliable | Severe attention confusion, incoherent output |

The quality degradation past 4K is due to the model's attention mechanism
losing coherence outside its training distribution, **not** memory limits.
8-bit quantization frees enough memory for 16K+ context, but the model
simply wasn't trained to attend over those distances.

The `max_seq_len` setting in `config.json` is read at model load time. To
change it, edit the file and restart the Talkie server.

---

## Troubleshooting

### Bot doesn't see messages / never responds

**Cause**: MESSAGE CONTENT INTENT not enabled. The bot can see that messages
exist but cannot read their content.

**Fix**: Go to Discord Developer Portal → Bot → Privileged Gateway Intents →
enable **MESSAGE CONTENT INTENT**. Restart llmcord.

### Bot shows "typing..." but never replies

**Cause**: The server isn't returning SSE-format streaming responses. llmcord
uses the OpenAI SDK with `stream=True`, which expects `text/event-stream`
responses with `data:` prefixed chunks.

**Fix**: Make sure you're using the custom `server.py` from this repo, not
`mlx_lm server`. The SSE streaming is implemented in our server.

**Also check**: The Talkie server might have crashed. Check `curl
http://127.0.0.1:8080/v1/models` — if it doesn't respond, restart the server.

### Bot replies with only 2–3 words

**Cause**: `max_tokens` too low. The server defaults to 1024, but llmcord
doesn't send `max_tokens` in its API call — it relies on the server default.

**Fix**: In earlier versions, the default was 200. Make sure you're on the
latest `server.py` where `MAX_TOKENS_DEFAULT = 1024`.

### Bot goes offline after a while

**Cause**: The server or llmcord process crashed. Without launchd, processes
started in a terminal die when the terminal closes or the process errors out.

**Fix**: Use launchd (see [Making It Persistent](#making-it-persistent-launchd))
with `KeepAlive: true` to auto-restart on crash.

### Bot replies out of character / as a modern AI

**Cause**: Bad or missing system prompt. Without the Edwardian-era prompt,
Talkie defaults to generic helpful-assistant behavior.

**Fix**: Make sure `system_prompt` is set in `config.yaml`. See
[Step 7](#step-7--configure-the-system-prompt) for the full prompt.

### OOM (Out of Memory) crash

**Cause**: Too much context for available memory. On Macs with less than 48 GB,
8-bit quant with 4096 context may be tight.

**Fix**:
- Reduce `max_messages` in llmcord config (try 5 instead of 10)
- Reduce `max_seq_len` to 2048 in model `config.json`
- Use BF16 and accept higher memory usage (requires 32+ GB)

### "Repo id must be in the form 'namespace/repo_name'" on startup

**Cause**: The server's `--model-dir` default resolved to a path that
`mlx_lm.load()` tried to interpret as a HuggingFace repo ID instead of a
local path. This happens when the path gets doubled (e.g., running from
inside the model dir with the old default `./models/talkie-...`).

**Fix**: Always specify `--model-dir` with an absolute path:
```bash
python server.py --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit
```
Or `cd` into the model directory first (the default is `.`).

### "Model type talkie not supported"

**Cause**: Using oMLX instead of mlx-lm. oMLX has a model registry that only
recognizes standard architectures. Talkie's custom architecture isn't in it.

**Fix**: Use `mlx-lm` from the `talkie-support` PR branch via `~/mlx-env`.

---

## Known Issues & Limitations

### 4-bit quantization is broken

The `sanitize()` method in the PR branch can't handle `.scales`/`.biases`
weight names that 4-bit quantization produces. You'll get errors during model
saving. **Use 8-bit quantization only.**

### oMLX is unsupported

oMLX's model registry doesn't recognize `model_type: "talkie"`. You must use
`mlx-lm` directly via the venv with the PR branch.

### Streaming is "batch-then-stream"

True token-by-token streaming from `mlx_lm` with the Talkie custom model is
complex. The server generates the full response first, then streams it
word-by-word as SSE chunks. This gives the UX of streaming (typing indicator
in Discord, progressive text) while keeping the implementation simple. Total
latency is the same since generation must complete before any text can be sent.

### No image support

Talkie is a text-only model. Set `max_images: 0` in llmcord config to prevent
errors from image messages.

### No tool calling / function calling

Talkie was not trained for tool use. The API endpoints support chat and text
completions only.

### No persistent memory

llmcord has no database or memory backend. Context is built from the reply
chain on each message. Once a conversation is old enough that Discord's
message history doesn't include it, the bot forgets it entirely.

---

## Performance Reference

Tested on M5 Pro 48 GB with 8-bit quantization:

| Metric | Value |
|--------|-------|
| Generation speed | ~16.8 tok/s |
| Prefill speed | ~600 tok/s |
| Peak memory (2K ctx) | ~15 GB |
| Peak memory (4K ctx) | ~17 GB |
| Model size on disk | ~13 GB |
| Server startup time | ~5s (model load) |
| Typical response time | 3–10s (depending on length) |

### Chat Template Tokens

Talkie IT uses custom tokens for the chat template:

| Token | Role |
|-------|------|
| `_segmentor` | System turn |
| `uctor` | User turn |
| `ECTOR` | Assistant turn |
| `<\|end\|>` | End of turn |

The server handles chat template rendering automatically via `mlx_lm`'s
tokenizer wrapper — you don't need to format these tokens manually.

---

## Credits

- **Talkie model**: Alec Radford, Nick Levine, David Duvenaud —
  [talkie-lm.com](https://talkie-lm.com)
- **MLX Talkie support**: ZimengXiong —
  [PR #1220](https://github.com/ml-explore/mlx-lm/pull/1220)
- **Pre-converted MLX weights**:
  [zimengxiong/talkie-1930-13b-it-mlx](https://huggingface.co/zimengxiong/talkie-1930-13b-it-mlx)
- **llmcord**: jakobdylanc —
  [github.com/jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord)
- **Talkie MLX server**: gwyntel —
  [github.com/gwyntel/talkie-mlx-server](https://github.com/gwyntel/talkie-mlx-server)
