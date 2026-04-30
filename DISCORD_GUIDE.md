# Running Talkie 1930 on Discord with llmcord

A guide to running the Talkie 1930 13B IT model (a 13B LLM trained exclusively
on pre-1931 English) on your local Apple Silicon Mac, then connecting it to
Discord so your friends can chat with an Edwardian-era conversationalist.

## Architecture

```
Discord Users
     |
     v
  llmcord (Discord bot) --- OpenAI API (streaming SSE) ---> server.py (port 8080)
                                                                |
                                                                v
                                                        Talkie 1930 13B IT
                                                        8-bit quantized, MLX
                                                        (~15 GB RAM on M-series)
```

Two processes run on your Mac:
1. **Talkie inference server** -- OpenAI-compatible API with SSE streaming
2. **llmcord** -- Discord bot that proxies messages to the inference server

## Requirements

- Apple Silicon Mac (M1+, 48 GB recommended for 8-bit quant)
- macOS 15+
- Python 3.13+
- A Discord account and a server where you can add bots

## Step 1: Set Up the Talkie Inference Server

### 1a. Create a Python virtual environment

```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
```

### 1b. Install mlx-lm with Talkie support

Talkie support is in an open PR on mlx-lm:

```bash
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
pip install tiktoken huggingface-hub
```

### 1c. Download and quantize the model

```bash
# Download pre-converted BF16 weights
hf download zimengxiong/talkie-1930-13b-it-mlx \
  --local-dir ~/.omlx/models/talkie-1930-13b-it-mlx

# Quantize to 8-bit (~13 GB, ~15 GB peak memory)
python -c "
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

### 1d. Get the server

```bash
git clone https://github.com/gwyntel/talkie-mlx-server.git ~/Projects/talkie-mlx-server
```

### 1e. Start the server

```bash
source ~/mlx-env/bin/activate
python ~/Projects/talkie-mlx-server/server.py \
  --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit \
  --port 8080
```

Wait for `[talkie-server] Serving on http://127.0.0.1:8080`.

Verify:
```bash
curl http://127.0.0.1:8080/v1/models
```

## Step 2: Set Up the Discord Bot

### 2a. Create a Discord application

1. Go to https://discord.com/developers/applications
2. Click **New Application** → name it (e.g., "Talkie 1930")
3. Go to **Bot** section → click **Reset Token** → copy the token
4. Go to **OAuth2** section → copy the **Client ID**
5. **CRITICAL**: Under Bot settings → enable **MESSAGE CONTENT INTENT**

### 2b. Invite the bot to your server

1. Go to **OAuth2 → URL Generator**
2. Scopes: check **bot**
3. Bot permissions: **Send Messages**, **Read Message History**, **Attach Files**
4. Open the generated URL to invite the bot to your server

## Step 3: Set Up llmcord

### 3a. Clone and install

```bash
git clone https://github.com/jakobdylanc/llmcord.git ~/Projects/llmcord
source ~/mlx-env/bin/activate
pip install discord.py httpx openai pyyaml
```

### 3b. Configure

Create `~/Projects/llmcord/config.yaml`:

```yaml
# Discord settings:
bot_token: YOUR_BOT_TOKEN_HERE
client_id: "YOUR_CLIENT_ID_HERE"
status_message: Talking like it's 1929

max_text: 3000
max_images: 0
max_messages: 15

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

### 3c. Start llmcord

```bash
source ~/mlx-env/bin/activate
cd ~/Projects/llmcord
python llmcord.py
```

## Step 4: Chat!

In your Discord server, @mention the bot to start a conversation:

> @Talkie 1930 Hello there! What do you think of aeroplanes?

Reply to the bot's messages to continue the conversation (context is built
from reply chains). The bot streams responses in real-time via Discord embeds.

## Running Both as Background Services

For long-running use, you can run both processes in the background:

```bash
# Start Talkie server
source ~/mlx-env/bin/activate
python ~/Projects/talkie-mlx-server/server.py \
  --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit \
  --port 8080 &

# Start llmcord
cd ~/Projects/llmcord && python llmcord.py &
```

Or use Docker (llmcord includes a Dockerfile):

```bash
cd ~/Projects/llmcord
docker compose up -d
```

Note: Docker uses `network_mode: host` so llmcord can reach the Talkie
server on localhost:8080.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Bot doesn't see messages | MESSAGE CONTENT INTENT not enabled | Discord Developer Portal → Bot → enable it |
| Bot types but never replies | Server not returning SSE format | Make sure you're using the latest server.py with streaming support |
| Bot replies with 2-3 words only | max_tokens too low | Server defaults to 1024; make sure you're on latest server.py |
| Bot replies out of character | Bad system prompt | Use the 1920s-era prompt from the config above |
| OOM crash | Too much context for 48 GB | Reduce `max_messages` in llmcord config or `max_seq_len` in model config |
| "Model type talkie not supported" | Using oMLX instead of mlx-lm | Must use mlx-lm via ~/mlx-env with the PR branch |

## Performance Reference

On M5 Pro 48 GB with 8-bit quantization:

| Metric | Value |
|--------|-------|
| Generation speed | ~16.8 tok/s |
| Prefill speed | ~600 tok/s |
| Peak memory (2K ctx) | ~15 GB |
| Model size on disk | ~13 GB |
| Context window | 4096 tokens (2x training length) |
| Quality at 2K context | Perfect |
| Quality at 4K context | Good |
| Quality beyond 4K | Unreliable |

## What is Talkie?

Talkie is a 13B parameter language model by Alec Radford, Nick Levine, and
David Duvenaud, trained on 260 billion tokens of pre-1931 English text. It
has genuinely no knowledge of post-1930 events -- no WWII, no computers, no
internet. It writes in early 20th century prose about steam engines, the
wireless telegraph, and aeroplanes with complete authenticity.

Website: https://talkie-lm.com/introducing-talkie
GitHub: https://github.com/talkie-lm/talkie

## Credits

- **Talkie model**: Alec Radford, Nick Levine, David Duvenaud
- **MLX Talkie support**: ZimengXiong (PR #1220)
- **llmcord**: jakobdylanc
- **Talkie MLX server**: gwyntel
