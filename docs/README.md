# TalkieLM Discord Bot — Documentation

> A period-accurate Edwardian conversationalist on your Discord server, powered by the Talkie 1930 13B IT model running locally on Apple Silicon.

## What Is TalkieLM?

[Talkie](https://talkie-lm.com/introducing-talkie) is a 13B parameter language model trained exclusively on pre-1931 English text. It has *genuinely* no knowledge of post-1930 events — it writes in early 20th century prose about steam engines, telegraphs, aeroplanes, and the marvels of the modern age.

This project connects Talkie to Discord via a dual-service architecture:

```
 Discord Users
      │
      ▼
 llmcord (Discord bot)  ──OpenAI API (SSE streaming)──►  server.py (port 8080)
                                                              │
                                                              ▼
                                                      Talkie 1930 13B IT
                                                      8-bit quantized, MLX
                                                      (~15 GB RAM, ~17 tok/s)
```

### Key Features

- 🎩 **Period-accurate persona** — Edwardian/1920s English, no post-1930 knowledge
- 🍎 **Apple Silicon native** — Runs on M-series Macs via MLX with 8-bit quantization
- 🔌 **OpenAI-compatible API** — Drop-in `/v1/chat/completions`, `/v1/completions`, `/v1/models` endpoints
- 📡 **SSE streaming** — Word-by-word response streaming to Discord with live-updating embeds
- 💬 **Reply-chain context** — Builds conversation from Discord reply threads (up to 10 messages)
- 🔄 **Permanent via launchd** — Auto-start on login, auto-restart on crash
- 📂 **Mutable context windows** — `/context reopen` lets multiple users join a conversation for 15 minutes

## Quick Start

```bash
# 1. Set up Python environment
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
pip install tiktoken huggingface-hub discord.py httpx openai pyyaml

# 2. Download and quantize the model
hf download zimengxiong/talkie-1930-13b-it-mlx --local-dir ~/.omlx/models/talkie-1930-13b-it-mlx
# Then quantize to 8-bit (see full guide)

# 3. Start the inference server
python server.py --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit --port 8080

# 4. Configure and start the Discord bot
cd ~/Projects/llmcord
python llmcord.py
```

## Documentation Index

| Document | Description |
|----------|-------------|
| **[USER_GUIDE.md](USER_GUIDE.md)** | How to use the bot as a Discord user — mentions, replies, commands |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design, data flow, component diagrams, and internal mechanics |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues, error messages, and fixes |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Setting up a dev environment, testing, and contributing |
| **[API.md](API.md)** | OpenAI-compatible API reference with request/response examples |

## System Requirements

| Requirement | Details |
|---|---|
| **Apple Silicon Mac** | M1 or later. 16 GB+ unified memory minimum. 48 GB recommended. |
| **macOS** | 15.0+ (Sequoia) |
| **Python** | 3.13+ (3.14 tested) |
| **Disk space** | ~13 GB for 8-bit model, ~25 GB for BF16 |
| **Discord account** | A server where you can add bots |

## Performance (M5 Pro 48 GB, 8-bit)

| Metric | Value |
|--------|-------|
| Generation speed | ~16.8 tok/s |
| Prefill speed | ~600 tok/s |
| Peak memory (4K context) | ~17 GB |
| Model size on disk | ~13 GB |
| Server startup time | ~5 seconds |

## Credits

- **Talkie model**: Alec Radford, Nick Levine, David Duvenaud — [talkie-lm.com](https://talkie-lm.com)
- **MLX Talkie support**: ZimengXiong — [PR #1220](https://github.com/ml-explore/mlx-lm/pull/1220)
- **Pre-converted MLX weights**: [zimengxiong/talkie-1930-13b-it-mlx](https://huggingface.co/zimengxiong/talkie-1930-13b-it-mlx)
- **llmcord**: jakobdylanc — [github.com/jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord)
- **Talkie MLX server & Discord integration**: gwyntel — [github.com/gwyntel/talkie-mlx-server](https://github.com/gwyntel/talkie-mlx-server)

## License

The Talkie model and server code are provided for research and personal use. See the [Talkie repository](https://github.com/talkie-lm/talkie) for model license details.
