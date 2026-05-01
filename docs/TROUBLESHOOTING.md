# TalkieLM Troubleshooting Guide

Solutions to common problems, error messages, and misconfigurations.

---

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

```bash
# 1. Is the inference server running?
curl -s http://127.0.0.1:8080/v1/models
# Expected: {"object":"list","data":[{"id":"talkie-1930-13b-it-mlx-8bit",...}]}

# 2. Is the Discord bot connected?
tail -5 ~/Library/Logs/llmcord.log
# Look for: "Logged in as Talkie 1930#..."

# 3. Are there errors?
cat ~/Library/Logs/talkie-server.err
cat ~/Library/Logs/llmcord.err

# 4. Are both launchd services running?
launchctl list | grep gwyntel
# Should show two PIDs (not "-" for exit status)
```

---

## Common Issues

### Bot doesn't respond to messages at all

**Symptoms:** The bot appears online in Discord but never replies, no matter how you @mention it.

**Root cause #1: MESSAGE CONTENT INTENT not enabled**

The most common cause. Without this privileged intent, the bot can see that messages exist but cannot read their content — `message.content` is always empty.

**Fix:**
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Select your application → Bot section
3. Scroll to **Privileged Gateway Intents**
4. Enable **MESSAGE CONTENT INTENT**
5. Restart llmcord

**Root cause #2: Inference server is down**

If the server crashed or never started, llmcord will show "typing..." briefly then time out with an error.

**Fix:**
```bash
curl http://127.0.0.1:8080/v1/models
# If no response:
launchctl kickstart -k gui/$(id -u)/com.gwyntel.talkie-server
# Or start manually:
source ~/mlx-env/bin/activate
python ~/.omlx/models/talkie-1930-13b-it-mlx-8bit/server.py \
  --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit --port 8080
```

**Root cause #3: Wrong bot token or client ID**

Check that `bot_token` and `client_id` in `config.yaml` match your Discord application.

---

### Bot shows "typing..." but never replies

**Symptoms:** You @mention the bot, the typing indicator appears, but no message is ever sent.

**Root cause #1: Server not using SSE streaming**

llmcord uses the OpenAI SDK with `stream=True`, which expects `text/event-stream` responses with `data:`-prefixed chunks. If using `mlx_lm server` instead of the custom `server.py`, streaming won't work.

**Fix:** Make sure you're using the custom `server.py` from the talkie-mlx-server repo. The standard `mlx_lm server` does NOT work with Talkie for multiple reasons (see below).

**Root cause #2: Server crashed during generation**

Check the server logs for exceptions:
```bash
tail -20 ~/Library/Logs/talkie-server.err
```

**Root cause #3: Network/localhost issue**

Make sure the `base_url` in `config.yaml` points to `http://127.0.0.1:8080/v1` (with `/v1` suffix).

---

### Bot replies with only 1–2 sentences

**Symptoms:** The bot starts well but stops after barely a sentence or two. Responses feel truncated.

**Root cause #1: EOS suppression not working (server bug)**

This was a real bug in earlier versions where streaming responses bypassed the EOS suppression logits processor. The fix is in the current `server.py` — both streaming and non-streaming paths use `_generate_with_eos_suppress()`.

**How to verify:**
```bash
# Check server startup log for:
tail -20 ~/Library/Logs/talkie-server.log | grep "EOS bias"
# Expected: "EOS bias: -100.0 for first 80 tokens"
```

**Root cause #2: `max_tokens` too low**

The server defaults to 1024 tokens for the response. If this was changed or overridden, responses will be cut short.

**Fix:** In `server.py`, check `MAX_TOKENS_DEFAULT`. It should be at least 1024. llmcord does NOT send `max_tokens` in its API call, so the server default is always used.

**Root cause #3: Context window overflow**

If the input (system prompt + conversation history) fills most of the 4096 token context, there's little room left for the response.

**Fix:**
- Use `/context show` to check token usage
- Reduce `max_messages` in `config.yaml` (try 5 instead of 10)
- Use `/context clear` to reset

---

### Bot goes offline after a while

**Symptoms:** The bot works for a while, then disappears from Discord.

**Root cause: Process crashed without auto-restart**

Without launchd, processes started in a terminal die when:
- The terminal is closed
- The machine goes to sleep and the process errors out on wake
- An unhandled exception crashes the process

**Fix:** Use launchd with `KeepAlive: true`:
```bash
launchctl load ~/Library/LaunchAgents/com.gwyntel.talkie-server.plist
launchctl load ~/Library/LaunchAgents/com.gwyntel.llmcord.plist
```

---

### Bot replies out of character / as a modern AI assistant

**Symptoms:** The bot sounds like ChatGPT instead of an Edwardian gentleman. It mentions modern technology, uses modern slang, or breaks the 1920s persona.

**Root cause #1: Missing or bad system prompt**

Without the Edwardian-era system prompt, Talkie defaults to generic helpful-assistant behavior it learned during RL refinement.

**Fix:** Make sure `system_prompt` is set in `config.yaml`. See the full prompt in the setup guide. Key elements:
- "You are an Edwardian-era conversationalist speaking from the year nineteen hundred and twenty-nine"
- "You know nothing of events after approximately nineteen thirty"
- The "discord as wireless telegraph" framing

**Root cause #2: Context contamination**

If the conversation includes messages from a modern-day perspective, the model may follow that style instead of its persona.

**Fix:** Use `/context clear` and start a new conversation.

---

### OOM (Out of Memory) crash

**Symptoms:** The server process crashes with memory errors, or macOS shows system memory pressure warnings.

**Root cause:** Too much context for available memory.

| Config | Observed Peak RAM (8-bit) |
|--------|---------------------------|
| 2K context | ~15.7 GB |
| 4K context | ~17.0 GB |
| 8K context | ~21.1 GB |
| 16K+ context | ~24+ GB |

**Fix (in order of impact):**

1. **Reduce `max_messages`** in llmcord `config.yaml` (try 5 instead of 10)
2. **Reduce `max_seq_len`** to 2048 in model `config.json` (saves ~2 GB)
3. **Ensure no other GPU-heavy apps** are running (Final Cut, Blender, etc.)
4. **Upgrade to a Mac with more unified memory** (32 GB+ recommended, 48 GB comfortable)

---

### "Repo id must be in the form 'namespace/repo_name'"

**Symptoms:** Server fails to start with this error from HuggingFace.

**Root cause:** The `--model-dir` path was not recognized as a local directory, so `mlx_lm.load()` tried to interpret it as a HuggingFace repo ID.

**Fix:** Always specify `--model-dir` with an **absolute path**:
```bash
python server.py --model-dir ~/.omlx/models/talkie-1930-13b-it-mlx-8bit
```
Or `cd` into the model directory first (the default is `.`).

---

### "Model type talkie not supported"

**Symptoms:** Error when trying to load the model with oMLX.

**Root cause:** oMLX has a model registry that only recognizes standard architectures (Llama, Mistral, etc.). Talkie's custom architecture (`model_type: "talkie"`) isn't in it.

**Fix:** Use `mlx-lm` from the `talkie-support` PR branch:
```bash
source ~/mlx-env/bin/activate
pip install "mlx-lm @ git+https://github.com/ZimengXiong/mlx-lm.git@talkie-support"
```

---

### Response text is garbled or incoherent

**Symptoms:** Bot's replies don't make sense, mix topics, or contain character salad.

**Root cause #1: Context window overflowed**

When the total tokens (system prompt + history + response) exceed 4096, the model's attention mechanism degrades because it was only trained at 2048.

**Fix:**
1. `/context clear` to reset
2. Reduce `max_messages` in `config.yaml`
3. Use `/context show` to monitor token usage

**Root cause #2: `max_messages` too high**

With `max_messages: 25` (the llmcord default), a long conversation can easily overflow 4096 tokens.

**Fix:** Set `max_messages: 10` (or lower for short conversations).

---

### Bot doesn't reply to DMs

**Symptoms:** Bot works in server channels but ignores DMs.

**Root cause:** `allow_dms: false` in `config.yaml`.

**Fix:** Set `allow_dms: true` in `config.yaml` and restart llmcord.

---

### Discord mentions appear as raw text in bot's response

**Symptoms:** Bot's response contains `<@123456789>` literally instead of rendering as a Discord ping.

**Root cause:** The `era_to_discord()` postprocessor is not being applied. This shouldn't happen in the current code, but could occur if someone modified the streaming callback.

**Fix:** Ensure the streaming display code in `llmcord.py` calls `era_to_discord()` on the response text before sending it to Discord:
```python
display_text = era_to_discord(response_contents[-1])
```

---

### Bot mentions someone as "Correspondent No. 123456789" instead of pinging

**Symptoms:** Bot's response contains literal "Correspondent No. 123456789:" text that isn't converted to a Discord ping.

**Root cause:** The `era_to_discord()` regex didn't match. This can happen if:
1. The bot adds extra whitespace or punctuation around the correspondent number
2. The model generates the correspondent ID in a slightly different format

**Fix:** This is typically a model generation quirk. The regex matches `Correspondent No.\s*(\d+)(:)` — if the model generates "Correspondent No. 123." (with period instead of colon), it won't match. The system prompt instructs the model to use the correct format, so this is rare.

---

### Server takes very long to generate / extremely slow responses

**Symptoms:** Responses take 30+ seconds for short messages.

**Root cause #1: Memory pressure**

macOS may be swapping if the model plus other apps exceed unified memory. Check Activity Monitor for swap usage.

**Fix:** Close other GPU-intensive apps. Ensure the model fits in memory with headroom.

**Root cause #2: BF16 instead of 8-bit**

BF16 runs at ~10 tok/s vs ~17 tok/s for 8-bit. If you accidentally loaded the BF16 model:

```bash
# Check which model is loaded:
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

---

## Reading the Logs

Both services produce detailed logs. In DEBUG mode, they are "obsessive" — logging every chunk, every message, every token.

### Server logs (`~/Library/Logs/talkie-server.log`)

| Log line | Meaning |
|----------|---------|
| `[GENERATE] 3.21s, 245 chars, max_tokens=1024` | Generation stats |
| `[CHAT_STREAM] req_id=... starting generation prompt_tokens=350` | Starting stream |
| `[CHAT_STREAM] req_id=... DONE chunks_sent=47 chars_sent=245` | Stream complete |
| `[POST] chat/completions (stream) total_time=5.2s peak_mem=16.8GB` | Request complete |

### Bridge logs (`~/Library/Logs/llmcord.log`)

| Log line | Meaning |
|----------|---------|
| `Message received (user ID: ... conversation length: 5)` | Incoming message |
| `[API_CALL] model=talkie-1930-13b-it-mlx-8bit stream=True messages_count=6` | API call |
| `[STREAM_DONE] chunks=47 parts_assembled=245 segments_assembled=245` | Stream complete |
| `[MISMATCH] parts (250) != segments (245)` | **BUG** — streaming assembly mismatch |

### Enabling DEBUG logging

For the server, set the `--log-level DEBUG` flag or `LOG_LEVEL=DEBUG` environment variable.

For llmcord, edit the logging level in `llmcord.py`:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

**Warning:** DEBUG logging is extremely verbose. Every SSE chunk, every message node operation, and every token is logged. Only enable this when actively debugging.

---

## Known Issues & Limitations

### 4-bit quantization is broken
The `sanitize()` method in the PR branch can't handle `.scales`/`.biases` weight names. **Use 8-bit only.**

### oMLX is unsupported
oMLX's model registry doesn't recognize `model_type: "talkie"`. Use mlx-lm directly.

### Streaming is "batch-then-stream"
The server generates the full response first, then streams it word-by-word. Total latency is the same as non-streaming, but the progressive display gives a better UX.

### No image support
Talkie is text-only. Set `max_images: 0` in llmcord config.

### No tool calling or function calling
Talkie was not trained for tool use. The API supports chat and text completions only.

### No persistent memory
llmcord has no database or memory backend. Context is built from the reply chain on each message. Old conversations are forgotten once they drop out of the message cache.

### Reply chain context only (without /context reopen)
By default, the bot only sees messages in the current reply chain. Use `/context reopen` for multi-participant conversations.
