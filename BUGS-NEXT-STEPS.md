# Talkie 1930 — Bugs & Next Steps Briefing

> **Last updated:** 2026-05-01 15:00 PT (Friday) — feature additions by same session.
> **Status:** Two new features implemented (status display + auto context merging). Server hang bug still needs ThreadingHTTPServer fix.

---

## System Overview

| Component | File | Role |
|-----------|------|------|
| **server.py** | `~/Projects/talkie-mlx-server/server.py` | OpenAI-compatible MLX inference server on `127.0.0.1:8080` |
| **llmcord.py** | `~/Projects/llmcord/llmcord.py` | Discord bot (discord.py async) connecting to the server |
| **config.yaml** | `~/Projects/llmcord/config.yaml` | Bot config: provider URL, model name, system prompt |

- **Model:** Talkie 1930 13B IT, 8-bit quantized  
  `~/.omlx/models/talkie-1930-13b-it-mlx-8bit`
- **Python:** Both components use `~/mlx-env` (virtualenv, NOT system Python)
- **Process management:** Both run via `launchd` (launchd plists)
- **Repos:**
  - `~/Projects/talkie-mlx-server` → `github.com/gwyntel/talkie-mlx-server`
  - `~/Projects/llmcord` → `github.com/gwyntel-git/talkie-llmcord`

### Git state at time of writing

| Repo | Branch | HEAD commit | Uncommitted? |
|------|--------|-------------|--------------|
| talkie-mlx-server | `main` | `de21940` | Yes — constant fix on disk (see Bug #2) |
| talkie-llmcord | `main` | `536a794` | — |

---

## BUG #1: `stream=False` Request Hangs (CRITICAL — ACTIVE)

### Symptom

When llmcord sends a `POST /v1/chat/completions` with `stream=False`, the server accepts the request but **never returns** for longer generation requests. Short requests (`max_tokens=50`) complete instantly.

**Evidence from server log:**
```
14:36:01 POST /v1/chat/completions stream=False max_tokens=1024
```
No subsequent `[GENERATE]` or `[CHAT]` completion log appears. The server appears deadlocked.

### Root Cause Analysis

The server uses `http.server.HTTPServer` + `BaseHTTPRequestHandler` — **single-threaded, synchronous**. The call chain for non-streaming is:

```
do_POST → _chat_completion() → _generate_full() → _generate_with_eos_suppress() → mlx_lm.generate()
```

`mlx_lm.generate()` is a synchronous blocking call that occupies the only server thread. For short generations this is fast enough; for longer ones the server is unresponsive for the duration.

**Critical additional clue:** Server process showed **0% CPU** during the hang. This suggests the process is NOT actually generating — it may be stuck on something else entirely:
- The EOS suppression processor could be throwing a silent error
- The prompt could exceed the model's context window (`max_seq_len`)
- A deadlock in the MLX Metal framework (GPU sync issue)

On the **llmcord side** (`llmcord.py:551-552`):
```python
async with new_msg.channel.typing():
    response = await openai_client.chat.completions.create(**openai_kwargs)
```
The `AsyncOpenAI` client does an async HTTP call, but if the server is blocked and can't write the response, the client just awaits forever. No `timeout` parameter is set.

### Fix Plan (ordered by priority)

1. **Replace `HTTPServer` with `ThreadingHTTPServer`** in `server.py` line 505:
   ```python
   # Current:
   server = HTTPServer((args.host, args.port), Handler)
   # Fix:
   from http.server import ThreadingHTTPServer
   server = ThreadingHTTPServer((args.host, args.port), Handler)
   ```
   This allows the server to handle blocking generation without freezing.

2. **Add `timeout=60.0`** to the OpenAI call in `llmcord.py` line 531:
   ```python
   openai_kwargs = dict(model=model, messages=messages[::-1], stream=False,
                         extra_headers=extra_headers, extra_query=extra_query,
                         extra_body=extra_body, timeout=60.0)
   ```
   Prevents the bot from hanging indefinitely.

3. **Add generation-started logging** in `server.py` — `_generate_with_eos_suppress()` should log immediately upon entry, BEFORE calling `generate()`. Currently the `[GENERATE]` log only appears after generation completes, which is useless for diagnosing hangs.

4. **Test with intermediate `max_tokens`** (e.g., `200`) to narrow down whether it's a timing or a deadlock issue.

5. **Investigate the 0% CPU mystery** — if `generate()` isn't running, what is happening? Try wrapping the `generate()` call in a `try/except` that logs any exception, and add a thread timer that logs if generation hasn't started within 5 seconds.

---

## BUG #2: Hermes Sanitizes Numeric Constants (RECURRING — META-BUG)

### Symptom

Hermes Agent (the AI tool editing these files) replaces integer/float literals with `***` when reading Python source. If a file is then patched or rewritten based on that read, `***` gets written back as a literal, causing `SyntaxError`.

**This has already happened TWICE** to these constants in `server.py`:
- `MAX_TOKENS_DEFAULT` (should be `1024`) — line 61
- `_MIN_RESPONSE_TOKENS` (should be `80`) — line 72

### Current State ON DISK

As of this writing, `server.py` lines 61 and 72 still contain `***`:
```python
MAX_TOKENS_DEFAULT=***          # line 61 — MUST be 1024
_MIN_RESPONSE_TOKENS=***         # line 72 — MUST be 80
```

These **must be fixed** before the server can run. The running server process (PID 57704) has stale values loaded in memory and must be restarted after fixing.

### Mitigation Protocol

After ANY file edit performed by Hermes Agent:
1. `grep -n '=\*\*\*\|=\*\*' ~/Projects/talkie-mlx-server/server.py`
2. If found, manually restore the correct values:
   - `MAX_TOKENS_DEFAULT = 1024` (line 61)
   - `_MIN_RESPONSE_TOKENS = 80` (line 72)
3. Delete `__pycache__`: `rm -rf ~/Projects/talkie-mlx-server/__pycache__`
4. **Restart the server** — in-memory values are stale until process restart

### Why This Happens

Hermes sanitizes personally identifiable information (PIIs), and numeric literals sometimes trigger that filter. There is no known fix other than post-edit verification.

---

## BUG #3: Streaming Disabled but Still Compiled In (LOW PRIORITY)

### Status

`llmcord.py` now uses `stream=False` exclusively (line 531). The server still contains full SSE streaming code:

- `_chat_completion_stream()` (line 154)
- `_generate_tokens()` (line 116)
- `_sse_response()` (line 350)

This is harmless (backward compat, no performance impact for disabled path) but the server logs `"SSE streaming: enabled"` on startup (line 503), which is misleading.

**No action required** unless you want to clean up. If removed, also remove the misleading log line.

---

## Architecture Reference

### server.py

- `BaseHTTPRequestHandler` — single-threaded, synchronous
- `mlx_lm.load()` for model init, `mlx_lm.generate()` for inference
- EOS suppression via `logits_processors` callback — biases tokens `{65535, 65536, 65537, 65539}` by -100.0 for the first 80 generated tokens (prevents premature `<|end|>` stopping)
- Context window: 4096 tokens (`_model.args.max_seq_len`)
- Port: 8080 (configurable via `--port`)

### llmcord.py

- `discord.py` async bot with `commands.Bot`
- `openai.AsyncOpenAI` client pointing at `http://127.0.0.1:8080/v1`
- **Non-streaming only** (`stream=False` in `openai_kwargs`, line 531)
- Era-appropriate translation: Discord `<@ID>` → "Correspondent No. ID" → `era_to_discord()` regex back to `<@ID>`
- Mutable context: `/context reopen` opens 15-minute window for multi-user conversations

### config.yaml

- Provider: `talkie` at `http://127.0.0.1:8080/v1`, no API key needed
- Model: `talkie/talkie-1930-13b-it-mlx-8bit`
- System prompt: Edwardian-era persona, date/time tags, correspondent IDs
- `use_plain_responses: false` (uses Discord embeds)
- `max_text: 3000`, `max_images: 0`, `max_messages: 10`

---

## Key File Locations

| What | Path |
|------|------|
| MLX inference server | `~/Projects/talkie-mlx-server/server.py` |
| Discord bot | `~/Projects/llmcord/llmcord.py` |
| Bot configuration | `~/Projects/llmcord/config.yaml` |
| Server log | `~/Library/Logs/talkie-server.log` |
| Bot log | `~/Library/Logs/llmcord-v6.log` |
| Model weights | `~/.omlx/models/talkie-1930-13b-it-mlx-8bit` |
| Python virtualenv | `~/mlx-env` |

---

## Immediate Next Steps (Prioritized)

### Step 1: Fix `***` constants in server.py

```bash
# Edit these two lines:
# Line 61: MAX_TOKENS_DEFAULT=***  →  MAX_TOKENS_DEFAULT = 1024
# Line 72: _MIN_RESPONSE_TOKENS=***  →  _MIN_RESPONSE_TOKENS = 80
```

> **IMPORTANT:** The correct values are one-thousand-and-twenty-four (for MAX_TOKENS_DEFAULT)
> and eighty (for _MIN_RESPONSE_TOKENS). The sanitizer may have replaced those digits.

Verify:
```bash
grep -n 'MAX_TOKENS_DEFAULT\|_MIN_RESPONSE_TOKENS' ~/Projects/talkie-mlx-server/server.py
rm -rf ~/Projects/talkie-mlx-server/__pycache__
```

### Step 2: Replace HTTPServer with ThreadingHTTPServer

In `server.py`:
- Add `ThreadingHTTPServer` to the import on line 34:  
  `from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler`
- Change line 505:  
  `server = ThreadingHTTPServer((args.host, args.port), Handler)`

### Step 3: Add timeout to llmcord OpenAI call

In `llmcord.py` line 531, add `timeout=60.0` to `openai_kwargs`:
```python
openai_kwargs = dict(model=model, messages=messages[::-1], stream=False,
                     extra_headers=extra_headers, extra_query=extra_query,
                     extra_body=extra_body, timeout=60.0)
```

### Step 4: Add generation-started logging

In `_generate_with_eos_suppress()` (server.py line 93), add a log BEFORE the `generate()` call:
```python
logger.info(f"[GENERATE] STARTING max_tokens={max_tokens} prompt_len={len(prompt)}")
```

### Step 5: Restart both services

```bash
# Via launchd (or however the services are managed):
launchctl kickstart -k gui/$(id -u)/talkie-server
launchctl kickstart -k gui/$(id -u)/llmcord
# Or kill + restart manually if launchd plist paths differ
```

### Step 6: Verify the fix

```bash
# Quick smoke test:
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool

# Non-streaming generation test:
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"talkie-1930-13b-it-mlx-8bit","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50,"stream":false}' \
  | python3 -m json.tool

# Longer generation test (the one that hangs):
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"talkie-1930-13b-it-mlx-8bit","messages":[{"role":"user","content":"Tell me about the marvels of the wireless telegraph."}],"max_tokens":200,"stream":false}' \
  | python3 -m json.tool

# Then test from Discord
```

### Step 7: Push fixes to GitHub

```bash
cd ~/Projects/talkie-mlx-server
git add -A && git commit -m "fix: use ThreadingHTTPServer, fix corrupted constants, add generation logging"
git push origin main

cd ~/Projects/llmcord
git add -A && git commit -m "fix: add timeout to OpenAI client call"
git push origin main
```

---

## Open Questions

1. **What caused 0% CPU during the hang?** If `generate()` was running, CPU should be nonzero. This needs investigation — possibly an MLX Metal deadlock or a silent exception being swallowed somewhere.
2. **Should we re-enable streaming?** It worked previously for the bot and avoids the blocking-request problem entirely. The non-streaming switch was made as a workaround for a different issue (truncated responses). Since EOS suppression is now in place, streaming might work correctly again.
3. **`/health` endpoint:** The current server has no health check. Adding one that does a tiny generation (e.g., `max_tokens=1`) would verify the full inference pipeline, not just that the HTTP server is up.

---

## Features Implemented (2026-05-01 15:00 PT)

### Feature 1: Dynamic Bot Status (tokens/6h + uptime)

The bot's Discord status (custom activity) now updates every 60 seconds with:
```
Talking like it's 1929 · 12.4k tok/6h · up 3h 22m
```

**Implementation:**
- `_bot_start_time`: recorded at module load
- `_output_log`: `deque` of `(timestamp, estimated_tokens)` — pruned to 6h window on each status update
- `_status_updater()`: background asyncio task, runs every 60s, updates `discord.CustomActivity`
- Token estimation: uses `response.usage.completion_tokens` if available, else `len(raw_text) // 4`
- `/context status` now also shows uptime and token output count

**Key functions:** `_format_uptime()`, `_format_tokens()`, `_build_status_text()`

### Feature 2: Auto Multi-User Context Merging (Reply-Graph Stitching)

When Person B replies to Person A's message in a channel where the bot has also been talking to Person A, the bot automatically stitches the orphan bot response into Person B's context. This lets separate conversations merge at reply-graph intersection points, without forcing everything into a shared pool.

**How it works:**
1. After the standard reply-chain walk, collect `chain_msg_ids` — the set of Discord message IDs walked
2. Scan channel history for "orphan" bot replies: messages where the bot replied to a user message that's IN our chain, but the bot's reply itself is NOT in our chain
3. Also scan for users who reply to bot messages that are in our chain but from a different conversation branch
4. Merge found items chronologically (oldest first, prepended before the chain)

**Example scenario:**
- Person A: `@bot hello` → Bot: `Greetings!` (A's chain)
- Person B: `@bot what's up` → Bot: `Quite well!` (B's separate chain)
- Person B replies to Person A's `hello`: Bot now sees A's conversation (A's message + bot response) AND B's message when building context for its response

**Gating:** Only activates in guild channels (not DMs). Respects `max_messages` limit. Deduplicates by message ID.
