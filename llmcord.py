import asyncio
from base64 import b64encode
from collections import deque
import json
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
import time
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# --- Vision proxy: describe images via a local VLM in Edwardian voice ---
# When enabled, image attachments are sent to a separate vision model which
# describes them in period-appropriate language. The description is then
# injected into the text message so the text-only Talkie model can "see".

VISION_PROXY_SYSTEM = (
    "You are describing a photograph for a person living in the year 1929. "
    "Describe ONLY what is visible in the image using the language and "
    "vocabulary of an educated Edwardian gentleman. Refer to modern objects "
    "using the nearest equivalent from your era (e.g. 'motor-car' not 'car', "
    "'wireless set' not 'radio', 'aeroplane' not 'plane'). Do not mention "
    "anything that could not exist in 1929. Be specific and vivid but concise "
    "(2-4 sentences). Do not preface with 'I see' or 'This shows' — simply "
    "describe."
)


async def _describe_image_via_vision_proxy(
    b64_image: str,
    content_type: str,
    vision_url: str,
    vision_model: str,
    httpx_client: httpx.AsyncClient,
) -> str | None:
    """Send an image to a local VLM and get an Edwardian-era description."""
    try:
        import logging as _log
        _logger = _log.getLogger("vision_proxy")
        data_url = f"data:{content_type};base64,{b64_image}"
        payload = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": VISION_PROXY_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this photograph."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "max_tokens": 150,
            "temperature": 0.4,
            "stream": False,
        }
        resp = await httpx_client.post(
            f"{vision_url}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            desc = body["choices"][0]["message"]["content"].strip()
            _logger.info(f"[VISION_PROXY] Image described: {desc[:80]}...")
            return desc
        else:
            _logger.warning(f"[VISION_PROXY] VLM returned {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception:
        import logging as _log
        _log.getLogger("vision_proxy").exception("[VISION_PROXY] Failed to describe image")
        return None

# --- Era-format <-> Discord ping translator ---
# Talkie sees "Correspondent No. 123456789:" but Discord pings with <@123456789>
_CORRESPONDENT_PATTERN = re.compile(r"Correspondent No\.\s*(\d+)(:)")

def era_to_discord(text: str) -> str:
    """Convert era-appropriate 'Correspondent No. N:' back to Discord <@N> pings."""
    return _CORRESPONDENT_PATTERN.sub(r"<@\1>:", text)


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

# Per-channel/thread mutable context tracking
# When a user runs /context open, subsequent messages in that channel can be pulled
t_mutable_context = {}  # channel_id -> {opener_id, opened_at, ttl_seconds}
MUTABLE_CONTEXT_TTL = 900  # 15 minutes

# ── Token output tracking + uptime (for dynamic bot status) ──────────────
_bot_start_time = time.time()
_output_log: deque = deque()  # (timestamp, estimated_token_count) — pruned every update
_STATUS_UPDATE_INTERVAL = 60  # seconds between status updates
_TOKEN_WINDOW_SECONDS = 6 * 3600  # rolling 6h window

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


# ── Dynamic status updater ────────────────────────────────────────────────
def _format_uptime(seconds: float) -> str:
    """Human-friendly uptime: '3h 22m', '47m', '2d 5h'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def _format_tokens(count: int) -> str:
    """Compact token count: '1.2k', '12.4k', '142k'."""
    if count < 1000:
        return str(count)
    elif count < 100_000:
        return f"{count / 1000:.1f}k"
    else:
        return f"{count / 1000:.0f}k"


def _build_status_text() -> str:
    """Build the bot status string: '<base> · X.Xk tokens/6h · up Nh Nm'."""
    base = config.get("status_message") or "Talking like it's 1929"
    now = time.time()

    # Prune old entries
    while _output_log and (now - _output_log[0][0]) > _TOKEN_WINDOW_SECONDS:
        _output_log.popleft()

    total_tokens = sum(t[1] for t in _output_log)
    uptime = _format_uptime(now - _bot_start_time)
    token_str = _format_tokens(total_tokens)

    status = f"{base} · {token_str} tok/6h · up {uptime}"
    return status[:128]  # Discord limit


async def _status_updater():
    """Background task: update bot status every _STATUS_UPDATE_INTERVAL seconds."""
    await discord_bot.wait_until_ready()
    while not discord_bot.is_closed():
        try:
            text = _build_status_text()
            activity = discord.CustomActivity(name=text)
            await discord_bot.change_presence(activity=activity)
        except Exception:
            logging.exception("[status_updater] failed to update status")
        await asyncio.sleep(_STATUS_UPDATE_INTERVAL)


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.tree.command(name="context", description="View context, clear memory cache, or show bot status")
async def context_command(
    interaction: discord.Interaction,
    action: str,
) -> None:
    channel = interaction.channel

    # ── /context status ──────────────────────────────────────────────
    if action == "status":
        max_messages = config.get("max_messages", 25)
        max_seq_len = 4096
        system_prompt_enabled = bool(config.get("system_prompt"))
        system_tokens_est = 200 if system_prompt_enabled else 0
        cached_nodes = len(msg_nodes)
        max_nodes = MAX_MESSAGE_NODES
        uptime = _format_uptime(time.time() - _bot_start_time)

        # Count tokens in 6h window
        now_ts = time.time()
        recent_tokens = sum(t[1] for t in _output_log if now_ts - t[0] < _TOKEN_WINDOW_SECONDS)
        tok_str = _format_tokens(recent_tokens)

        lines = [
            f"**📊 Bot Status**\n",
            f"```\n"
            f"Model:            {curr_model}\n"
            f"Max messages:      {max_messages}\n"
            f"Context window:    {max_seq_len} tokens\n"
            f"System prompt:     {'✅ enabled' if system_prompt_enabled else '❌ disabled'} (~{system_tokens_est} tokens)\n"
            f"Msg cache:         {cached_nodes} / {max_nodes} nodes\n"
            f"Allow DMs:        {'✅' if config.get('allow_dms', False) else '❌'}\n"
            f"Max text:          {config.get('max_text', 100000):,} chars\n"
            f"Max images:        {config.get('max_images', 0)}\n"
            f"Plain responses:   {'✅' if config.get('use_plain_responses', False) else '❌'}\n"
            f"```",
            f"⏱️ Uptime: **{uptime}** — Output: **{tok_str} tokens** in last 6h",
        ]

        # Check if Talkie server is reachable
        try:
            import httpx as _hx
            async with _hx.AsyncClient() as _c:
                _r = await _c.get(config["providers"][curr_model.split("/")[0]]["base_url"].rstrip("/") + "/models", timeout=5.0)
                if _r.status_code == 200:
                    model_list = _r.json().get("data", [])
                    server_model = model_list[0]["id"] if model_list else "unknown"
                    lines.append(f"🔌 Inference server: **online** (`{server_model}`)")
                else:
                    lines.append(f"🔌 Inference server: ⚠️ responding with status {_r.status_code}")
        except Exception:
            lines.append("🔌 Inference server: **offline** — bot will not be able to respond")

        # Check vision proxy
        vision_proxy = config.get("vision_proxy")
        if vision_proxy and vision_proxy.get("enabled"):
            vp_model = vision_proxy.get("model", "unknown")
            vp_url = vision_proxy["base_url"].rstrip("/") + "/v1"
            try:
                import httpx as _hx
                async with _hx.AsyncClient() as _c:
                    _r = await _c.get(f"{vp_url}/models", timeout=5.0)
                    if _r.status_code == 200:
                        vlm_list = _r.json().get("data", [])
                        vlm_model = vlm_list[0]["id"] if vlm_list else "unknown"
                        # Friendly short name
                        vlm_short = vlm_model.rsplit("/", 1)[-1] if "/" in vlm_model else vlm_model
                        lines.append(f"👁️ Vision proxy: **online** (`{vlm_short}`)")
                    else:
                        lines.append(f"👁️ Vision proxy: ⚠️ responding with status {_r.status_code}")
            except Exception:
                lines.append("👁️ Vision proxy: **offline** — images will show as *\"wireless too faint\"*")
        else:
            lines.append("👁️ Vision proxy: ❌ disabled")

        embed = discord.Embed(description="\n".join(lines), color=discord.Color.blurple())
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return

    # ── /context clear ───────────────────────────────────────────────
    if action == "clear":
        if channel is None:
            await interaction.response.send_message("⚠️ This can only be used in a channel or DM.", ephemeral=True)
            return

        # Count before
        before = len(msg_nodes)

        if channel.type == discord.ChannelType.private:
            # DMs: clear all nodes for this DM channel
            to_remove = [mid for mid, node in msg_nodes.items() if node.parent_msg is not None]
            # Simpler approach: clear everything for a DM reset
            msg_nodes.clear()
            after = 0
        else:
            # Guild channel: clear nodes whose messages belong to this channel
            # We can't easily know which nodes belong to which channel from the
            # node data alone, so clear the entire cache (it rebuilds on next message)
            msg_nodes.clear()
            after = 0

        cleared = before - after
        output = f"🗑️ Cleared **{cleared}** message{'s' if cleared != 1 else ''} from cache.\n\nNext reply to the bot will start with fresh context. The cache rebuilds automatically as new messages are processed."

        embed = discord.Embed(description=output, color=discord.Color.red())
        await interaction.response.send_message(embed=embed, ephemeral=True)
        logging.info(f"[context clear] Purged {cleared} message nodes from cache (invoked by {interaction.user})")
        return

    # ── /context reopen ───────────────────────────────────────────────
    if action == "reopen":
        if channel is None:
            await interaction.response.send_message("⚠️ This can only be used in a channel or DM.", ephemeral=True)
            return

        t_mutable_context[channel.id] = dict(
            opener_id=interaction.user.id,
            opened_at=time.time(),
            ttl_seconds=MUTABLE_CONTEXT_TTL
        )
        output = (
            f"📂 **This conversation is now open for 15 minutes.**\n\n"
            f"Any user who @-mentions the bot in this channel will be included in the context.\n\n"
            f"Use `/context clear` to reset manually, or wait for the timer to expire."
        )
        embed = discord.Embed(description=output, color=discord.Color.green())
        await interaction.response.send_message(embed=embed, ephemeral=False)
        logging.info(f"[context reopen] Channel {channel.id} opened by {interaction.user.id} for {MUTABLE_CONTEXT_TTL}s")
        return

    # ── /context show (default) ──────────────────────────────────────
    if channel is None:
        await interaction.response.send_message("⚠️ This can only be used in a channel or DM.", ephemeral=True)
        return

    # Try to get the most recent message in the channel to trace context from
    try:
        history = [m async for m in channel.history(limit=1)]
        target_msg = history[0] if history else None
    except (discord.Forbidden, discord.HTTPException):
        target_msg = None

    if not target_msg or target_msg.author == discord_bot.user:
        # If the latest message is from the bot, try one more back
        try:
            history = [m async for m in channel.history(limit=2)]
            target_msg = next((m for m in history if m.author != discord_bot.user), None)
        except (discord.Forbidden, discord.HTTPException):
            target_msg = None

    if not target_msg:
        await interaction.response.send_message("⚠️ Could not find a message to trace context from. Try using this in a channel with messages.", ephemeral=True)
        return

    # Replicate the context-building logic from on_message
    max_messages = config.get("max_messages", 25)
    max_text = config.get("max_text", 100000)
    messages = []
    curr_msg = target_msg
    chain_count = 0

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.get(curr_msg.id)

        if curr_node and curr_node.text is not None:
            text = curr_node.text
            role = curr_node.role
        else:
            # Message not in cache — build a summary from what we can see
            role = "assistant" if curr_msg.author == discord_bot.user else "user"
            text = curr_msg.content[:max_text] if curr_msg.content else "(no text / not in cache)"

        content = text[:max_text] if text else "(empty)"
        if content:
            messages.append(dict(role=role, content=content, author=curr_msg.author.display_name))

        chain_count += 1

        # Walk the reply chain
        parent = None
        try:
            if curr_msg.reference and curr_msg.reference.message_id:
                parent = curr_msg.reference.cached_message or await channel.fetch_message(curr_msg.reference.message_id)
            elif curr_msg.channel.type == discord.ChannelType.public_thread:
                # In threads without explicit replies, just stop
                break
            else:
                # Check if previous message in channel is from the same chain
                prev_msgs = [m async for m in channel.history(before=curr_msg, limit=1)]
                if prev_msgs:
                    prev = prev_msgs[0]
                    if prev.author == (discord_bot.user if channel.type == discord.ChannelType.private else curr_msg.author):
                        if prev.type in (discord.MessageType.default, discord.MessageType.reply):
                            parent = prev
        except (discord.NotFound, discord.HTTPException):
            break

        curr_msg = parent

    # Build the display
    max_seq_len = 4096  # Default for Talkie; could be read from model config
    system_prompt_enabled = bool(config.get("system_prompt"))
    system_tokens_est = 200 if system_prompt_enabled else 0

    # Rough token estimate: ~4 chars per token for English
    total_chars = sum(len(m["content"]) for m in messages)
    estimated_tokens = total_chars // 4 + system_tokens_est

    lines = []
    lines.append(f"**📌 Context for this channel**\n")
    lines.append(f"```\n"
                 f"Model:            {curr_model}\n"
                 f"Max messages:      {max_messages}\n"
                 f"Context window:    {max_seq_len} tokens\n"
                 f"System prompt:     {'✅ enabled' if system_prompt_enabled else '❌ disabled'} (~{system_tokens_est} tokens)\n"
                 f"Cache nodes:       {len(msg_nodes)} / {MAX_MESSAGE_NODES}\n"
                 f"```\n")
    lines.append(f"**Reply chain:** {chain_count} message{'s' if chain_count != 1 else ''} traced (estimated ~{estimated_tokens} tokens + ~{system_tokens_est} system = **~{estimated_tokens + system_tokens_est} of {max_seq_len}** tokens used)\n")

    # Token budget bar
    pct = min((estimated_tokens + system_tokens_est) / max_seq_len, 1.0)
    bar_len = 20
    filled = round(pct * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    color = "🟢" if pct < 0.5 else "🟡" if pct < 0.8 else "🔴"
    lines.append(f"{color} `{bar}` {pct:.0%}\n")

    # Show the message chain
    lines.append(f"**Messages in chain** (newest → oldest):\n")
    for i, msg in enumerate(messages):
        role_icon = "🤖" if msg["role"] == "assistant" else "👤"
        preview = msg["content"][:120].replace("\n", " ")
        if len(msg["content"]) > 120:
            preview += "…"
        lines.append(f"{role_icon} `{msg['role']:9s}` **{msg['author']}**: {preview}")

    if chain_count >= max_messages:
        lines.append(f"\n⚠️ Chain truncated at {max_messages} messages. Older messages are not included.")

    output = "\n".join(lines)

    # Discord has a 4096 char limit on embed descriptions
    if len(output) > 4000:
        output = output[:3990] + "\n…(truncated)"

    embed = discord.Embed(description=output, color=discord.Color.blurple())
    await interaction.response.send_message(embed=embed, ephemeral=True)


@context_command.autocomplete("action")
async def context_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    options = [
        ("show", "📌 Show context for this channel"),
        ("status", "📊 Bot status & server health"),
        ("clear", "🗑️ Clear message cache (fresh start)"),
        ("reopen", "📂 Re-open this conversation for 15 min (allow others to @-mention)"),
    ]
    return [Choice(name=label, value=value) for value, label in options if curr_str.lower() in value or curr_str.lower() in label.lower()]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()

    # Start the dynamic status updater (idempotent — only starts once)
    if not any(t.get_name() == "status_updater" for t in asyncio.all_tasks()):
        asyncio.create_task(_status_updater(), name="status_updater")
        logging.info("[status_updater] background task started")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    # ── Gating: when does the bot respond? ────────────────────────────
    if new_msg.author.bot:
        return

    # Always respond in DMs
    should_respond = is_dm

    # In guild channels: require @-mention OR reply to the bot
    bot_mentioned = discord_bot.user in new_msg.mentions
    is_reply_to_bot = bool(new_msg.reference and new_msg.reference.resolved and new_msg.reference.resolved.author == discord_bot.user)

    if not is_dm and (bot_mentioned or is_reply_to_bot):
        should_respond = True

    # Mutable context: if no @-mention, check if someone ran /context reopen
    if not should_respond and not is_dm:
        mc = t_mutable_context.get(new_msg.channel.id)
        if mc and (time.time() - mc["opened_at"]) < mc["ttl_seconds"]:
            # Only respond if this message @-mentions the bot (others still need to ping)
            if bot_mentioned:
                should_respond = True
        elif mc:
            # Expired — clean up
            del t_mutable_context[new_msg.channel.id]

    if not should_respond:
        return

    # ── End gating ────────────────────────────────────────────────────

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = []

                # --- Vision proxy: describe images instead of passing raw pixels ---
                image_attachments = [
                    (att, resp)
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                vision_proxy = config.get("vision_proxy")
                if image_attachments and vision_proxy and vision_proxy.get("enabled"):
                    vp_url = vision_proxy["base_url"].rstrip("/") + "/v1"
                    vp_model = vision_proxy["model"]
                    vp_descs = await asyncio.gather(*[
                        _describe_image_via_vision_proxy(
                            b64encode(resp.content).decode("utf-8"),
                            att.content_type,
                            vp_url,
                            vp_model,
                            httpx_client,
                        )
                        for att, resp in image_attachments
                    ])
                    photo_lines = []
                    for desc in vp_descs:
                        if desc:
                            photo_lines.append(f"[The correspondent sends a photograph depicting: {desc}]")
                        else:
                            photo_lines.append("[The correspondent sends a photograph, but the wireless is too faint to make it out.]")
                    if photo_lines:
                        curr_node.text = (curr_node.text + "\n" + "\n".join(photo_lines)) if curr_node.text else "\n".join(photo_lines)
                elif image_attachments:
                    # No vision proxy — pass through as base64 (only works if model supports vision)
                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in image_attachments
                    ]

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"Correspondent No. {curr_msg.author.id}: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    # ── Auto multi-user context merging (reply-graph stitching) ──────────
    # When Person B replies to Person A's message (not a bot message), the
    # chain walk goes: B → A → ... but misses the bot's RESPONSE to A,
    # because A's parent isn't the bot message (the bot replied TO A).
    #
    # We fix this by scanning for "orphan" bot replies: messages in the
    # channel where the bot replied to a user message that's IN our chain
    # but whose bot response is NOT. We then thread those branches in.
    #
    # This lets separate user→bot conversations merge automatically when
    # users interact with each other's messages, without forcing all
    # conversations into one shared pool.
    chain_msg_ids: set[int] = set()  # discord message IDs in our walked chain
    if not is_dm:
        # Reconstruct which message IDs we walked (via msg_nodes cache).
        # We do this by re-walking the parent chain from new_msg and
        # collecting every message ID we find in msg_nodes.
        _walk = new_msg
        _walked_ids = set()
        while _walk is not None and len(_walked_ids) < max_messages:
            _walked_ids.add(_walk.id)
            _n = msg_nodes.get(_walk.id)
            _walk = _n.parent_msg if _n else None
        chain_msg_ids = _walked_ids

        # Find bot replies that respond to messages IN our chain but
        # whose response is NOT in our chain. These are orphan branches.
        if len(messages) < max_messages:
            # Check recent channel messages for bot replies to our chain
            _merge_items = []  # (bot_msg_id, text, role) — ordered by time
            try:
                async for hist_msg in new_msg.channel.history(limit=50, before=new_msg):
                    if len(messages) + len(_merge_items) >= max_messages:
                        break
                    # Is this a bot reply to a message in our chain?
                    if hist_msg.author != discord_bot.user:
                        continue
                    if not hist_msg.reference or not hist_msg.reference.message_id:
                        continue
                    # The bot is replying to this message_id — is it in our chain?
                    replied_to_id = hist_msg.reference.message_id
                    if replied_to_id not in chain_msg_ids:
                        continue
                    # Is the bot's reply itself already in our chain?
                    if hist_msg.id in chain_msg_ids:
                        continue
                    # Found an orphan bot reply! Get its cached text.
                    bot_node = msg_nodes.get(hist_msg.id)
                    if bot_node and bot_node.text:
                        _merge_items.append((hist_msg, bot_node.text, "assistant"))
                    else:
                        # Not in cache — build from history content
                        _text = hist_msg.content[:max_text] if hist_msg.content else ""
                        if _text:
                            _merge_items.append((hist_msg, _text, "assistant"))
            except (discord.Forbidden, discord.HTTPException):
                logging.debug("[AUTO_MERGE] could not read channel history for orphan detection")

            # Also check for user messages that REPLY to a bot message in our chain
            # but whose bot-response is in a different branch
            try:
                async for hist_msg in new_msg.channel.history(limit=50, before=new_msg):
                    if len(messages) + len(_merge_items) >= max_messages:
                        break
                    if hist_msg.author == discord_bot.user:
                        continue
                    if hist_msg.id in chain_msg_ids:
                        continue
                    if not hist_msg.reference or not hist_msg.reference.message_id:
                        continue
                    replied_to_id = hist_msg.reference.message_id
                    # Is this user replying to a bot message that's in our chain?
                    if replied_to_id not in chain_msg_ids:
                        continue
                    # The replied-to message must be from the bot
                    replied_msg = hist_msg.reference.cached_message or await new_msg.channel.fetch_message(replied_to_id)
                    if replied_msg.author != discord_bot.user:
                        continue
                    # This user is joining the conversation by replying to the bot
                    _text = hist_msg.content[:max_text] if hist_msg.content else ""
                    _text = _text.removeprefix(discord_bot.user.mention).lstrip()
                    _text = f"Correspondent No. {hist_msg.author.id}: {_text}"
                    _merge_items.append((hist_msg, _text, "user"))
            except (discord.Forbidden, discord.HTTPException):
                logging.debug("[AUTO_MERGE] could not read channel history for cross-branch users")

            if _merge_items:
                # Dedup by message ID (our main chain scanned from the bottom up,
                # history scanned newest-first — so we reverse to get chronological)
                _seen = set(chain_msg_ids)
                _added = []
                for hist_msg, text, role in reversed(_merge_items):
                    if hist_msg.id in _seen:
                        continue
                    _seen.add(hist_msg.id)
                    _added.append(dict(content=text, role=role))

                if _added:
                    messages = _added + messages  # prepend chronologically
                    logging.info(
                        f"[AUTO_MERGE] stitched {len(_added)} orphan messages "
                        f"into context (total now {len(messages)})"
                    )

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")
    # OBSESSIVE: log the entire message chain that was built
    logging.debug(f"[MSG_CHAIN] {len(messages)} messages traced; reversed for API:")
    for i, msg in enumerate(messages[::-1]):
        content_preview = repr(msg.get('content', '')[:200] if isinstance(msg.get('content'), str) else str(msg.get('content'))[:200])
        logging.debug(f"  [MSG {i}] role={msg.get('role')} content={content_preview}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) — NON-STREAMING for simplicity.
    # The server already generates fully before returning; streaming just
    # added complexity (lag bugs, duplication bugs, chunk reassembly).
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=False, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    logging.info(f"[API_CALL] model={model} stream=False messages_count={len(messages[::-1])}")
    logging.debug(f"[API_KWARGS] {json.dumps(openai_kwargs, default=str)[:3000]}")

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 4000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    if not use_plain_responses:
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            response = await openai_client.chat.completions.create(**openai_kwargs)

        raw_text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        logging.info(f"[GENERATE] finish={finish_reason} raw_len={len(raw_text)}")
        logging.debug(f"[GENERATE] raw_text[:500]={repr(raw_text[:500])}")

        # ── Record token output for status display ──────────────────────
        # Use usage.completion_tokens if available, else estimate ~4 chars/token
        if hasattr(response, 'usage') and response.usage and response.usage.completion_tokens:
            est_tokens = response.usage.completion_tokens
        else:
            est_tokens = max(1, len(raw_text) // 4)
        _output_log.append((time.time(), est_tokens))

        translated = era_to_discord(raw_text)

        # Split into Discord-safe segments
        while translated:
            segment = translated[:max_message_length]
            response_contents.append(segment)
            translated = translated[max_message_length:]

        # Send segments
        for i, content in enumerate(response_contents):
            is_last = (i == len(response_contents) - 1)
            if use_plain_responses:
                await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))
            else:
                embed.description = content
                embed.color = EMBED_COLOR_COMPLETE if is_last else EMBED_COLOR_INCOMPLETE
                await reply_helper(embed=embed, silent=True if i > 0 else False)

    except Exception:
        logging.exception("Error while generating response")
        # Attempt to send a failure notice so the user isn't left hanging
        try:
            await new_msg.reply("⚠️ I'm afraid the telegraph lines are tangled. Please try again.", silent=True)
        except Exception:
            pass
    else:
        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = "".join(response_contents)
            msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
