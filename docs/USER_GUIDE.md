# TalkieLM User Guide

Everything you need to know about chatting with the Talkie 1930 bot on Discord.

## Quick Reference

| Action | How |
|--------|-----|
| Start a conversation | @mention the bot or send a DM |
| Continue a conversation | Reply to the bot's message |
| See current context | `/context show` |
| Open group conversation | `/context reopen` |
| Clear the bot's memory | `/context clear` |
| Check bot/server status | `/context status` |
| Switch model (admin) | `/model` |

---

## Starting a Conversation

### In a Server Channel

@mention the bot to get its attention:

> **You:** @Talkie 1930 Good morning! What do you make of these horseless carriages?

The bot will respond in character as an Edwardian conversationalist. Its response appears as a Discord embed that updates live as the bot "types."

### In Direct Messages

If the bot is configured to allow DMs (`allow_dms: true`), you can simply send a message — no @mention needed.

### By Replying

You can also **reply to any of the bot's previous messages** without an @mention. The bot will see the reply and continue the conversation from that point.

> **You:** *(replies to bot's message)* And what about aeroplanes? Have you ever seen one?

---

## Continuing a Conversation

TalkieLM builds context from **reply chains**. To continue a conversation:

1. **Reply to the bot's message** — this is the most reliable way.
2. The bot walks backwards through the reply chain, collecting up to 10 messages.
3. The entire chain (plus the system prompt) is sent to the model as context.

### What "Reply Chain" Means

```
Message A (you @mention bot)
 └── Message B (bot replies to A)
      └── Message C (you reply to B)
           └── Message D (bot replies to C)
```

Messages A through D form a single reply chain. When you send message C, the bot sees messages A + B as context. When the bot generates D, it sees A + B + C.

### Multi-User Conversations

Multiple people can participate in the same reply chain. If Alice @mentions the bot, Bob replies to Alice, and the bot replies to Bob — all messages are in the same chain. The bot treats everyone in the chain as a single conversation.

To make multi-user conversations easier without requiring everyone to reply-chain, use `/context reopen` (see below).

---

## Slash Commands

### `/context` — Manage Conversation Context

The `/context` command is your Swiss army knife for managing what the bot remembers and knows.

#### `/context show`

Shows the current context for the channel — the reply chain that will be sent to the model on the next message.

- Displays the message chain (newest → oldest)
- Shows an estimated token usage bar
- Lists each message's role (user/assistant), author, and a preview

#### `/context status`

Shows the bot's operational status:

- Current model name
- Max messages and context window settings
- Whether the system prompt is enabled
- Message cache usage
- Whether the inference server is online

Use this to check if the bot is healthy before starting a conversation.

#### `/context clear`

Clears the bot's message cache. The next time someone talks to the bot, it will start with fresh context — as if meeting for the first time.

**When to use this:**
- The bot is confused by old context
- You want to start a completely new topic
- The token budget bar is in the red zone (>80%)

The cache rebuilds automatically as new messages are processed.

#### `/context reopen`

Opens a **15-minute mutable context window** for the current channel. During this window:

- Anyone who @mentions the bot in the channel is included in the context
- You don't need to be in a reply chain — the bot will see all messages that @mention it
- After 15 minutes, the window expires and the bot returns to reply-chain-only context

**When to use this:**
- Group conversations where multiple people want to participate
- When reply chains get too tangled
- For casual "lounge" style conversations

Use `/context clear` to reset manually, or wait for the 15-minute timer to expire.

### `/model` — Switch the Active Model

Allows admins to switch the model the bot is using. Provides an autocomplete dropdown of available models configured in `config.yaml`.

Non-admin users can view the current model but cannot switch it.

---

## Understanding the Bot's Behavior

### The Edwardian Persona

The bot speaks as an educated person from the year 1929. Key characteristics:

- **No post-1930 knowledge** — It has never heard of the Second World War, computers, the internet, television, space travel, or any event after ~1930.
- **British spelling conventions** — "colour", "aeroplane", "marvellous", etc.
- **Elaborate prose** — It does not give terse or monosyllabic answers. It elaborates.
- **Warm, companionable tone** — It engages with curiosity and good humour.
- **"Discord" as wireless telegraph** — The system prompt explains Discord as "a sort of wireless telegraph" so the model can interact naturally without breaking character.

### User Identifiers

Instead of seeing your Discord username, the bot sees you as **"Correspondent No. 123456789"** — where the number is your Discord user ID. This is because:

1. The Talkie tokenizer doesn't know Discord's `<@123456789>` mention format (those tokens are out-of-vocabulary).
2. Converting to prose ("Correspondent No. 123456789") gives the model tokens it can actually process.
3. The bot converts back to Discord pings in its output so you see proper @mentions.

You may see the bot address you as "Correspondent No. 123456789" in its replies — this is expected and intentional.

### Streaming Responses

The bot streams its responses live in Discord. You'll see:

1. The bot starts "typing..." (Discord typing indicator)
2. A message appears with a ⚪ indicator — the response is still being generated
3. The message updates every ~1 second with new text
4. When complete, ⚪ disappears and the embed color changes from orange 🟠 to green 🟢

If the response is very long (>4096 characters), the bot will split it across multiple messages.

### Context Limits

The bot remembers a limited amount of conversation:

| Limit | Value | What Happens When Exceeded |
|-------|-------|---------------------------|
| Messages | 10 | Oldest messages are silently dropped |
| Text per message | 3,000 characters | Message is truncated |
| Context window | 4,096 tokens | Responses become garbled — use `/context clear` |
| Images | 0 | Talkie is text-only; images are ignored |

**Tip:** Use `/context show` to see your current token usage. If the bar is orange or red, the bot is running low on context space.

---

## Tips for Great Conversations

1. **Give the bot something to work with.** Short one-word messages get short replies. Rich, descriptive questions get elaborate Edwardian prose.

2. **Stay in character.** The bot responds best when you engage with its 1920s worldview. Ask about technology, travel, politics, or culture of the era.

3. **Use reply chains.** The bot builds context from reply chains. Always reply to the bot's last message to maintain a coherent conversation.

4. **Don't expect post-1930 knowledge.** The model genuinely doesn't know about modern events. It won't break character spontaneously, but it may hallucinate if you force modern topics.

5. **Start fresh for new topics.** Use `/context clear` when switching to an unrelated topic. Old context about a different subject can confuse the model.

6. **Keep chains under 10 messages.** If conversations get very long, the oldest messages are dropped. For deep conversations, start a new thread.
