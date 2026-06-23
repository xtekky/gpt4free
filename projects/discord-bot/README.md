# g4f Discord Bot

A Discord bot powered by [gpt4free (g4f)](https://github.com/xtekky/gpt4free) that lets your server chat with AI models — **no API keys required**.

## Features

- 🤖 **`/ask`** — one-shot questions (no history kept)
- 💬 **`/chat`** — conversational mode with per-user message history
- 🧹 **`/clear`** — reset your conversation history
- 🏷️ **`/model`** — show the currently configured model
- 🔧 **`/tools`** — list, enable, and disable MCP tools
- 🛠️ **MCP tool-calling** — the AI can autonomously call tools (web search, web scraping, image generation, text-to-audio, and more) via g4f's built-in MCP server. The bot executes the tool, feeds the result back, and loops until the AI has a final answer.
- ⚡ **Streaming responses** — edits the message in-place for a live "typing" effect
- 🔒 Per-user history isolation with configurable length

## Setup

### 1. Create a Discord application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create a new application → **Bot** tab → **Reset Token** to get your token.
3. Enable **Message Content Intent** under *Privileged Gateway Intents*.
4. Invite the bot to your server using the **OAuth2 → URL Generator** (scopes: `bot`, `applications.commands`; permissions: `Send Messages`, `Read Message History`).

### 2. Configure environment

```bash
cd projects/discord-bot
cp .env.example .env
# Edit .env and paste your DISCORD_TOKEN
```

### 3. Install dependencies

The bot needs `discord.py` and `python-dotenv` on top of g4f's requirements:

```bash
pip install discord.py python-dotenv
```

### 4. Run

```bash
python bot.py
```

You should see:

```
[INFO] Logged in as YourBot#1234 (id=...)
[INFO] Synced 4 slash commands
```

## Configuration

All settings live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DISCORD_TOKEN` | *(required)* | Your Discord bot token |
| `G4F_MODEL` | `gpt-4o-mini` | Model name passed to g4f |
| `G4F_SYSTEM_PROMPT` | *(see .env.example)* | System prompt for the assistant |
| `G4F_MAX_HISTORY` | `12` | Max messages stored per user |
| `G4F_PROXY` | *(none)* | Optional proxy for g4f requests |
| `G4F_ENABLED_TOOLS` | *(safe set)* | Comma-separated MCP tools to enable at startup |
| `G4F_MAX_TOOL_LOOPS` | `4` | Max tool-calling rounds before forcing a final answer |

## MCP tools

The bot integrates g4f's built-in [MCP server](../../g4f/mcp/) so the AI can call tools autonomously during a conversation. The flow:

1. You ask a question (e.g. *"What's the latest news on X?"*).
2. The model decides to call `web_search` and returns a tool call.
3. The bot executes the tool via `MCPServer`, appends the result to the conversation, and asks the model again.
4. The loop repeats until the model produces a final answer (or `G4F_MAX_TOOL_LOOPS` is hit).

### Available tools

| Tool | Description | Enabled by default? |
|---|---|---|
| `web_search` | Search the web via DuckDuckGo | ✅ |
| `web_scrape` | Extract text content from a URL | ✅ |
| `mark_it_down` | Convert a URL to markdown | ✅ |
| `text_to_audio` | Generate an audio URL from text | ✅ |
| `image_generation` | Generate an image from a prompt | ✅ |
| `python_execute` | Run Python in a sandboxed environment | ❌ |
| `apply_patch` | Apply a unified diff patch | ❌ |
| `file_read` | Read a file from `~/.g4f/workspace` | ❌ |
| `file_read_lines` | Read a line range from a workspace file | ❌ |
| `file_search` | Search files in the workspace | ❌ |
| `file_write` | Write a file to the workspace | ❌ |
| `file_list` | List workspace files | ❌ |
| `file_delete` | Delete a workspace file | ❌ |

File/Python/patch tools are **disabled by default** because they operate on the bot's local filesystem. Enable them only if you trust your Discord users.

### Managing tools at runtime

Use the `/tools` slash command:

```
/tools                    # list enabled and available tools
/tools action:enable name:python_execute
/tools action:disable name:image_generation
```

You can also set the startup set via `G4F_ENABLED_TOOLS` in `.env`:

```
G4F_ENABLED_TOOLS=web_search,web_scrape,image_generation
```

### Disabling tools per request

Both `/ask` and `/chat` accept an optional `tools` boolean (defaults to `true`):

```
/ask question:"What is 2+2?" tools:False
```

## Changing the provider

`bot.py` imports `OpenaiChat` as the default provider. To use a different one, edit the import and the `AsyncClient` constructor:

```python
from g4f.Provider import Gemini, OpenaiChat, BingCreateImages

client = AsyncClient(provider=Gemini)
```

See all available providers with:

```bash
g4f --help
```

## Project structure

```
projects/discord-bot/
├── bot.py          # Main bot logic (commands, tool-calling loop)
├── mcp_tools.py    # MCP tool manager (definitions, execution, display)
├── .env.example    # Template environment file
└── README.md       # This file
```

## Notes

- g4f relies on free third-party providers; availability and quality vary. If a request fails, try a different model or provider.
- The bot uses `AsyncClient` so it stays responsive while streaming.
- Discord limits messages to 2000 characters; long replies are truncated.
