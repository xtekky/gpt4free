"""
Discord bot powered by g4f (gpt4free).

Features:
- /ask command for one-shot questions
- /chat command for conversation with per-user history
- /clear to reset conversation history
- /model to show the configured model
- /tools to list, enable, and disable MCP tools
- MCP tool-calling loop: the model can call tools (web search, scraping,
  image generation, etc.) and the bot executes them via g4f's MCPServer,
  feeds the results back, and produces a final answer.
- Streaming responses edited in-place for a "typing" effect
- Configurable model and provider via environment variables
"""

from __future__ import annotations

import os
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

import g4f.Provider
from g4f.client import ClientFactory

from mcp_tools import MCPToolManager, ALL_AVAILABLE_TOOLS, SAFE_DEFAULT_TOOLS

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOKEN = os.getenv("DISCORD_TOKEN")
MODEL = os.getenv("G4F_MODEL", "auto")
SYSTEM_PROMPT = os.getenv(
    "G4F_SYSTEM_PROMPT",
    "You are a helpful, friendly Discord assistant. Keep answers concise "
    "and formatted with Discord markdown when useful. "
    "When a question needs fresh information, use the web_search tool. "
    "When asked to generate an image, use the image_generation tool.",
)
MAX_HISTORY = int(os.getenv("G4F_MAX_HISTORY", "12"))  # messages per user
PROXY = os.getenv("G4F_PROXY")  # optional, e.g. "socks5://127.0.0.1:1080"
MAX_TOOL_LOOPS = int(os.getenv("G4F_MAX_TOOL_LOOPS", "4"))  # safety cap

# Simple validation for runtime model names.
ALLOWED_MODEL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)

# Comma-separated list of tools to enable at startup (default: safe set).
_enabled_env = os.getenv("G4F_ENABLED_TOOLS", "")
ENABLED_TOOLS = (
    {t.strip() for t in _enabled_env.split(",") if t.strip()}
    if _enabled_env
    else set(SAFE_DEFAULT_TOOLS)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("g4f-discord")

# ---------------------------------------------------------------------------
# g4f async client + MCP tool manager (shared across requests)
# ---------------------------------------------------------------------------
client = ClientFactory.create_async_client(provider="default",
                                           api_key=os.getenv("G4F_API_KEY"),
                                           media_provider=getattr(g4f.Provider, os.getenv("G4F_MEDIA_PROVIDER", "PollinationsImage")))
mcp = MCPToolManager(enabled_tools=ENABLED_TOOLS)

# Per-user conversation history: user_id -> deque of {"role", "content"}
histories: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True  # required to read user messages

bot = commands.Bot(command_prefix="!", intents=intents)


# ---------------------------------------------------------------------------
# Auto image generation from channel messages
# ---------------------------------------------------------------------------
# If enabled, the bot listens to messages in the given channel(s) and
# runs image generation.
#
# Env vars:
# - G4F_IMAGE_CHANNELS: comma-separated channel ids (required to enable)
_image_channels_env = os.getenv("G4F_IMAGE_CHANNELS", "")
IMAGE_CHANNELS = {
    int(x.strip())
    for x in _image_channels_env.split(",")
    if x.strip().isdigit()
}


def _build_messages(user_id: int, user_content: str) -> List[dict]:
    """Return the full message list including system prompt and history."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(histories[user_id])
    messages.append({"role": "user", "content": user_content})
    return messages


def _truncate(text: str, limit: int = 1900) -> str:
    """Truncate text to stay within Discord's 2000-char message limit."""
    return text if len(text) <= limit else text[:limit] + "…"


def _normalize_model_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Model name cannot be empty")
    if any(ch not in ALLOWED_MODEL_CHARS for ch in name):
        raise ValueError("Model name contains invalid characters")
    return name


# ---------------------------------------------------------------------------
# Completion helpers
# ---------------------------------------------------------------------------
async def _stream_response(
    interaction: discord.Interaction, messages: List[dict]
) -> str:
    """
    Stream a g4f completion (no tools) and edit the interaction response
    in-place. Returns the full accumulated text.
    """
    accumulated = ""
    last_sent = ""
    update_threshold = 80  # characters before each edit

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        proxy=PROXY,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated += chunk.choices[0].delta.content
            if len(accumulated) - len(last_sent) >= update_threshold:
                last_sent = accumulated
                try:
                    await interaction.edit_original_response(
                        content=_truncate(accumulated) + " ▌"
                    )
                except discord.HTTPException:
                    pass

    return accumulated


async def _completion_with_tools(
    messages: List[dict],
    use_tools: bool,
) -> tuple[str, Optional[list]]:
    """
    Non-streaming completion that may return tool_calls.

    Returns (content, tool_calls) where tool_calls is None when no tools
    were requested or none were called.
    """
    kwargs: dict = {"model": MODEL, "messages": messages, "stream": False, "proxy": PROXY}
    if use_tools and mcp.definitions:
        kwargs["tools"] = mcp.definitions
        kwargs["tool_choice"] = "auto"

    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""
    tool_calls = getattr(choice.message, "tool_calls", None)
    return content, tool_calls


async def _send_streamed_text(interaction: discord.Interaction, text: str) -> None:
    """Send a block of text, editing in chunks for a typing effect."""
    if not text.strip():
        return
    for i in range(0, len(text), 120):
        sent = text[: i + 120]
        try:
            await interaction.edit_original_response(content=_truncate(sent) + " ▌")
        except discord.HTTPException:
            pass


async def _run_tool_loop(
    interaction: discord.Interaction,
    messages: List[dict],
    use_tools: bool,
) -> tuple[str, List[dict]]:
    """
    Run the full tool-calling loop.

    1. Ask the model for a completion (with tools available).
    2. If it returns tool_calls, execute them via the MCP server.
    3. Append the assistant message + tool results to the conversation.
    4. Repeat until the model stops calling tools or MAX_TOOL_LOOPS is hit.
    5. Stream the final answer into the interaction response.

    Returns (final_text, tool_result_messages).
    """
    tool_results_log: List[dict] = []

    if not (use_tools and mcp.definitions):
        # No tools — just stream a normal response.
        final = await _stream_response(interaction, messages)
        return final, tool_results_log

    # --- Tool-calling phase (non-streaming) ---
    working_messages = list(messages)

    for loop in range(MAX_TOOL_LOOPS):
        content, tool_calls = await _completion_with_tools(working_messages, use_tools=True)

        if not tool_calls:
            # No (more) tool calls — stream this final content to the user.
            if content.strip():
                await _send_streamed_text(interaction, content)
            return content, tool_results_log

        # Show the user that tools are running.
        names = ", ".join(tc.function.name for tc in tool_calls)
        try:
            await interaction.edit_original_response(
                content=f"🔧 Running tools: {names}…"
            )
        except discord.HTTPException:
            pass

        # Execute the tool calls via the MCP server.
        tool_results = await mcp.execute_tool_calls(tool_calls)
        tool_results_log.extend(tool_results)

        # Append the assistant's tool-call message + tool results to the
        # conversation so the model can see the outcomes.
        working_messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": getattr(tc, "id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            }
        )
        working_messages.extend(tool_results)

    # Exhausted the loop cap — do one final streaming pass without tools
    # so the model summarises what it learned.
    log.warning("Tool loop cap (%d) reached; generating final summary", MAX_TOOL_LOOPS)
    final = await _stream_response(interaction, working_messages)
    return final, tool_results_log


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------
@bot.tree.command(name="ask", description="Ask a one-shot question (no history).")
@app_commands.describe(
    question="Your question for the AI",
    tools="Allow the AI to use MCP tools (web search, etc.)",
)
async def ask(interaction: discord.Interaction, question: str, tools: bool = True):
    await interaction.response.defer(thinking=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        reply, tool_results = await _run_tool_loop(interaction, messages, use_tools=tools)
    except Exception as e:
        log.exception("g4f request failed")
        await interaction.followup.send(f"⚠️ Error: {e}")
        return

    await _finalize_response(interaction, reply, tool_results)


@bot.tree.command(name="chat", description="Chat with conversation history.")
@app_commands.describe(
    message="Your message to the AI",
    tools="Allow the AI to use MCP tools (web search, etc.)",
)
async def chat(interaction: discord.Interaction, message: str, tools: bool = True):
    await interaction.response.defer(thinking=True)
    user_id = interaction.user.id
    messages = _build_messages(user_id, message)
    try:
        reply, tool_results = await _run_tool_loop(interaction, messages, use_tools=tools)
    except Exception as e:
        log.exception("g4f request failed")
        await interaction.followup.send(f"⚠️ Error: {e}")
        return

    await _finalize_response(interaction, reply, tool_results)

    # Store turn in history
    histories[user_id].append({"role": "user", "content": message})
    histories[user_id].append({"role": "assistant", "content": reply})


@bot.tree.command(name="clear", description="Clear your conversation history.")
async def clear(interaction: discord.Interaction):
    histories.pop(interaction.user.id, None)
    await interaction.response.send_message(
        "🧹 Your conversation history has been cleared.", ephemeral=True
    )


@bot.tree.command(name="model", description="Show the currently configured model.")
async def model(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Current model: `{MODEL}`", ephemeral=True
    )


@bot.tree.command(name="setmodel", description="Set the currently used model.")
@app_commands.describe(
    value="Model name/id",
)
async def setmodel(interaction: discord.Interaction, value: str):
    global MODEL
    try:
        MODEL = _normalize_model_name(value)
    except ValueError as e:
        await interaction.response.send_message(f"❌ {e}", ephemeral=True)
        return

    await interaction.response.send_message(
        f"✅ Model set to: `{MODEL}`", ephemeral=True
    )


@bot.tree.command(name="models", description="List available models (best effort).")
async def models(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)

    available = None
    try:
        available = client.models.get_all()
    except Exception as e:
        log.exception(e)

    if not available:
        await interaction.followup.send(
            "⚠️ Unable to fetch a model catalog from the current provider. "
            "Use `/setmodel` with a model id that works for your provider.",
            ephemeral=True,
        )
        return
    if isinstance(available, dict):
        lines = [f"• `{n.get('label', k)}` ({n.get('requests', 0)})" for k, n in available.items() if not "requests" in n or n.get("requests") >= 5]
    else:
        lines = [f"• `{n}`" for n in available]
    msg = "**Available models:**\n" + "\n".join(lines)
    await interaction.followup.send(_truncate(msg, 1900), ephemeral=True)


@bot.tree.command(name="image", description="Generate an image from a prompt.")
@app_commands.describe(
    prompt="What to generate",
    model="Optional image model (defaults to current /model)",
)
async def image(
    interaction: discord.Interaction,
    prompt: str,
    model: Optional[str] = None,
):
    await interaction.response.defer(thinking=True)
    try:
        m = _normalize_model_name(model) if model else MODEL
        generated = await _generate_image(prompt=prompt, model=m)
    except Exception as e:
        log.exception("Image generation failed")
        await interaction.followup.send(f"⚠️ Image generation failed: {e}")
        return

    if generated.startswith("http://") or generated.startswith("https://"):
        embed = discord.Embed(title="🖼️ Generated image")
        embed.set_image(url=generated)
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(
            "🖼️ Generated image (non-URL result):\n"
            f"```{_truncate(generated, 1800)}```"
        )


@bot.tree.command(name="tools", description="List, enable, or disable MCP tools.")
@app_commands.describe(
    action="What to do (default: list)",
    name="Tool name (for enable/disable)",
)
@app_commands.choices(
    action=[
        app_commands.Choice(name="list", value="list"),
        app_commands.Choice(name="enable", value="enable"),
        app_commands.Choice(name="disable", value="disable"),
    ]
)
async def tools(
    interaction: discord.Interaction,
    action: Optional[app_commands.Choice[str]] = None,
    name: Optional[str] = None,
):
    value = action.value if action is not None else "list"

    if value == "list":
        enabled = mcp.enabled_names
        lines = ["**Enabled MCP tools:**"]
        if enabled:
            lines.extend(f"• `{t}`" for t in enabled)
        else:
            lines.append("_(none)_")
        lines.append("\n**Available (not enabled):**")
        disabled = sorted(ALL_AVAILABLE_TOOLS - set(enabled))
        if disabled:
            lines.extend(f"• `{t}`" for t in disabled)
        else:
            lines.append("_(all enabled)_")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    elif value == "enable":
        if not name:
            await interaction.response.send_message(
                "Provide a tool `name` to enable.", ephemeral=True
            )
            return
        if mcp.enable(name):
            await interaction.response.send_message(
                f"✅ Enabled tool `{name}`.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                f"❌ Tool `{name}` not found. Use `/tools list` to see options.",
                ephemeral=True,
            )

    elif value == "disable":
        if not name:
            await interaction.response.send_message(
                "Provide a tool `name` to disable.", ephemeral=True
            )
            return
        if mcp.disable(name):
            await interaction.response.send_message(
                f"✅ Disabled tool `{name}`.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                f"❌ Tool `{name}` was not enabled.", ephemeral=True
            )


@bot.event
async def on_message(message: discord.Message):
    # Let slash commands etc. work as usual.
    await bot.process_commands(message)

    if not IMAGE_CHANNELS:
        return

    if message.author.bot:
        return

    if message.channel.id not in IMAGE_CHANNELS:
        return

    if not message.content:
        return

    prompt = message.content.strip()
    if not prompt:
        await message.channel.send("🖼️ Usage: !img <prompt>")
        return

    try:
        # Create a placeholder then edit/send final result.
        placeholder = await message.channel.send("🖼️ Generating…")
        m = MODEL
        generated = await _generate_image(prompt=prompt, model=m)

        if generated.startswith("http://") or generated.startswith("https://"):
            embed = discord.Embed(title="🖼️ Generated image")
            embed.set_image(url=generated)
            await placeholder.edit(content="", embed=embed)
        else:
            await placeholder.edit(
                content="🖼️ Generated image (non-URL result):\n"
                + _truncate(generated, 1800)
            )
    except Exception as e:
        log.exception("Auto image generation failed")
        await message.channel.send(f"⚠️ Image generation failed: {e}")


# ---------------------------------------------------------------------------
# Response finalisation
# ---------------------------------------------------------------------------
async def _finalize_response(
    interaction: discord.Interaction,
    reply: str,
    tool_results: List[dict],
) -> None:
    """Edit the interaction response with the final reply + tool summary."""
    if not reply.strip():
        if tool_results:
            tool_text = MCPToolManager.format_tool_results_for_discord(tool_results)
            await interaction.edit_original_response(content=_truncate(tool_text))
        else:
            await interaction.edit_original_response(
                content="⚠️ The model returned an empty response."
            )
        return

    # If tools were used, append a collapsible summary.
    if tool_results:
        tool_text = MCPToolManager.format_tool_results_for_discord(tool_results)
        combined = f"{_truncate(reply)}\n\n{tool_text}"
        await interaction.edit_original_response(content=_truncate(combined, 1900))
    else:
        await interaction.edit_original_response(content=_truncate(reply))


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
async def _generate_image(prompt: str, model: str) -> str:
    """Generate an image via g4f and return a URL or base64-like string."""
    result = await client.images.generate(
        prompt=prompt,
        model=model,
        proxy=PROXY,
        response_format="url",
    )
    return str(result.data[0].url)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
    log.info("MCP tools enabled: %s", mcp.enabled_names)
    try:
        synced = await bot.tree.sync()
        log.info("Synced %d slash commands", len(synced))
    except Exception:
        log.exception("Failed to sync slash commands")


def main():
    if not TOKEN:
        raise SystemExit(
            "DISCORD_TOKEN not set. Put it in .env or export it as an env var."
        )
    bot.run(TOKEN)


if __name__ == "__main__":
    main()
