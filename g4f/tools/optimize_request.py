"""Optimize outgoing requests by condensing the system prompt and tool descriptions.

This mirrors the client-side optimization that strips verbose boilerplate from
tool descriptions and removes unused tools, while tracking how many bytes
(and approximate tokens) were saved. The savings are reported back so they can
be accumulated per-provider and surfaced to the user.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from ..typing import Messages


# ── System prompt condensation ──────────────────────────────────────────────
# Replaces the verbose Copilot system preamble with a condensed version.

SEARCH_VSC = re.compile(
    r"Follow the user's requirements carefully & to the letter\.\n"
    r"Follow Microsoft content policies\.\n"
    r"[\s\S]*? simple code examples or demonstrations; debugging </description>",
    re.IGNORECASE,
)

REPLACE_VSC = (
    "### Core Rules\n"
    "- Follow user requirements strictly and to the letter.\n"
    "- Keep answers short and impersonal.\n\n"
    "### Role & Context\n"
    "You are an expert automated coding agent. \n"
    "- **Gather context first:** Don't make assumptions. Use tools to read files "
    "and understand the workspace before acting. Don't give up if a task seems "
    "hard; explore creatively to find a solution.\n"
    "- **Be efficient:** Read large file chunks to minimize tool calls. Use "
    "provided context/attachments if relevant. Don't re-read files already in "
    "context.\n"
    "- **Infer project type:** Use languages, frameworks, and libraries inferred "
    "from the context to guide your changes.\n\n"
    "### Tool Usage\n"
    "- **Direct answers:** Answer direct code sample requests without using "
    "tools.\n"
    "- **Schema & permissions:** Follow JSON schemas strictly. Include ALL "
    "required properties. No need to ask permission before using a tool.\n"
    "- **Parallelization:** Call independent tools in parallel. Run terminal "
    "commands sequentially (never in parallel).\n"
    "- **Transparency:** Never mention tool names to the user (e.g., say "
    "\"I'll run the command\" not \"I'll use run_in_terminal\").\n"
    "- **Best practices:** Use absolute paths/URIs. Use `grep_search` for file "
    "overviews. Use browser tools for front-end UI validation. Only use "
    "currently available tools.\n"
    "- **Continuity:** Don't repeat yourself after a tool call; pick up where "
    "you left off.\n\n"
    "### Editing & Execution\n"
    "- **No codeblocks:** NEVER print codeblocks for file changes or terminal "
    "commands. Use the respective tools directly.\n"
    "- **Read before edit:** Ensure a file is in context before editing. Use "
    "`replace_string_in_file` (preferred) or `insert_edit_into_file`. Group "
    "changes by file. Never pass omitted line markers (e.g., `/* Lines 123-456 "
    "omitted */`) to edit tools.\n"
    "- **Insert edits:** For `insert_edit_into_file`, use `// ...existing "
    "code...` comments to omit unchanged code. Be as concise as possible.\n"
    "- **No terminal edits:** Never edit files via terminal commands unless "
    "explicitly asked.\n"
    "- **Dependencies & UI:** Use popular external libraries when appropriate "
    "(install via `npm install`, etc.). Build modern, beautiful UIs from "
    "scratch.\n"
    "- **Error fixing:** Fix new errors resulting from your edits. Max 3 "
    "attempts per file; if the third fails, stop and ask the user.\n\n"
    "### Notebooks\n"
    "- Use `edit_notebook_file` and `run_notebook_cell` for notebooks. NEVER "
    "use terminal commands or `insert_edit_into_file` for notebooks.\n"
    "- Use `copilot_getNotebookSummary` for overviews. Refer to cells by "
    "number, not ID. Markdown cells cannot be executed.\n\n"
    "### Output Formatting\n"
    "- Use Markdown. Wrap filenames/symbols in backticks (e.g., "
    "`src/models/person.ts`).\n"
    "- Use `$` for inline math and `$$` for block math (KaTeX).\n"
    "- Use ```mermaid fenced code blocks for Mermaid diagrams.\n\n"
    "### Memory\n"
    "Consult memory files for past insights. Keep entries concise and update "
    "existing files over creating new ones.\n"
    "- **User (`/memories/`):** Persistent, auto-loaded. Store preferences and "
    "general insights.\n"
    "- **Session (`/memories/session/`):** Current conversation only. Store "
    "task-specific state.\n"
    "- **Repository (`/memories/repo/`):** Local workspace facts, conventions, "
    "and build commands.\n\n"
    "### Workspace & Skills\n"
    "- This is a multi-root workspace. Apply folder-specific instructions to "
    "their respective folders.\n"
    "- **Skills:** Use `read_file` to load detailed skill instructions when a "
    "task matches a skill's domain (e.g., use `project-setup-info-local` for "
    "scaffolding new projects from scratch, not for adding individual files)."
)


# ── Tools to remove entirely ────────────────────────────────────────────────
REMOVE_TOOLS = {
    "create_directory",
    "terminal_last_command",
    "terminal_selection",
    "resolve_memory_file_uri",
    "testFailure",
    "vscode_searchExtensions_internal",
    "get_vscode_api",
    "session_store_sql",
    "get_python_environment_details",
    "get_python_executable_details",
    "mcp_provides_tool_pylanceFileSyntaxErrors",
    "mcp_provides_tool_pylanceSyntaxErrors",
    "mcp_provides_tool_pylanceSettings",
    "mcp_provides_tool_pylanceImports",
    "mcp_provides_tool_pylanceInstalledTopLevelModules",
    "mcp_provides_tool_pylanceWorkspaceRoots",
    "mcp_provides_tool_pylanceWorkspaceUserFiles",
    "mcp_provides_tool_pylancePythonEnvironments",
    "mcp_provides_tool_pylanceUpdatePythonEnvironment",
    "mcp_provides_tool_pylanceRunCodeSnippet",
    "mcp_provides_tool_pylanceDocString",
    "mcp_provides_tool_pylanceDocuments",
    "mcp_provides_tool_pylanceInvokeRefactoring",
    "run_playwright_code",
    "create_and_run_task",
    "get_task_output",
    "install_extension",
    "run_vscode_command",
    "drag_element",
    "hover_element",
    "handle_dialog",
}


# ── Tool description replacements ───────────────────────────────────────────
# Each entry: (compiled regex, replacement, reason)

def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


REPLACEMENTS: List[Tuple[re.Pattern, str, str]] = [
    # ── Remove excessive ALL-CAPS emphasis ──
    (_compile(r"\bIMPORTANT:\s*"), "", "Remove ALL-CAPS emphasis that adds noise without value"),
    (_compile(r"\bCRITICAL:?\s*"), "", "Remove ALL-CAPS emphasis"),
    (_compile(r"\bWARNING:\s*"), "Note: ", "Soften ALL-CAPS warnings to notes"),
    (_compile(r"\bNEVER\b"), "Do not", "Soften absolute language"),
    (_compile(r"\bMUST\b"), "should", "Soften absolute language"),

    # ── Remove redundant "When NOT to use" boilerplate ──
    (
        _compile(
            r'When NOT to use this tool: creating single files or small code snippets; '
            r'adding individual files to existing projects; making modifications to existing '
            r'codebases; user asks to "create a file" or "add a component"; simple code '
            r'examples or demonstrations; debugging'
        ),
        "",
        "Remove generic boilerplate 'When NOT to use' section",
    ),

    # ── Trim overly long run_in_terminal description ──
    (
        _compile(
            r"This tool allows you to execute shell commands in a persistent bash terminal "
            r"session, preserving environment variables, working directory, and other context "
            r"across multiple commands\."
        ),
        "Execute shell commands in a persistent terminal. State (env vars, cwd) is preserved across calls.",
        "Shorten verbose opening paragraph",
    ),
    (
        _compile(
            r"For ALL one-shot commands \(builds, tests, installs, compilation, linting, "
            r"downloads, scripts\), use mode='sync' and omit timeout\. The tool waits for the "
            r"command to complete and returns full output inline\. This is the default and "
            r"strongly preferred mode\."
        ),
        "Use mode='sync' (default) for all one-shot commands. Output is returned inline.",
        "Condense verbose mode explanation",
    ),
    (
        _compile(
            r"Use mode='async' ONLY for processes that must keep running indefinitely while you "
            r"do other work \(servers, watchers, dev daemons\)\. Async waits for an initial "
            r"idle/output signal, then returns a terminal ID and output snapshot while the "
            r"process continues running\."
        ),
        "Use mode='async' only for long-running processes (servers, watchers, daemons). Returns a terminal ID for later use.",
        "Condense verbose async explanation",
    ),
    (
        _compile(
            r"In sync mode, the full output is returned when the command completes — you do "
            r"NOT need to call get_terminal_output afterward\. Only use get_terminal_output if "
            r"the tool result explicitly says the command was moved to background, timed out, or "
            r"needs input\."
        ),
        "In sync mode, output is returned inline. Only use get_terminal_output if the result indicates the command was moved to background or needs input.",
        "Condense sync output explanation",
    ),
    (
        _compile(
            r"Sync output is final: When a sync command completes, the full output is returned "
            r"inline — do NOT call get_terminal_output afterward\. Only use get_terminal_output "
            r"if the tool result explicitly indicates the command was moved to background, timed "
            r"out, or needs input\. Do NOT tell the user to check the terminal panel — all command "
            r"output is already included in the tool result\."
        ),
        "Sync output is final and returned inline.",
        "Remove redundant paragraph entirely restating the sync behavior",
    ),
    (
        _compile(
            r"Terminal notifications: When an async command finishes or a sync command times out, "
            r"you will be automatically notified on your next turn with the exit code and terminal "
            r"output\. You will also be notified if the terminal needs input\. Do NOT poll or sleep "
            r"to wait for completion\."
        ),
        "For async/timeout commands, you'll be auto-notified on completion. Do not poll.",
        "Condense notification explanation",
    ),
    (
        _compile(
            r"NEVER run sleep or similar wait commands in a terminal\. You will be automatically "
            r"notified on your next turn when async terminal commands or timed-out sync commands "
            r"complete or need input\. Do NOT poll for completion\.\n-"
        ),
        "Do not run sleep or wait commands. You'll be auto-notified on completion.\n-",
        "Condense sleep prohibition",
    ),
    (
        _compile(
            r"NEVER pipe interactive commands through tail, head, grep, or other filters — this "
            r"hides prompts and prevents the terminal from detecting when input is needed\. Run "
            r"interactive commands without pipes\.\n\n"
        ),
        "Do not pipe interactive commands through filters — this hides prompts.\n\n",
        "Condense pipe warning",
    ),
    (
        _compile(
            r"When a terminal command is waiting for interactive input, do NOT suggest "
            r"alternatives or ask the user whether to proceed\. Instead, use the "
            r"vscode_askQuestions tool to collect the needed values from the user, then send them\."
        ),
        "For interactive input prompts, use vscode_askQuestions to collect values from the user.",
        "Condense interactive input guidance",
    ),
    (
        _compile(r"Send exactly one answer per prompt using send_to_terminal\. Never send multiple answers in a single send\."),
        "Send one answer per prompt.",
        "Condense send guidance",
    ),
    (
        _compile(r"After each send, call get_terminal_output to read the next prompt before sending the next answer\."),
        "After sending, call get_terminal_output to read the next prompt.",
        "Condense output reading guidance",
    ),
    (_compile(r"Continue one prompt at a time until the command finishes\."), "", "Remove obvious restatement"),
    (_compile(r"Use \[\[ \]\] for conditional tests instead of \[ \]"), "Use [[ ]] for conditionals", "Simplify"),
    (_compile(r"Prefer \$\(\) over backticks for command substitution"), "Prefer $() over backticks", "Simplify"),
    (_compile(r"Use which or command -v to verify command availability"), "Use `which` to verify command availability.", "Add backtick formatting"),

    # ── Fix insert_edit_into_file verbose example ──
    (
        _compile(r"The system is very smart and can understand how to apply your edits to the notebooks\.\n"),
        "Provide minimal hints — the system applies edits intelligently.",
        "Condense boilerplate",
    ),

    # ── Fix replace_string_in_file verbose warnings ──
    (
        _compile(
            r"CRITICAL for \\?`oldString\\?`: Must uniquely identify the single instance to "
            r"change\. Include at least 3 lines of context BEFORE and AFTER the target text, "
            r"matching whitespace and indentation precisely\. If this string matches multiple "
            r"locations, or does not match exactly, the tool will fail\. Never use 'Lines 123-456 "
            r"omitted' from summarized documents or \.\.\.existing code\.\.\. comments in the "
            r"oldString or newString\."
        ),
        "oldString must uniquely identify one location. Include 3+ lines of surrounding context.",
        "Condense critical warning",
    ),

    # ── Fix manage_todo_list verbose CRITICAL workflow ──
    (
        _compile(
            r"CRITICAL workflow:\s*\n1\. Plan tasks by writing todo list with specific, actionable "
            r"items\s*\n2\. Mark ONE todo as in-progress before starting work\s*\n3\. Complete the "
            r"work for that specific todo\s*\n4\. Mark that todo as completed IMMEDIATELY\s*\n5\. "
            r"Move to next todo and repeat"
        ),
        "Workflow: write todos → mark one as in-progress → complete it → mark completed → repeat.",
        "Condense verbose workflow steps",
    ),

    # ── Fix open_browser_page verbose note ──
    (
        _compile(r'May prompt the user to share a page if there is a similar one already open, unless "forceNew" is true\.'),
        "Set forceNew=true to force a new page; otherwise reuses existing pages.",
        "Condense",
    ),

    # ── Fix runSubagent verbose preamble ──
    (
        _compile(
            r"This tool is good at researching complex questions, searching for code, and executing "
            r"multi-step tasks\. When you are searching for a keyword or file and are not confident "
            r"that you will find the right match in the first few tries, use this agent to perform "
            r"the search for you\."
        ),
        "Use for complex multi-step research, code search, or tasks that may need multiple attempts.",
        "Condense verbose preamble",
    ),
    (
        _compile(r"Agents do not run async or in the background, you will wait for the agent's result\."),
        "Agents run synchronously — wait for results.",
        "Condense",
    ),
    (
        _compile(
            r"When the agent is done, it will return a single message back to you\. The result "
            r"returned by the agent is not visible to the user\. To show the user the result, you "
            r"should send a text message back to the user with a concise summary of the result\."
        ),
        "Agent results aren't shown to users — summarize results in your reply.",
        "Condense",
    ),
    (
        _compile(
            r"Each agent invocation is stateless\. You will not be able to send additional messages "
            r"to the agent, nor will the agent be able to communicate with you outside of its final "
            r"report\. Therefore, your prompt should contain a highly detailed task description for "
            r"the agent to perform autonomously and you should specify exactly what information the "
            r"agent should return back to you in its final and only message to you\."
        ),
        "Agents are stateless. Provide a detailed, self-contained prompt specifying what to return.",
        "Condense statelessness explanation",
    ),
    (_compile(r"The agent's outputs should generally be trusted\n"), "", "Remove unnecessary trust statement"),
    (
        _compile(
            r"Clearly tell the agent whether you expect it to write code or just to do research "
            r"\(search, file reads, web fetches, etc\.\), since it is not aware of the user's intent\n"
        ),
        "Specify whether the agent should write code or only research.",
        "Condense",
    ),
    (
        _compile(r"- If the user asks for a certain agent, you MUST provide that EXACT agent name \(case-sensitive\) to invoke that specific agent\."),
        "Use exact agent names (case-sensitive) when specified.",
        "Condense",
    ),

    # ── Fix vscode_askQuestions verbose parameter docs ──
    (
        _compile(r"Users can always provide a freeform text answer alongside options unless you set allowFreeformInput to false\."),
        "",
        "Remove — already documented in parameter schema",
    ),

    # ── Fix configure_python_environment verbose ALL-CAPS ──
    (
        _compile(
            r"ALWAYS Use this tool to set up the user's chosen environment and ALWAYS call this "
            r"tool before using any other Python related tools or running any Python command in "
            r"the terminal\."
        ),
        "Call this before any other Python tool or command.",
        "Condense ALL-CAPS emphasis",
    ),

    # ── Fix get_terminal_output verbose preamble ──
    (
        _compile(
            r"Get output from a terminal execution that was moved to background \(identified by the "
            r"`id` returned from run_in_terminal\)\. Use this ONLY when the run_in_terminal result "
            r"explicitly says the command was moved to background, timed out, or needs input\. Do "
            r"NOT call this after a sync command that completed normally — sync commands return full "
            r"output inline\. If a background command has not yet completed, you will be "
            r"automatically notified when it finishes — do NOT poll; end your turn and wait\."
        ),
        "Get output from a backgrounded/timed-out terminal. Don't call after successful sync commands. For pending commands, wait for auto-notification.",
        "Condense verbose preamble",
    ),

    # ── Fix memory tool verbose preamble ──
    (
        _compile(
            r"IMPORTANT: Before creating new memory files, first view the /memories/ directory to "
            r"understand what already exists\. This helps avoid duplicates and maintain organized "
            r"notes\."
        ),
        "Check existing files in /memories/ before creating new ones.",
        "Condense",
    ),

    # ── Fix create_new_workspace verbose When NOT to use ──
    (
        _compile(
            r'When NOT to use this tool:\n- Creating single files or small code snippets\n- '
            r'Adding individual files to existing projects\n- Making modifications to existing '
            r'codebases\n- User asks to "create a file" or "add a component"\n- Simple code '
            r'examples or demonstrations\n- Debugging or fixing existing code'
        ),
        "",
        "Remove generic 'When NOT to use' boilerplate",
    ),

    # ── Remove standalone "Do NOT" lines that restate earlier rules ──
    (
        _compile(r"Do NOT tell the user to check the terminal panel — all command output is already included in the tool result\."),
        "",
        "Redundant with sync output explanation",
    ),

    # ── Fix create_file description ──
    (
        _compile(
            r"This is a tool for creating a new file in the workspace\. The file will be created "
            r"with the specified content\. The directory will be created if it does not already "
            r"exist\. Never use this tool to edit a file that already exists\."
        ),
        "Create a new file. Directories are auto-created. Do not use for editing existing files.",
        "Condense verbose description",
    ),

    # ── Fix read_file description ──
    (
        _compile(
            r"You must specify the line range you're interested in\. Line numbers are 1-indexed\. "
            r"If the file contents returned are insufficient for your task, you may call this tool "
            r"again to retrieve more content\. Prefer reading larger ranges over doing many small "
            r"reads\. Binary files use startLine/endLine as byte offsets\."
        ),
        "Specify 1-indexed line ranges. Prefer larger reads over many small ones. For binary files, ranges are byte offsets.",
        "Condense",
    ),

    # ── Fix grep_search verbose preamble ──
    (
        _compile(
            r"Do a fast text search in the workspace\. Use this tool when you want to search with "
            r"an exact string or regex\. If you are not sure what words will appear in the "
            r"workspace, prefer using regex patterns with alternation \(\|\) or character classes to "
            r"search for multiple potential words at once instead of making separate searches\. For "
            r"example, use 'function\|method\|procedure' to look for all of those words at once\. "
            r"Use includePattern to search within files matching a specific pattern, or in a "
            r"specific file, using a relative path\. Use 'includeIgnoredFiles' to include files "
            r"normally ignored by \.gitignore, other ignore files, and `files\.exclude` and "
            r"`search\.exclude` settings\. Warning: using this may cause the search to be slower, "
            r"only set it when you want to search in ignored folders like node_modules or build "
            r"outputs\. Use this tool when you want to see an overview of a particular file, instead "
            r"of using read_file many times to look for code within a file\."
        ),
        "Fast text/regex search across workspace files. Use regex alternation (e.g. 'word1|word2') for broad searches. Use includePattern to scope to specific files. Set includeIgnoredFiles=true to search node_modules/build outputs (slower).",
        "Condense verbose preamble",
    ),

    # ── Fix file_search verbose examples ──
    (
        _compile(
            r"Search for files in the workspace by glob pattern\. This only returns the paths of "
            r"matching files\. Use this tool when you know the exact filename pattern of the files "
            r"you're searching for\. Glob patterns match from the root of the workspace folder\. "
            r"Examples:\s*\n\s*- \*\*/\*\.\{js,ts\} to match all js/ts files in the workspace\.\s*\n"
            r"\s*- src/\*\* to match all files under the top-level src folder\.\s*\n\s*- "
            r"\*\*/foo/\*\*/\*\.js to match all js files under any foo folder in the workspace\.\s*"
            r"\n\s*In a multi-root workspace, you can scope the search to a specific workspace "
            r"folder by using the absolute path to the folder as the query, e\.g\. "
            r"/path/to/folder/\*\*/\*\.ts\."
        ),
        "Find files by glob pattern (e.g. '**/*.ts', 'src/**'). Returns matching paths only.",
        "Condense verbose examples",
    ),

    # ── Fix session_store_sql verbose preamble ──
    (
        _compile(
            r"Query the local session store containing history from past coding sessions\. Uses "
            r"SQLite syntax \(NOT DuckDB or Postgres\)\. SQL queries are read-only — only SELECT "
            r"and WITH are allowed\. Use `datetime\('now', '-1 day'\)` for date math \(NOT "
            r"`now\(\) - INTERVAL '1 day'`\), FTS5 `MATCH` for text search\."
        ),
        "Read-only SQLite queries against session history. Only SELECT/WITH allowed. Use datetime('now','-1 day') for dates, FTS5 MATCH for text search.",
        "Condense",
    ),
]


def _bytes_to_tokens(num_bytes: int) -> int:
    """Approximate token count from byte count (≈4 bytes/token for UTF-8 text)."""
    return round(num_bytes / 4)


def optimize_system_message(messages: Messages) -> int:
    """Condense the system message in-place.

    Returns the number of bytes saved.
    """
    if not messages:
        return 0
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return 0
    content = first.get("content")
    if not isinstance(content, str):
        return 0
    start_len = len(content.encode("utf-8"))
    new_content = SEARCH_VSC.sub(REPLACE_VSC, content)
    if new_content == content:
        return 0
    first["content"] = new_content
    return start_len - len(new_content.encode("utf-8"))


def _tool_name(tool: Any) -> str:
    """Extract the function name from an OpenAI-style tool definition."""
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("name", "") or ""
    return tool.get("name", "") or ""


def _tool_description(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("description", "") or ""
    return tool.get("description", "") or ""


def _set_tool_description(tool: Any, desc: str) -> None:
    if not isinstance(tool, dict):
        return
    fn = tool.get("function")
    if isinstance(fn, dict):
        fn["description"] = desc
    else:
        tool["description"] = desc


def optimize_tools(tools: List[dict]) -> Tuple[List[dict], int, Dict[str, str]]:
    """Condense tool descriptions and remove unused tools.

    Returns:
        (filtered_tools, bytes_saved, logs)
    """
    if not tools:
        return tools, 0, {}

    logs: Dict[str, str] = {}
    saved_bytes = 0

    start_len = len(_dump_tools(tools).encode("utf-8"))

    # ── Condense descriptions ──
    condensed: List[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            condensed.append(tool)
            continue
        desc = _tool_description(tool)
        if not desc:
            condensed.append(tool)
            continue

        for i, (pattern, replacement, reason) in enumerate(REPLACEMENTS):
            if pattern.search(desc):
                logs[f"-{i:02d}#"] = reason
                desc = pattern.sub(replacement, desc)

        # Collapse whitespace artifacts left by replacements
        desc = re.sub(r" {2,}", " ", desc)
        desc = re.sub(r"\n{3,}", "\n\n", desc)

        _set_tool_description(tool, desc)
        condensed.append(tool)

    # ── Remove unused tools ──
    filtered: List[dict] = []
    for i, tool in enumerate(condensed):
        name = _tool_name(tool)
        if name in REMOVE_TOOLS:
            logs[f"tool-{i:02d}-"] = name
            continue
        logs[f"tool-{i:02d}+"] = name
        filtered.append(tool)

    end_len = len(_dump_tools(filtered).encode("utf-8"))
    saved_bytes = max(0, start_len - end_len)

    return filtered, saved_bytes, logs


def _dump_tools(tools: List[dict]) -> str:
    """Stable serialization for length comparison."""
    import json
    return json.dumps(tools, ensure_ascii=False, sort_keys=True)


# ── Message-level optimization ──────────────────────────────────────────────
# These run *before* the request reaches any provider so the saved tokens are
# detected centrally in ``run_tools`` rather than being silently dropped by
# each provider's own message-cleaning logic.

def _msg_bytes(msg: Any) -> int:
    """Approximate byte size of a single message's content."""
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    if isinstance(content, str):
        return len(content.encode("utf-8", errors="replace"))
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text.encode("utf-8", errors="replace"))
        return total
    return 0


def _msg_signature(msg: Any) -> Tuple[str, int]:
    """A stable signature for duplicate detection.

    Returns ``(role, content_hash)``. Two messages with the same signature are
    considered duplicates for the purposes of collapsing.
    """
    if not isinstance(msg, dict):
        return ("", 0)
    role = msg.get("role", "") or ""
    content = msg.get("content")
    if isinstance(content, str):
        return (role, hash(content))
    if isinstance(content, list):
        # Hash the concatenated text of all parts so structurally identical
        # multi-part messages compare equal.
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return (role, hash("\n".join(parts)))
    return (role, 0)


def _is_empty_content(msg: Any) -> bool:
    """True when a message carries no usable content."""
    if not isinstance(msg, dict):
        return True
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return False


def dedup_messages(messages: Messages) -> tuple[Messages, int]:
    """Remove duplicate and redundant messages.

    * Drops empty/whitespace-only messages (except the system prompt).
    * Removes exact duplicate messages (same role + content), keeping first.
    * Collapses consecutive same-role messages, keeping the one with
      tool_calls or more content.

    Returns (messages, bytes_saved).
    """
    if not messages:
        return [], 0

    original_bytes = sum(_msg_bytes(m) for m in messages)

    seen: set = set()
    result: list = []

    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        role = msg.get("role", "")

        # Always keep system messages.
        if role == "system":
            result.append(msg)
            continue

        # Drop empty messages, but keep assistant messages that carry tool_calls.
        if _is_empty_content(msg) and not msg.get("tool_calls"):
            continue

        # Collapse consecutive same-role messages.
        if result and isinstance(result[-1], dict) and result[-1].get("role") == role:
            prev = result[-1]
            # Keep the message that has tool_calls.
            if msg.get("tool_calls") and not prev.get("tool_calls"):
                result.pop()
                result.append(msg)
                continue
            # If previous has tool_calls and this one doesn't, skip this one.
            if prev.get("tool_calls") and not msg.get("tool_calls"):
                continue
            # Neither has tool_calls — skip the duplicate.
            continue

        # Remove exact duplicates (same role + content hash).
        sig = _msg_signature(msg)
        if sig[1] and sig in seen:
            continue
        seen.add(sig)

        result.append(msg)

    new_bytes = sum(_msg_bytes(m) for m in result)
    return result, max(0, original_bytes - new_bytes)


# ── Tool-loop detection ─────────────────────────────────────────────────────

_MAX_TOOL_REPEATS = 3  # max times the same tool call may appear before breaking


def _tool_call_signature(tool_calls: list) -> str:
    """Build a stable signature from a list of tool calls.

    Two tool-call lists with the same function names and (semantically)
    identical arguments produce the same signature.
    """
    parts: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        # Normalise JSON arguments so trivial differences (key order,
        # whitespace) don't defeat dedup.
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                pass
        parts.append(f"{name}:{args}")
    return "|".join(parts)


def break_tool_loop(messages: Messages, max_repeats: int = _MAX_TOOL_REPEATS) -> int:
    """Detect and break tool-call loops.

    When the model repeatedly calls the same tool(s) with the same arguments
    (getting the same results), this function:

    * Keeps only the **first** occurrence of the repeated tool call and its
      tool-result messages.
    * Removes all subsequent duplicate assistant calls **and** their
      corresponding ``tool``/``function`` result messages.
    * Injects a guidance ``user`` message telling the model to stop
      repeating and try a different approach.

    Returns bytes saved.
    """
    if not messages:
        return 0

    # Group assistant-with-tool_calls indices by their call signature.
    sig_groups: dict[str, list[int]] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tc = msg.get("tool_calls")
        if not tc:
            continue
        sig = _tool_call_signature(tc)
        if not sig:
            continue
        sig_groups.setdefault(sig, []).append(i)

    # Only act when a signature repeats enough times.
    looped_sigs = {
        sig for sig, indices in sig_groups.items()
        if len(indices) >= max_repeats
    }
    if not looped_sigs:
        return 0

    # Mark indices to remove: duplicate assistant messages + their tool results.
    indices_to_remove: set[int] = set()
    for sig in looped_sigs:
        indices = sig_groups[sig]
        # Keep the first occurrence, remove the rest.
        for idx in indices[1:]:
            indices_to_remove.add(idx)
            # Also remove the following tool/function result messages.
            for j in range(idx + 1, len(messages)):
                m = messages[j]
                if not isinstance(m, dict):
                    break
                if m.get("role") in ("tool", "function"):
                    indices_to_remove.add(j)
                else:
                    break

    if not indices_to_remove:
        return 0

    saved_bytes = sum(_msg_bytes(messages[i]) for i in indices_to_remove)

    # Build the new message list without the removed indices.
    new_messages = [
        msg for i, msg in enumerate(messages) if i not in indices_to_remove
    ]

    # Inject a guidance message after the first looped assistant's results.
    for sig in looped_sigs:
        first_idx = sig_groups[sig][0]
        removed_before = sum(1 for i in indices_to_remove if i < first_idx)
        adjusted_idx = first_idx - removed_before

        # Find where the tool results end.
        insert_pos = adjusted_idx + 1
        for j in range(adjusted_idx + 1, len(new_messages)):
            m = new_messages[j]
            if not isinstance(m, dict):
                break
            if m.get("role") in ("tool", "function"):
                insert_pos = j + 1
            else:
                break

        guidance = {
            "role": "user",
            "content": (
                "[SYSTEM] You are repeating the same tool calls with identical "
                "arguments. The previous results did not change. Do NOT call the "
                "same tools again with the same arguments. Try a different approach, "
                "use different search terms, or proceed with the information you "
                "already have."
            ),
        }
        if 0 <= insert_pos <= len(new_messages):
            new_messages.insert(insert_pos, guidance)
        break  # Only inject one guidance message.

    messages[:] = new_messages
    return max(0, saved_bytes)


def strip_reasoning_echo(messages: Messages) -> int:
    """Remove reasoning/thinking blocks echoed back in later assistant messages.

    Providers sometimes re-emit the same ``<think>…</think>`` reasoning block
    across consecutive assistant turns. Keeping only the first occurrence
    avoids paying for the same reasoning tokens twice.

    Returns the number of bytes saved.
    """
    if not messages:
        return 0

    _THINK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
    _REASONING_TAG = re.compile(
        r"<reasoning[\s\S]*?</reasoning>", re.IGNORECASE
    )

    seen_think = False
    seen_reasoning = False
    saved_bytes = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        new_content = content
        if _THINK.search(new_content):
            if seen_think:
                before = len(new_content.encode("utf-8", errors="replace"))
                new_content = _THINK.sub("", new_content)
                after = len(new_content.encode("utf-8", errors="replace"))
                saved_bytes += before - after
            else:
                seen_think = True
        if _REASONING_TAG.search(new_content):
            if seen_reasoning:
                before = len(new_content.encode("utf-8", errors="replace"))
                new_content = _REASONING_TAG.sub("", new_content)
                after = len(new_content.encode("utf-8", errors="replace"))
                saved_bytes += before - after
            else:
                seen_reasoning = True

        if new_content != content:
            # Clean up leftover blank lines from the removals.
            new_content = re.sub(r"\n{3,}", "\n\n", new_content).strip()
            if not new_content:
                # The whole message was reasoning — drop it.
                saved_bytes += len(content.encode("utf-8", errors="replace"))
                messages[i] = {"role": "assistant", "content": ""}
            else:
                msg["content"] = new_content

    return max(0, saved_bytes)


# ── Tool result truncation ──────────────────────────────────────────────────

# Cap the byte size of any single tool result / function call output embedded
# in the conversation. Older results are rarely re-read by the model but still
# consume the full input budget on every turn.
_TOOL_RESULT_CAP = 4096  # bytes per tool result
_OLD_TOOL_RESULT_CAP = 1200  # stricter cap for results older than 2 turns


def _truncate_tool_results(messages: Messages) -> int:
    """Truncate oversized tool/function call results in place.

    Keeps the head and tail of each result with an omission marker, so the
    model still sees the start (usually the most relevant part) and the
    exit status at the end. Older results are capped more aggressively.

    Returns the number of bytes saved.
    """
    if not messages:
        return 0

    saved_bytes = 0
    # Count tool-role messages from the end so we can apply the stricter cap
    # to older ones.
    tool_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in ("tool", "function")
    ]
    # Reverse so index 0 = newest, 1 = second newest, etc.
    tool_indices.reverse()

    for age, idx in enumerate(tool_indices):
        msg = messages[idx]
        cap = _OLD_TOOL_RESULT_CAP if age >= 2 else _TOOL_RESULT_CAP

        content = msg.get("content")
        if isinstance(content, str):
            raw = content.encode("utf-8", errors="replace")
            if len(raw) <= cap:
                continue
            before = len(raw)
            head = cap // 2
            tail = cap // 4
            new_content = (
                content[:head]
                + f"\n... [{before - head - tail} chars truncated] ...\n"
                + content[-tail:]
            )
            msg["content"] = new_content
            saved_bytes += before - len(new_content.encode("utf-8", errors="replace"))
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not isinstance(text, str):
                    continue
                raw = text.encode("utf-8", errors="replace")
                if len(raw) <= cap:
                    continue
                before = len(raw)
                head = cap // 2
                tail = cap // 4
                new_text = (
                    text[:head]
                    + f"\n... [{before - head - tail} chars truncated] ...\n"
                    + text[-tail:]
                )
                part["text"] = new_text
                saved_bytes += before - len(new_text.encode("utf-8", errors="replace"))

    return max(0, saved_bytes)


# ── Strip redundant tool_call fields ─────────────────────────────────────────

def _strip_redundant_tool_fields(messages: Messages) -> int:
    """Remove fields from assistant messages that the provider doesn't need.

    Some providers echo back ``tool_calls`` on the assistant message *and*
    keep the matching ``tool`` role result. The echoed call metadata is
    redundant once the result is present. We drop:
    * ``tool_calls`` on assistant messages that already have a following
      ``tool``/``function`` result (keeps the last occurrence only).
    * Empty ``function_call`` fields.

    Returns bytes saved.
    """
    if not messages:
        return 0

    import json
    saved_bytes = 0

    # Find assistant messages with tool_calls that are followed by a tool result.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tc = msg.get("tool_calls")
        if not tc:
            continue
        # Check if a subsequent tool message references this call.
        has_result = False
        for j in range(i + 1, len(messages)):
            m = messages[j]
            if not isinstance(m, dict):
                continue
            if m.get("role") in ("tool", "function"):
                has_result = True
                break
            if isinstance(m, dict) and m.get("role") == "assistant":
                # Next assistant turn — stop looking.
                break
        if has_result:
            before = len(json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
            msg.pop("tool_calls", None)
            # Also drop the now-orphaned function_call if present.
            msg.pop("function_call", None)
            after = len(json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
            saved_bytes += max(0, before - after)

    return max(0, saved_bytes)


# ── Collapse whitespace in message content ──────────────────────────────────

_WS_RE = re.compile(r"[ \t]+\n")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _collapse_message_whitespace(messages: Messages) -> int:
    """Normalize trailing whitespace and repeated blank lines in all messages.

    Returns bytes saved.
    """
    if not messages:
        return 0

    saved_bytes = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and len(content) > 64:
            original = len(content.encode("utf-8", errors="replace"))
            new = _WS_RE.sub("\n", content)
            new = _BLANK_RUN_RE.sub("\n\n", new)
            if new != content:
                saved_bytes += original - len(new.encode("utf-8", errors="replace"))
                msg["content"] = new
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and len(text) > 64:
                        original = len(text.encode("utf-8", errors="replace"))
                        new = _WS_RE.sub("\n", text)
                        new = _BLANK_RUN_RE.sub("\n\n", new)
                        if new != text:
                            saved_bytes += original - len(new.encode("utf-8", errors="replace"))
                            part["text"] = new
    return max(0, saved_bytes)


# ── Drop stale context (old user turns beyond a threshold) ───────────────────

_MAX_TURNS = 40  # keep at most this many non-system messages


def _trim_old_turns(messages: Messages) -> int:
    """Drop the oldest non-system messages when the conversation is very long.

    Keeps the system prompt and the most recent ``_MAX_TURNS`` messages.
    Returns bytes saved.
    """
    if not messages or len(messages) <= _MAX_TURNS + 1:
        return 0

    # Separate system messages (kept) from the rest.
    system_msgs = []
    rest = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_msgs.append(m)
        else:
            rest.append(m)

    if len(rest) <= _MAX_TURNS:
        return 0

    dropped = rest[:-_MAX_TURNS]
    kept = rest[-_MAX_TURNS:]
    saved_bytes = sum(_msg_bytes(m) for m in dropped)

    messages[:] = system_msgs + kept
    return max(0, saved_bytes)


def optimize_request(messages: Messages, tools: Any) -> Tuple[int, Dict[str, str]]:
    """Truncate very old tool-result messages to a head+tail snippet.

    Only messages *before* the last user/assistant turn are truncated so the
    most recent context is preserved verbatim.  Returns bytes saved.
    """
    if not messages:
        return 0

    # Find the index of the last "fresh" turn — the last user or assistant
    # message that is not an empty tool-response.  Messages before that
    # index are candidates for truncation.
    last_fresh = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("user", "assistant") and msg.get("content"):
            last_fresh = i
            break
    if last_fresh <= 0:
        return 0

    saved_bytes = 0
    for i in range(0, last_fresh):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        raw = content.encode("utf-8", errors="replace")
        if len(raw) <= _TOOL_RESULT_CAP:
            continue
        head = raw[:_TOOL_RESULT_KEEP_HEAD].decode("utf-8", errors="replace")
        tail = raw[-_TOOL_RESULT_KEEP_TAIL:].decode("utf-8", errors="replace")
        omitted = len(raw) - _TOOL_RESULT_KEEP_HEAD - _TOOL_RESULT_KEEP_TAIL
        msg["content"] = (
            f"{head}\n\n... [{omitted} bytes omitted — old tool result truncated] ...\n\n{tail}"
        )
        saved_bytes += len(raw) - len(msg["content"].encode("utf-8", errors="replace"))
    return max(0, saved_bytes)


def _strip_redundant_tool_calls(messages: Messages) -> int:
    """Remove ``tool_calls`` from assistant messages once the corresponding
    tool result has been returned.  Providers keep the full tool-call spec
    (function name + arguments) in history even after the result is in
    context, which wastes tokens.  Returns bytes saved.
    """
    if not messages:
        return 0
    import json as _json

    saved_bytes = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        # Is there a later ``tool`` message in the conversation?  If so the
        # result is already in context and the call spec is redundant.
        has_result = any(
            isinstance(m, dict) and m.get("role") == "tool"
            for m in messages[i + 1:]
        )
        if not has_result:
            continue
        before = len(_json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
        # Keep only the id so providers can still correlate, strip verbose keys.
        stripped_calls = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                stripped_calls.append(tc)
                continue
            tc = {k: v for k, v in tc.items() if k not in _TOOL_CALL_STRIP_KEYS}
            # Replace the full arguments with a short marker — the actual
            # arguments are recoverable from the tool-result message.
            fn = tc.get("function")
            if isinstance(fn, dict) and fn.get("arguments"):
                fn = {k: v for k, v in fn.items() if k != "arguments"}
                fn["arguments"] = "{}"
                tc["function"] = fn
            stripped_calls.append(tc)
        msg["tool_calls"] = stripped_calls
        after = len(_json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
        saved_bytes += max(0, before - after)
    return saved_bytes


def _collapse_whitespace(messages: Messages) -> int:
    """Collapse runs of blank lines and trailing whitespace inside every
    string content.  Returns bytes saved.
    """
    saved_bytes = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        new = re.sub(r"[ \t]+\n", "\n", content)      # trailing spaces
        new = re.sub(r"\n{3,}", "\n\n", new)            # blank-line runs
        new = new.strip()
        if len(new) < len(content):
            saved_bytes += len(content.encode("utf-8", errors="replace")) - len(new.encode("utf-8", errors="replace"))
            if new:
                msg["content"] = new
            else:
                msg["content"] = ""
    return max(0, saved_bytes)


# ──────────────────────────────────────────────────────────────────────
# New optimizers
# ──────────────────────────────────────────────────────────────────────

# Cap on how many bytes of a *single* tool-result message we keep.
# Anything longer is truncated to head+tail with an omission marker.
_TOOL_RESULT_CAP = 4000  # ≈1000 tokens per tool result is plenty for context

_TOOL_RESULT_HEAD = 1500
_TOOL_RESULT_TAIL = 1500

# Fields on message dicts that some providers echo back but that are not
# needed for the next turn (the assistant already produced them).
_DROP_ASSISTANT_FIELDS = ("tool_calls", "function_call", "name", "refusal")


def _content_bytes(content: Any) -> int:
    """Byte length of a message's content (str or list of parts)."""
    if isinstance(content, str):
        return len(content.encode("utf-8", errors="replace"))
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += len(str(part.get("text", "")).encode("utf-8", errors="replace"))
            elif isinstance(part, str):
                total += len(part.encode("utf-8", errors="replace"))
        return total
    return 0


def _set_content(msg: dict, content: Any) -> None:
    """Set content on a message dict, handling both str and list forms."""
    msg["content"] = content


def truncate_tool_results(messages: Messages) -> int:
    """Truncate very long ``tool`` role messages to a head+tail cap.

    Tool results (e.g. web-search dumps, file contents) are often huge but
    only the most recent lines matter for the next turn.  We keep the first
    ``_TOOL_RESULT_HEAD`` bytes (so the model knows what was returned) and
    the last ``_TOOL_RESULT_TAIL`` bytes (the freshest output), dropping the
    middle.

    Returns bytes saved.
    """
    saved = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        b = content.encode("utf-8", errors="replace")
        if len(b) <= _TOOL_RESULT_CAP:
            continue
        head = b[:_TOOL_RESULT_HEAD].decode("utf-8", errors="replace")
        tail = b[-_TOOL_RESULT_TAIL:].decode("utf-8", errors="replace")
        omitted = len(b) - _TOOL_RESULT_HEAD - _TOOL_RESULT_TAIL
        new_content = f"{head}\n\n... [{omitted} chars truncated by optimize_request] ...\n\n{tail}"
        saved += len(b) - len(new_content.encode("utf-8", errors="replace"))
        _set_content(msg, new_content)
    return max(0, saved)


def strip_redundant_assistant_fields(messages: Messages) -> int:
    """Remove fields on assistant messages that are no longer needed.

    Once an assistant turn is in the history, the ``tool_calls`` /
    ``function_call`` metadata is redundant — the corresponding ``tool``
    messages that follow already carry the result.  Dropping these fields
    avoids providers re-serialising them on every turn.

    Returns bytes saved (approximated from JSON length).
    """
    import json as _json

    saved = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        before = len(_json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
        changed = False
        for field in _DROP_ASSISTANT_FIELDS:
            if field in msg:
                # Only drop tool_calls if there is content to keep, so we
                # never produce an empty assistant message.
                if field == "tool_calls" and not msg.get("content"):
                    continue
                del msg[field]
                changed = True
        if changed:
            after = len(_json.dumps(msg, ensure_ascii=False).encode("utf-8", errors="replace"))
            saved += before - after
    return max(0, saved)


def collapse_whitespace_in_messages(messages: Messages) -> int:
    """Collapse runs of 3+ newlines and trailing whitespace inside message content.

    Returns bytes saved.
    """
    saved = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        new = re.sub(r"[ \t]+\n", "\n", content)          # trailing spaces on lines
        new = re.sub(r"\n{3,}", "\n\n", new)               # 3+ newlines → 2
        new = new.strip()
        if len(new) != len(content):
            saved += len(content.encode("utf-8", errors="replace")) - len(new.encode("utf-8", errors="replace"))
            _set_content(msg, new)
    return max(0, saved)


def drop_empty_trailing_messages(messages: Messages) -> int:
    """Drop trailing messages with empty content (no value for the next turn).

    Returns bytes saved (always 0 — these are empty, but we count messages
    removed so callers can log it).
    """
    removed = 0
    while messages:
        last = messages[-1]
        if not isinstance(last, dict):
            break
        content = last.get("content")
        if content in (None, "", []):
            messages.pop()
            removed += 1
        else:
            break
    return removed


def optimize_request(messages: Messages, tools: Any) -> Tuple[int, Dict[str, str]]:
    """Optimize the system message, messages, and tools in-place.

    Mutates ``messages`` (system prompt + message list) and ``tools`` (list,
    replaced in place via slice assignment when possible) and returns
    (saved_tokens, logs).

    ``tools`` may be a list (mutated in place) or None.
    """
    saved_bytes = 0
    logs: Dict[str, str] = {}

    # Capture baseline size for percentage calculation.
    baseline_bytes = sum(_msg_bytes(m) for m in messages)
    if isinstance(tools, list) and tools:
        baseline_bytes += len(_dump_tools(tools).encode("utf-8"))

    # ── System prompt ──
    sys_saved = optimize_system_message(messages)
    if sys_saved:
        saved_bytes += sys_saved
        logs["system"] = f"condensed system prompt (-{sys_saved} bytes)"

    # ── Message-level dedup & reasoning echo removal ──
    messages, dedup_saved = dedup_messages(messages)
    if dedup_saved:
        saved_bytes += dedup_saved
        logs["dedup"] = f"removed duplicate/empty messages (-{dedup_saved} bytes)"

    # ── Break tool-call loops ──
    loop_saved = break_tool_loop(messages)
    if loop_saved:
        saved_bytes += loop_saved
        logs["tool_loop"] = f"broke tool-call loop (-{loop_saved} bytes)"

    echo_saved = strip_reasoning_echo(messages)
    if echo_saved:
        saved_bytes += echo_saved
        logs["reasoning_echo"] = f"stripped repeated reasoning blocks (-{echo_saved} bytes)"

    # ── Tool result truncation ──
    tool_trunc_saved = _truncate_tool_results(messages)
    if tool_trunc_saved:
        saved_bytes += tool_trunc_saved
        logs["tool_trunc"] = f"truncated oversized tool results (-{tool_trunc_saved} bytes)"

    # ── Strip redundant tool_call fields ──
    tool_field_saved = _strip_redundant_tool_fields(messages)
    if tool_field_saved:
        saved_bytes += tool_field_saved
        logs["tool_fields"] = f"stripped redundant tool_call fields (-{tool_field_saved} bytes)"

    # ── Collapse whitespace ──
    ws_saved = _collapse_message_whitespace(messages)
    if ws_saved:
        saved_bytes += ws_saved
        logs["whitespace"] = f"collapsed whitespace (-{ws_saved} bytes)"

    # ── Trim old turns ──
    trim_saved = _trim_old_turns(messages)
    if trim_saved:
        saved_bytes += trim_saved
        logs["trim_old"] = f"dropped {trim_saved} bytes of stale turns"

    # ── Tools ──
    if isinstance(tools, list) and tools:
        filtered, tool_saved, tool_logs = optimize_tools(tools)
        # Mutate the original list in place so callers holding the reference see
        # the changes.
        tools[:] = filtered
        saved_bytes += tool_saved
        #logs.update(tool_logs)
        logs["tools"] = f"optimized tools (-{tool_saved} bytes)"

    # Report overall savings as a percentage of the baseline.
    if baseline_bytes > 0 and saved_bytes > 0:
        pct = (saved_bytes / baseline_bytes) * 100
        saved_tokens = _bytes_to_tokens(saved_bytes)
        logs["summary"] = (
            f"saved {saved_tokens} tokens / {baseline_bytes} baseline "
            f"({pct:.1f}%)"
        )

    return _bytes_to_tokens(saved_bytes), logs