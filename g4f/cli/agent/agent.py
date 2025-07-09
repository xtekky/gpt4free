"""Main G4F Agent implementation."""

import os
import json
import re
import shutil
import subprocess
import hashlib
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

# Dependency imports with error handling
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Confirm, Prompt
    from pygments.lexers import get_lexer_for_filename
    from pygments.util import ClassNotFound
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install rich pyyaml pygments")
    exit(1)

from .config import Config, CONFIG_DIR, ApprovalMode
from .actions import Action, ActionType
from .memory import AgentMemory
from .security import SecuritySandbox
from .git import GitIntegration
from .dependencies import DependencyManager
from .testing import TestRunner

# Initialize Rich Console
console = Console()


class ResponseCache:
    """Caches AI responses to reduce API calls and costs."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_dir / "responses.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f: self.cache = json.load(f)
            except (json.JSONDecodeError, IOError): self.cache = {}

    def _save_cache(self):
        with open(self.cache_dir / "responses.json", "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, prompt: str, context: str) -> str:
        return hashlib.md5(f"{prompt}{context}".encode()).hexdigest()

    def get(self, prompt: str, context: str) -> Optional[str]:
        return self.cache.get(self._get_cache_key(prompt, context))

    def set(self, prompt: str, context: str, response: str):
        self.cache[self._get_cache_key(prompt, context)] = response
        self._save_cache()


class ActionHistory:
    """Tracks and enables undo/redo for file system actions."""
    def __init__(self, history_dir: Path):
        self.history_dir = history_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        self.undo_stack: List[Dict] = []
        self.redo_stack: List[Dict] = []
        self._load_history()

    def _load_history(self):
        history_file = self.history_dir / "actions.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f: data = json.load(f)
                self.undo_stack = data.get("undo", [])
                self.redo_stack = data.get("redo", [])
            except (json.JSONDecodeError, IOError): pass

    def _save_history(self):
        with open(self.history_dir / "actions.json", "w", encoding="utf-8") as f:
            history_data = {"undo": self.undo_stack[-50:], "redo": self.redo_stack[-50:]}
            json.dump(history_data, f, indent=2)

    def record(self, action: Action, before_state: Dict[str, Any]):
        action_dict = {k: v for k, v in action.__dict__.items() if v is not None}
        action_dict["type"] = action.type.value
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action_dict,
            "before_state": before_state
        }
        self.undo_stack.append(record)
        self.redo_stack.clear()
        self._save_history()

    def can_undo(self) -> bool: return bool(self.undo_stack)
    def get_last_action(self) -> Optional[Dict[str, Any]]: return self.undo_stack[-1] if self.undo_stack else None


class EnhancedAgent:
    """
    An advanced AI agent that understands project context, plans actions,
    and interacts with the user to accomplish complex development tasks.
    """
    def __init__(self, config: Config):
        self.config = config
        self.console = console
        self.memory = AgentMemory(CONFIG_DIR)
        self.security = SecuritySandbox()
        self.git = GitIntegration()
        self.deps = DependencyManager()
        self.test_runner = TestRunner()
        self.response_cache = ResponseCache(CONFIG_DIR)
        self.action_history = ActionHistory(CONFIG_DIR)
        self.debug_mode = getattr(config, 'debug_mode', False)

        if self.config.git_integration and self.git.is_git_repo():
            self.backup_branch = self.git.create_backup_branch()
            if self.backup_branch:
                self.console.print(f"[green]✓[/green] Created backup branch: {self.backup_branch}")

    def debug_print(self, message: str, title: str = "DEBUG"):
        if self.debug_mode:
            self.console.print(Panel(str(message), title=f"🐛 {title}", style="dim", border_style="yellow"))

    def _get_system_prompt(self) -> str:
        """Constructs the system prompt with rules, context, and conversation history."""
        project_context = self.memory.get_context()
        conversation_context = self.memory.get_conversation_context()

        return f"""You are an autonomous AI software developer agent. Your sole purpose is to execute user requests by generating a sequence of actions in a JSON array. You are a tool for DOING, not for talking.

**CORE DIRECTIVES:**
1.  **BIAS FOR ACTION:** Your primary goal is to complete the user's task. Do not engage in conversation. Do not ask for clarification unless a request is impossible to interpret. Directly generate the actions needed to fulfill the request.
2.  **JSON ONLY:** You MUST respond with a valid JSON array of action objects. No other text or explanation is allowed outside the JSON structure.
3.  **OBSERVE AND ADAPT:** After you perform an action like `read` or `list`, the system will provide the result back to you in an `observation` block. You MUST use the information from the observation to plan your next action (e.g., use the content from a `read` observation to generate an `edit` action).
4.  **COMPLETE THE GOAL:** Continue generating actions until the user's request is fully completed. If you read a file to gather information, your next step must be to USE that information.
5.  **SELF-CONTAINED ACTIONS:** For tasks like translation, code generation, or modification, perform the task yourself within the `content` field of an `edit` or `create` action. DO NOT use `web_search` for tasks you are capable of doing yourself.

**EXAMPLE WORKFLOW:**
-   **User:** "translate the print statement in the hello function to Ukrainian"
-   **Your 1st Response:** `[{{"type": "read", "target": "app.py", "reason": "Need to read the file to find the hello function."}}]`
-   **System Observation:** `Content of app.py:\n```\ndef hello():\n    print("Hello, World!")\n```
-   **Your 2nd Response:** `[{{"type": "edit", "target": "app.py", "line_start": 2, "line_end": 2, "content": "    print(\\"Привіт, Світ!\\")", "reason": "Translating the print statement as requested based on the file content."}}]`

**AVAILABLE ACTIONS (JSON format):**
- `list`: List files in a directory. `{{"type": "list", "target": "./src"}}`
- `create`: Create a new file. `{{"type": "create", "target": "new.py", "content": "# New file content"}}`
- `read`: Read the content of a file. `{{"type": "read", "target": "main.py"}}`
- `edit`: Edit an existing file. `{{"type": "edit", "target": "main.py", "line_start": 10, "line_end": 12, "content": "new_function_code..."}}`
- `delete`: Delete a file or directory. `{{"type": "delete", "target": "old_file.txt"}}`
- `execute`: Execute a shell command. `{{"type": "execute", "method": "python", "content": "print('hello')"`
- `clarify`: Ask the user a question (USE SPARINGLY). `{{"type": "clarify", "question": "The request is ambiguous. Do you mean X or Y?"}}`

**PROVIDED CONTEXT:**
<project_context>
{project_context}
</project_context>

<conversation_history>
{conversation_context}
</conversation_history>
"""

    def run(self, prompt: str, is_follow_up: bool = False):
        """The main loop for the agent to process a user prompt."""
        import g4f

        if prompt.lower().strip() == "undo":
            self.undo_last_action()
            return
        
        if not is_follow_up:
            self.memory.add_conversation("user", prompt)

        system_prompt = self._get_system_prompt()
        
        # --- MODIFICATION START: Construct the final message list here ---
        # This ensures the correct, detailed system prompt is always used.
        messages = [{"role": "system", "content": system_prompt}] + self.memory.conversation_history
        # --- MODIFICATION END ---
        
        self.debug_print("\n".join([f"{m['role']}: {m['content'][:300]}" for m in messages]), "API_MESSAGES")

        # The cache key should be based on the full context sent to the model
        cache_context = "".join([m['content'] for m in messages])
        cached_response = self.response_cache.get(prompt, cache_context)
        
        if cached_response:
            response = cached_response
            self.debug_print("Using cached response.", "CACHE")
        else:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as p:
                p.add_task("🧠 Thinking...", total=None)
                try:
                    response = g4f.ChatCompletion.create(model=self.config.model, messages=messages, temperature=self.config.temperature)
                except Exception as e:
                    self.console.print(f"[red]✗ AI API Error: {e}[/red]")
                    return
            self.response_cache.set(prompt, cache_context, response)

        self.memory.add_conversation("assistant", response)
        json_data = self._extract_json_from_response(response)

        if not json_data:
            self.console.print(Panel(Markdown(response), title="🤖 Assistant", border_style="cyan"))
            return

        actions = self._parse_actions_from_json(json_data)
        if not actions:
            self.console.print("[red]✗ Could not parse any valid actions from the AI response.[/red]")
            self.debug_print(response, "INVALID_RESPONSE")
            return

        clarify_action = next((a for a in actions if a.type == ActionType.CLARIFY), None)
        if clarify_action and len(actions) == 1:
            self._handle_clarification(prompt, clarify_action)
        else:
            should_continue = self._execute_actions(actions)
            if should_continue:
                self.console.print("\n[bold cyan]...continuing task...[/bold cyan]")
                self.run(prompt, is_follow_up=True)

    def _handle_clarification(self, original_prompt: str, action: Action):
        """Asks the user the agent's question and re-runs the agent with the answer."""
        question = action.question or "How should I proceed?"
        self.console.print(f"\n[bold yellow]🤔 Assistant:[/bold yellow] {question}")
        answer = Prompt.ask("[bold green]Your reply[/bold green]")

        self.memory.add_conversation("assistant", action.question)
        self.memory.add_conversation("user", answer)
        
        self.run(original_prompt, is_follow_up=True)

    def _execute_actions(self, actions: List[Action], dry_run: bool = False) -> bool:
        """Displays the planned actions, executes them, and returns if the agent should continue."""
        self.console.print("\n[bold]🤖 Agent has a plan:[/bold]")
        
        plan_table = Table(title="Execution Plan", show_header=True, header_style="bold magenta")
        plan_table.add_column("#", style="dim")
        plan_table.add_column("Action", style="cyan")
        plan_table.add_column("Target")
        plan_table.add_column("Details / Reason", style="yellow")

        for i, action in enumerate(actions, 1):
            details = action.reason or action.content or ""
            if isinstance(details, str) and len(details) > 150:
                details = details[:150] + "..."
            plan_table.add_row(str(i), action.type.value, action.target or "-", str(details))
        
        self.console.print(plan_table)

        if not dry_run and self.config.approval_mode != ApprovalMode.FULL_AUTO:
            if not Confirm.ask("\n[bold]Proceed with execution?[/bold]"):
                self.console.print("[red]Execution cancelled by user.[/red]")
                return False

        has_new_observations = False
        for i, action in enumerate(actions):
            self.console.print(f"\n[cyan]▶️ Executing action {i+1}/{len(actions)}: {action.type.value}[/cyan]")
            try:
                result, is_read_only = self._execute_single_action(action, dry_run)
                if result:
                    self.memory.add_conversation("observation", str(result))
                    has_new_observations = True
                if not is_read_only:
                    pass 
            except Exception as e:
                self.console.print(f"[red]✗ Action failed: {e}[/red]")
                self.debug_print(traceback.format_exc(), "ACTION_ERROR")
                if not Confirm.ask("[yellow]An error occurred. Continue with next actions?[/yellow]", default=True):
                    return False
        
        return has_new_observations

    def _execute_single_action(self, action: Action, dry_run: bool) -> Tuple[Optional[str], bool]:
        """Executes a single action and returns its result and if it was a read-only op."""
        if dry_run:
            self.console.print(f"[cyan][DRY RUN][/cyan] Would {action.type.value}: {action.target or ''}")
            return None, True

        op_map = {
            ActionType.CREATE: self._create_file, ActionType.EDIT: self._edit_file,
            ActionType.DELETE: self._delete_file, ActionType.SEARCH: self._search_files,
            ActionType.EXECUTE: self._execute_code, ActionType.GIT: self._git_operation,
            ActionType.READ: self._read_file, ActionType.VIEW: self._read_file,
            ActionType.LIST: self._list_files, ActionType.WEB_SEARCH: self._web_search,
        }

        is_read_only = action.type in [ActionType.READ, ActionType.VIEW, ActionType.SEARCH, ActionType.LIST, ActionType.WEB_SEARCH, ActionType.CLARIFY]
        
        if not is_read_only:
            if action.risk_level == "high" and self.config.approval_mode != ApprovalMode.FULL_AUTO:
                if not Confirm.ask(f"[bold red]⚠️ High-risk action: {action.type.value} {action.target}. Confirm?[/bold red]"):
                    self.console.print("[red]Execution of high-risk action skipped.[/red]")
                    return None, True

        before_state = {} if is_read_only else self._capture_before_state(action)
        
        result = None
        if action.type in op_map:
            result = op_map[action.type](action)
            if not is_read_only:
                self.action_history.record(action, before_state)
                self.memory.add_action(action)
                if self.config.git_integration and self.git.is_git_repo() and action.type in [ActionType.CREATE, ActionType.EDIT, ActionType.DELETE]:
                    self.git.auto_commit(f"Agent: {action.type.value} {action.target}")
        elif action.type != ActionType.CLARIFY:
             self.console.print(f"[yellow]Warning: Action type '{action.type.value}' is not implemented for direct execution.[/yellow]")
        
        return result, is_read_only

    def undo_last_action(self):
        """Reverts the last executed file system action."""
        if not self.action_history.can_undo():
            self.console.print("[yellow]Nothing to undo.[/yellow]")
            return

        last_record = self.action_history.get_last_action()
        action_dict, before_state = last_record["action"], last_record["before_state"]
        
        try:
            action_dict['type'] = ActionType(action_dict['type'])
            action_params = {k: v for k, v in action_dict.items() if k in Action.__annotations__}
            action = Action(**action_params)
        except (ValueError, TypeError) as e:
            self.console.print(f"[red]✗ Failed to parse action for undo: {e}[/red]")
            self.debug_print(last_record, "UNDO_ERROR")
            return

        self.console.print(f"[cyan]Undoing: {action.type.value} on {action.target}[/cyan]")
        try:
            path = Path(action.target)
            if action.type == ActionType.CREATE:
                if path.is_file(): path.unlink(missing_ok=True)
                elif path.is_dir(): shutil.rmtree(path, ignore_errors=True)
            elif action.type == ActionType.EDIT and before_state.get("exists"):
                path.write_text(before_state["content"], encoding="utf-8")
            elif action.type == ActionType.DELETE and before_state.get("exists"):
                if before_state.get("is_dir"):
                    path.mkdir(exist_ok=True, parents=True)
                else:
                    path.write_text(before_state.get("content", ""), encoding="utf-8")
            
            self.action_history.undo_stack.pop()
            self.action_history.redo_stack.append(last_record)
            self.action_history._save_history()
            self.console.print("[green]✓ Undo successful.[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ Undo failed: {e}[/red]")
            self.debug_print(traceback.format_exc(), "UNDO_EXECUTION_ERROR")

    def _web_search(self, action: Action) -> Optional[str]:
        """Performs a web search and returns the results as a string."""
        query = action.query or action.content
        self.console.print(f"[cyan]🌐 Performing web search for:[/cyan] '{query}'")
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=3)]
                if results:
                    search_summary = ""
                    for i, res in enumerate(results, 1):
                        search_summary += f"### Result {i}\n**Title:** {res['title']}\n**Snippet:** {res['body']}\n**URL:** {res['href']}\n\n"
                    self.console.print(Panel(search_summary, title=f"Search Results for '{query}'", border_style="blue"))
                    return search_summary
                else:
                    self.console.print("[yellow]No web search results found.[/yellow]")
                    return "No web search results found."
        except ImportError:
            self.console.print("[red]Web search requires 'duckduckgo-search'. Please run: `pip install duckduckgo-search`[/red]")
        except Exception as e:
            self.console.print(f"[red]An error occurred during web search: {e}[/red]")
        return None

    def _extract_json_from_response(self, response: str) -> Optional[List[Dict]]:
        """Extracts the JSON array from the agent's raw response string."""
        match = re.search(r'```json\s*(\[.*\])\s*```|(\[.*\])', response, re.DOTALL)
        if not match:
            self.debug_print(response, "NO_JSON_FOUND")
            return None
        
        json_str = match.group(1) or match.group(2)
        if not json_str:
            return None
            
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.debug_print(f"JSON Decode Error: {e}\nAttempting to fix JSON: {json_str}", "JSON_ERROR")
            json_str_fixed = re.sub(r',\s*([\}\]])', r'\1', json_str)
            try:
                return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                self.console.print("[red]✗ Failed to parse AI response as valid JSON after attempting fix.[/red]")
                return None

    def _parse_actions_from_json(self, data: List[Dict]) -> List[Action]:
        """Parses a list of dictionaries into a list of Action objects."""
        actions = []
        if not isinstance(data, list):
            self.debug_print(f"Expected a list for actions, got {type(data)}", "PARSE_ERROR")
            return []
            
        for item in data:
            if isinstance(item, dict) and "type" in item:
                try:
                    action_data = item.copy()
                    action_type = ActionType(action_data.pop("type"))
                    valid_keys = Action.__annotations__.keys()
                    filtered_data = {k: v for k, v in action_data.items() if k in valid_keys}
                    actions.append(Action(type=action_type, **filtered_data))
                except (ValueError, TypeError) as e:
                    self.console.print(f"[yellow]Warning: Skipping invalid action item: {item}. Error: {e}[/yellow]")
        return actions
        
    def _capture_before_state(self, action: Action) -> Dict[str, Any]:
        """Captures the state of a file or directory before a modification."""
        state = {"exists": False}
        if not action.target: return state
        path = Path(action.target)
        if path.exists():
            state["exists"] = True
            if path.is_file():
                try:
                    state["content"] = path.read_text(encoding="utf-8")
                except (IOError, UnicodeDecodeError):
                    state["content"] = None
            elif path.is_dir():
                state["is_dir"] = True
        return state

    def _list_files(self, action: Action) -> Optional[str]:
        target = Path(action.target or ".")
        if not target.is_dir():
            self.console.print(f"[red]✗ Not a directory: {target}[/red]")
            return f"Error: Not a directory: {target}"
        
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Icon")
        table.add_column("Name")
        output_str = f"Contents of {target.resolve()}:\n"
        try:
            items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for item in items:
                if item.name.startswith('.'): continue
                icon = "📁" if item.is_dir() else "📄"
                style = "bright_cyan" if item.is_dir() else "white"
                table.add_row(icon, f"[{style}]{item.name}[/]")
                output_str += f"- {'d' if item.is_dir() else 'f'} {item.name}\n"
            self.console.print(f"\n[bold]Contents of {target.resolve()}:[/bold]")
            self.console.print(table)
            return output_str
        except PermissionError:
            self.console.print(f"[red]✗ Permission denied to read directory: {target}[/red]")
            return f"Error: Permission denied to read directory: {target}"

    def _read_file(self, action: Action) -> Optional[str]:
        path = Path(action.target)
        if not path.is_file():
            self.console.print(f"[red]✗ File not found: {path}[/red]")
            return f"Error: File not found: {path}"
        try:
            content = path.read_text('utf-8')
            try:
                lexer_name = get_lexer_for_filename(path.name, stripall=True)
            except ClassNotFound:
                lexer_name = "text"
            syntax = Syntax(content, lexer_name, theme=self.config.theme, line_numbers=True)
            self.console.print(Panel(syntax, title=f"📄 {path.name}", border_style="green", expand=False))
            return f"Content of {path.name}:\n```\n{content}\n```"
        except Exception as e:
            self.console.print(f"[red]✗ Could not read file {path}: {e}[/red]")
            return f"Error: Could not read file {path}: {e}"

    def _git_operation(self, action: Action) -> Optional[str]:
        op = action.content
        if op == "status":
            out = self.git.get_status() or "Working directory is clean."
            self.console.print(f"[green]{out}[/green]")
            return out
        elif op == "diff":
            out = self.git.get_diff() or "No changes to show."
            self.console.print(Syntax(out, "diff", theme=self.config.theme))
            return out
        elif op == "commit":
            msg = action.reason or Prompt.ask("[bold]Commit message[/bold]")
            if not msg:
                self.console.print("[yellow]Commit aborted by user.[/yellow]")
                return "Commit aborted."
            ok, out = self.git.commit_changes(msg, auto_add=True)
            self.console.print(f"[green]{out}[/green]" if ok else f"[red]Commit Failed:\n{out}[/red]")
            return out
        return None

    def _create_file(self, action: Action) -> None:
        path = Path(action.target)
        if path.exists() and not Confirm.ask(f"[yellow]⚠️ File '{path}' already exists. Overwrite?[/yellow]"):
             self.console.print("[red]Operation cancelled.[/red]")
             return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(action.content or "", 'utf-8')
        self.console.print(f"[green]✓ Created {path}[/green]")
    
    def _edit_file(self, action: Action) -> None:
        path = Path(action.target)
        if not path.is_file():
            self.console.print(f"[red]✗ Not a file: {path}[/red]")
            return
        
        try:
            original = path.read_text('utf-8')
            if action.line_start is not None and action.line_end is not None:
                lines = original.splitlines()
                start = action.line_start - 1
                end = action.line_end
                new_lines = action.content.splitlines() if action.content else []
                final_lines = lines[:start] + new_lines + lines[end:]
                new_content = "\n".join(final_lines)
            else:
                new_content = action.content or ""

            if original != new_content:
                path.write_text(new_content, 'utf-8')
                self.console.print(f"[green]✓ Edited {path}[/green]")
            else:
                self.console.print(f"[yellow]No changes made to {path}[/yellow]")
        except Exception as e:
            self.console.print(f"[red]✗ Failed to edit file {path}: {e}[/red]")

    def _delete_file(self, action: Action) -> None:
        path = Path(action.target)
        is_safe, reason = self.security.validate_file_operation("delete", str(path))
        if not is_safe:
            self.console.print(f"[red]✗ Security block: {reason}[/red]")
            return

        if not path.exists():
            self.console.print(f"[yellow]File or directory not found: {path}[/yellow]")
            return

        try:
            if path.is_file():
                path.unlink()
                self.console.print(f"[green]✓ Deleted file: {path}[/green]")
            elif path.is_dir():
                shutil.rmtree(path)
                self.console.print(f"[green]✓ Deleted directory: {path}[/green]")
        except OSError as e:
            self.console.print(f"[red]✗ Error deleting {path}: {e}[/red]")

    def _search_files(self, action: Action) -> Optional[str]:
        query = action.content
        target_dir = Path(action.target or ".")
        
        if not target_dir.is_dir():
            self.console.print(f"[red]✗ Search target is not a directory: {target_dir}[/red]")
            return f"Error: Search target is not a directory: {target_dir}"
            
        self.console.print(f"Searching for '{query}' in '{target_dir}'...")
        results_table = Table(title=f"Search Results for '{query}'")
        results_table.add_column("File Path", style="cyan")
        results_table.add_column("Line", style="magenta")
        results_table.add_column("Match", style="green")

        found_count = 0
        output_str = f"Search results for '{query}':\n"
        try:
            for path in target_dir.rglob("*"):
                if path.is_file() and not any(part.startswith('.') for part in path.parts):
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            for i, line in enumerate(f, 1):
                                if query.lower() in line.lower():
                                    found_count += 1
                                    results_table.add_row(str(path), str(i), line.strip())
                                    output_str += f"- {path}:{i}: {line.strip()}\n"
                                    if found_count >= 20: break
                    except Exception:
                        continue
                if found_count >= 20: break
        except PermissionError:
            self.console.print(f"[red]✗ Permission denied while searching in {target_dir}[/red]")
        
        if found_count > 0:
            self.console.print(results_table)
            return output_str
        else:
            self.console.print("[yellow]No results found.[/yellow]")
            return "No results found."

    def _execute_code(self, action: Action) -> Optional[str]:
        lang = action.method
        code = action.content
        
        self.console.print(Panel(Syntax(code, lang or "bash", theme=self.config.theme), title=f"Executing {lang} Code", border_style="yellow"))

        is_safe, reason = self.security.is_safe_command(code)
        if not is_safe:
            self.console.print(f"[red]✗ Security block: {reason}[/red]")
            return f"Execution blocked: {reason}"

        vulnerabilities = self.security.scan_code_for_vulnerabilities(code, lang)
        if vulnerabilities:
            self.console.print("[bold red]Vulnerability scan found issues:[/bold red]")
            for vuln in vulnerabilities:
                self.console.print(f"- {vuln}")
            if not Confirm.ask("[yellow]⚠️ Execute anyway?[/yellow]"):
                self.console.print("[red]Execution cancelled.[/red]")
                return "Execution cancelled by user due to vulnerabilities."

        returncode, stdout, stderr = self.security.sandbox_exec(code)
        
        output = ""
        if stdout:
            self.console.print(Panel(stdout, title="[green]Output (stdout)[/green]", border_style="green"))
            output += f"STDOUT:\n{stdout}\n"
        if stderr:
            self.console.print(Panel(stderr, title="[red]Errors (stderr)[/red]", border_style="red"))
            output += f"STDERR:\n{stderr}\n"
        
        self.console.print(f"Process finished with exit code: {returncode}")
        output += f"Exit Code: {returncode}"
        return output
