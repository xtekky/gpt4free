"""Agent and verification command execution."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import PipelineConfig, ProjectConfig
from .util import (
    CommandResult,
    run_command,
    run_command_streaming,
    truncate,
    utc_now,
)


TOOL_ARG_KEYS = (
    "command",
    "cmd",
    "path",
    "file_path",
    "filePath",
    "pattern",
    "query",
    "url",
)


def _summarise_tool_args(args: Any) -> str:
    """Return a short one-line description of a tool call's arguments."""
    if isinstance(args, dict):
        for key in TOOL_ARG_KEYS:
            value = args.get(key)
            if value:
                return " ".join(str(value).split())[:140]
        try:
            return " ".join(json.dumps(args, ensure_ascii=False).split())[:140]
        except (TypeError, ValueError):
            return ""
    if isinstance(args, str):
        return " ".join(args.split())[:140]
    return ""


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return ""


@dataclass
class AgentResult:
    model: str
    command: List[str]
    exit_code: int
    output: str
    duration: float
    started: str = field(default_factory=utc_now)
    finished: str = field(default_factory=utc_now)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class AgentRunner:
    # Models from these providers are already fully qualified; everything else
    # is assumed to come from the pi-freeflow proxy.
    PROVIDER_PREFIXES = ("freeflow/", "pollinations/")

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.binary = config.find_agent()

    def model_ref(self, model_id: str) -> str:
        if model_id.startswith(self.PROVIDER_PREFIXES):
            return model_id
        return "freeflow/" + model_id

    def run(
        self,
        project: ProjectConfig,
        model_id: str,
        prompt: str,
        timeout: Optional[int] = None,
        progress: Optional[Callable[[str], None]] = None,
        heartbeat: int = 30,
    ) -> AgentResult:
        model_ref = self.model_ref(model_id)
        command = [
            self.binary,
            "-p",
            "--model",
            model_ref,
            "--mode",
            "json",
            "--approve",
            "--no-session",
            "--no-mcp",
            "--thinking",
            self.config.thinking,
            prompt,
        ]
        started = utc_now()
        limit = timeout or self.config.task_timeout
        stop = threading.Event()
        final_parts: List[str] = []
        event_log: List[str] = []

        def emit(message: str) -> None:
            if progress:
                progress(message)

        def handle_line(line: str) -> None:
            text = line.strip()
            if not text:
                return
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                # Diagnostics on stdout are rare in JSON mode; surface them.
                if len(text) <= 200:
                    emit("  " + text)
                return
            if not isinstance(event, dict):
                return
            kind = event.get("type")
            if kind == "tool_execution_start":
                name = str(event.get("toolName") or "tool")
                summary = _summarise_tool_args(event.get("args"))
                message = f"  -> {name}: {summary}".rstrip()
                event_log.append(message)
                emit(message)
            elif kind == "tool_execution_end":
                name = str(event.get("toolName") or "tool")
                duration = event.get("durationMs")
                status = "error" if event.get("isError") else "ok"
                timing = f", {duration / 1000:.1f}s" if isinstance(duration, (int, float)) else ""
                message = f"  <- {name} ({status}{timing})"
                event_log.append(message)
                emit(message)
            elif kind == "message_end":
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") == "assistant":
                    text = _message_text(message)
                    if text:
                        final_parts.append(text)
            elif kind == "agent_settled" and event.get("aborted"):
                emit("  agent aborted")

        def beat() -> None:
            elapsed = 0
            while not stop.wait(heartbeat):
                elapsed += heartbeat
                emit(
                    f"  ... {model_ref} still running ({elapsed}s, timeout {limit}s)"
                )

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            result = run_command_streaming(
                command,
                cwd=project.path,
                timeout=limit,
                extra_env={
                    "DEV_PIPELINE": "1",
                    "PI_FREEFLOW_MODEL": model_id,
                },
                on_line=handle_line,
            )
        finally:
            stop.set()
            thread.join(timeout=1)

        output = "\n\n".join(part for part in final_parts if part).strip()
        if not output:
            output = "\n".join(event_log).strip() or result.output
        return AgentResult(
            model=model_id,
            command=command,
            exit_code=result.exit_code,
            output=truncate(output),
            duration=result.duration,
            started=started,
            finished=utc_now(),
        )


def run_checks(
    config: PipelineConfig,
    project: ProjectConfig,
    commands: Optional[Sequence[Sequence[str]]] = None,
) -> List[CommandResult]:
    checks = [list(command) for command in (commands if commands is not None else project.effective_checks())]
    results: List[CommandResult] = []
    for command in checks:
        results.append(
            run_command(
                command,
                cwd=project.path,
                timeout=config.check_timeout,
            )
        )
        # Stop after the first failure; later checks usually depend on it.
        if not results[-1].ok:
            break
    return results


def checks_ok(results: Sequence[CommandResult]) -> bool:
    return all(result.ok for result in results)


def format_check_results(results: Sequence[CommandResult], limit: int = 2500) -> str:
    if not results:
        return "no checks configured"
    parts = []
    for result in results:
        status = "ok" if result.ok else f"exit {result.exit_code}"
        parts.append(f"$ {' '.join(result.command)}\n[{status}]\n{truncate(result.output, limit)}")
    return "\n\n".join(parts)


def git_diff_stat(project: ProjectConfig) -> str:
    if not (project.path / ".git").exists():
        return ""
    result = run_command(
        ["git", "status", "--short"],
        cwd=project.path,
        timeout=30,
    )
    stat = run_command(
        ["git", "diff", "--stat"],
        cwd=project.path,
        timeout=30,
    )
    return truncate("\n".join(part for part in (result.output, stat.output) if part.strip()), 1500)


def maybe_commit(
    config: PipelineConfig,
    project: ProjectConfig,
    task: Dict,
    commit: bool,
    push: bool = False,
) -> Optional[str]:
    if not commit or not (project.path / ".git").exists():
        return None
    add = run_command(["git", "add", "-A"], cwd=project.path, timeout=60)
    if not add.ok:
        return "git add failed: " + add.output
    status = run_command(["git", "status", "--porcelain"], cwd=project.path, timeout=30)
    if not status.output.strip():
        return None
    message = f"pipeline: {task.get('id')} {task.get('title', '')}".strip()
    result = run_command(
        ["git", "commit", "-m", message],
        cwd=project.path,
        timeout=120,
    )
    if not result.ok:
        return "git commit failed: " + result.output
    if push:
        pushed = run_command(["git", "push"], cwd=project.path, timeout=180)
        if not pushed.ok:
            return "git push failed: " + pushed.output
    return "committed: " + message
