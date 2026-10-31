"""Agent and verification command execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import PipelineConfig, ProjectConfig
from .util import CommandResult, run_command, truncate, utc_now


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
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.binary = config.find_agent()

    def model_ref(self, model_id: str) -> str:
        if model_id.startswith("freeflow/"):
            return model_id
        return "freeflow/" + model_id

    def run(
        self,
        project: ProjectConfig,
        model_id: str,
        prompt: str,
        timeout: Optional[int] = None,
    ) -> AgentResult:
        command = [
            self.binary,
            "-p",
            "--model",
            self.model_ref(model_id),
            "--mode",
            "text",
            "--approve",
            "--no-session",
            "--no-mcp",
            "--thinking",
            self.config.thinking,
            prompt,
        ]
        started = utc_now()
        result = run_command(
            command,
            cwd=project.path,
            timeout=timeout or self.config.task_timeout,
            extra_env={
                "DEV_PIPELINE": "1",
                "PI_FREEFLOW_MODEL": model_id,
            },
        )
        return AgentResult(
            model=model_id,
            command=command,
            exit_code=result.exit_code,
            output=result.output,
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
