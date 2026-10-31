"""Configuration for the automatic development pipeline.

The pipeline is intentionally dependency-free (standard library only) so it can
run inside the slim gpt4free image before any optional extras are installed.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Models are tried in this order.  A short alias (as documented by
# pi-freeflow) or a full upstream id both work; matching is substring based.
DEFAULT_PREFERRED_MODELS: List[str] = [
    "deepseek-v4.1-flash-free",
    "glm-5.3-flash-free",
    "mimo-v2.6-flash-free",
    "nemotron-3.5-lightning-free",
    "big-pickle",
    "north-mini-code",
    "qwen3.6-27b",
    "qwen3.5-397b",
    "gpt-oss-120b",
    "mistral-small-3.2",
    "llama-3.3-70b",
    "laguna-s-2.1",
    "xing4.0",
    "flint-40b-free",
    "dots-3-note-preview",
    "nemotron-3-ultra-550b",
]

DEFAULT_PROJECTS: Dict[str, Dict[str, Any]] = {
    "gpt4free": {
        "path": ".",
        "agents_file": "AGENTS.md",
        "checks": [["python3", "-m", "compileall", "-q", "g4f"]],
    },
    "g4f.dev": {
        "path": "../g4f.dev",
        "agents_file": "AGENTS.md",
        # The website has no single test runner.  git diff --check is added
        # automatically for git checkouts; add project-specific checks here.
        "checks": [],
    },
}


class ConfigError(RuntimeError):
    """Raised when dev-pipeline.json is malformed."""


def _as_command(value: Any, project_name: str) -> List[str]:
    if isinstance(value, str):
        command = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        command = [str(part) for part in value]
    else:
        raise ConfigError(
            f"project {project_name!r}: checks must be strings or lists, got {type(value).__name__}"
        )
    if not command:
        raise ConfigError(f"project {project_name!r}: empty check command")
    return command


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass
class ProjectConfig:
    name: str
    path: Path
    agents_file: str = "AGENTS.md"
    checks: List[List[str]] = field(default_factory=list)
    enabled: bool = True

    @property
    def agents_path(self) -> Path:
        return self.path / self.agents_file

    @property
    def exists(self) -> bool:
        return self.enabled and self.path.is_dir() and self.agents_path.is_file()

    def effective_checks(self) -> List[List[str]]:
        """Return configured checks plus a whitespace check for git repos."""
        checks = [list(check) for check in self.checks]
        git_dir = self.path / ".git"
        git_check = ["git", "diff", "--check"]
        if git_dir.exists() and git_check not in checks:
            checks.append(git_check)
        return checks

    def display_path(self, root: Path) -> str:
        try:
            return str(self.path.relative_to(root))
        except ValueError:
            return str(self.path)


@dataclass
class PipelineConfig:
    root: Path
    projects: Dict[str, ProjectConfig]
    preferred_models: List[str] = field(default_factory=lambda: list(DEFAULT_PREFERRED_MODELS))
    base_url: Optional[str] = None
    probe_limit: int = 10
    min_models: int = 2
    model_cache_ttl: int = 300
    max_model_attempts: int = 3
    repair_attempts: int = 1
    agent: str = "pi"
    thinking: str = "high"
    task_timeout: int = 1800
    check_timeout: int = 300
    commit: bool = False
    push: bool = False
    enabled_projects: Optional[List[str]] = None

    @property
    def state_dir(self) -> Path:
        return self.root / ".dev-pipeline"

    @property
    def cache_path(self) -> Path:
        return self.state_dir / "models.json"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "run.lock"

    def available_projects(self) -> Dict[str, ProjectConfig]:
        selected = set(self.enabled_projects or [])
        result: Dict[str, ProjectConfig] = {}
        for name, project in self.projects.items():
            if selected and name not in selected:
                continue
            if project.exists:
                result[name] = project
        return result

    def project(self, name: str) -> ProjectConfig:
        try:
            project = self.projects[name]
        except KeyError as exc:
            raise ConfigError(
                f"unknown project {name!r}; configured projects: {', '.join(self.projects)}"
            ) from exc
        if not project.path.is_dir():
            raise ConfigError(f"project {name!r} path does not exist: {project.path}")
        return project

    def find_agent(self) -> str:
        """Return the agent executable to use."""
        configured = os.environ.get("DEV_PIPELINE_AGENT", self.agent)
        if os.path.sep in configured or configured.startswith("~"):
            path = Path(os.path.expanduser(configured))
            if path.is_file():
                return str(path)
            raise ConfigError(f"agent executable not found: {path}")
        found = shutil.which(configured)
        if found:
            return found
        for fallback in ("pi", "omp"):
            found = shutil.which(fallback)
            if found:
                return found
        raise ConfigError(
            "no agent executable found. Install Pi/OMP and the pi-freeflow extension:\n"
            "  curl -fsSL https://pi.dev/install.sh | sh\n"
            "  pi install npm:pi-freeflow"
        )


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}:{exc.lineno}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def load_config(root: Path, config_path: Optional[Path] = None) -> PipelineConfig:
    """Load ``dev-pipeline.json`` and merge it with built-in defaults."""
    root = root.resolve()
    if config_path is None:
        config_path = root / "dev-pipeline.json"
        data: Dict[str, Any] = _read_json(config_path) if config_path.is_file() else {}
    else:
        config_path = config_path.resolve()
        data = _read_json(config_path)
        # Relative paths in an external config are resolved from that file.
        if config_path.parent != root:
            root = config_path.parent

    raw_projects = data.get("projects")
    if raw_projects is None:
        raw_projects = DEFAULT_PROJECTS
    if not isinstance(raw_projects, dict):
        raise ConfigError("'projects' must be a JSON object")

    projects: Dict[str, ProjectConfig] = {}
    for name, raw in raw_projects.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"project {name!r} must be a JSON object")
        default = DEFAULT_PROJECTS.get(str(name), {})
        path_value = raw.get("path", default.get("path"))
        if not path_value:
            raise ConfigError(f"project {name!r} is missing 'path'")
        checks_raw = raw.get("checks", default.get("checks", []))
        if checks_raw is None:
            checks_raw = []
        if not isinstance(checks_raw, list):
            raise ConfigError(f"project {name!r}: 'checks' must be a list")
        checks = [_as_command(check, str(name)) for check in checks_raw]
        projects[str(name)] = ProjectConfig(
            name=str(name),
            path=_resolve_path(root, str(path_value)),
            agents_file=str(raw.get("agents_file", default.get("agents_file", "AGENTS.md"))),
            checks=checks,
            enabled=bool(raw.get("enabled", True)),
        )

    preferred = data.get("preferred_models", DEFAULT_PREFERRED_MODELS)
    if not isinstance(preferred, list) or not all(isinstance(item, str) for item in preferred):
        raise ConfigError("'preferred_models' must be a list of strings")

    enabled_projects = data.get("enabled_projects")
    if enabled_projects is not None and (
        not isinstance(enabled_projects, list)
        or not all(isinstance(item, str) for item in enabled_projects)
    ):
        raise ConfigError("'enabled_projects' must be a list of strings")

    def integer(key: str, default: int, minimum: int = 0) -> int:
        value = data.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key!r} must be an integer") from exc
        if value < minimum:
            raise ConfigError(f"{key!r} must be >= {minimum}")
        return value

    return PipelineConfig(
        root=root,
        projects=projects,
        preferred_models=list(preferred),
        base_url=data.get("base_url"),
        probe_limit=integer("probe_limit", 10, 1),
        min_models=integer("min_models", 2, 1),
        model_cache_ttl=integer("model_cache_ttl", 300, 0),
        max_model_attempts=integer("max_model_attempts", 3, 1),
        repair_attempts=integer("repair_attempts", 1, 0),
        agent=str(data.get("agent", "pi")),
        thinking=str(data.get("thinking", "high")),
        task_timeout=integer("task_timeout", 1800, 1),
        check_timeout=integer("check_timeout", 300, 1),
        commit=bool(data.get("commit", False)),
        push=bool(data.get("push", False)),
        enabled_projects=list(enabled_projects) if enabled_projects else None,
    )
