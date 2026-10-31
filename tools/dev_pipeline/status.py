"""Read and write pipeline state inside a project's ``AGENTS.md``.

The machine-readable block is canonical.  A generated task board and run log
sit next to it so humans (and coding agents) can see what the pipeline is doing
without parsing JSON.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .util import parse_utc, one_line, utc_now

STATE_BEGIN = "<!-- DEV-PIPELINE:STATE:BEGIN -->"
STATE_END = "<!-- DEV-PIPELINE:STATE:END -->"
BOARD_BEGIN = "<!-- DEV-PIPELINE:BOARD:BEGIN -->"
BOARD_END = "<!-- DEV-PIPELINE:BOARD:END -->"

STATE_RE = re.compile(
    re.escape(STATE_BEGIN) + r"(.*?)" + re.escape(STATE_END), re.DOTALL
)
BOARD_RE = re.compile(
    re.escape(BOARD_BEGIN) + r"(.*?)" + re.escape(BOARD_END), re.DOTALL
)

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
ACTIVE_STATUSES = {"pending", "in-progress", "blocked"}


class StatusError(RuntimeError):
    """Raised when AGENTS.md contains an unreadable state block."""


def _project_prefix(project: str) -> str:
    if project == "gpt4free":
        return "G4F"
    if project == "g4f.dev":
        return "G4FD"
    letters = re.sub(r"[^A-Za-z0-9]", "", project).upper()
    return (letters[:4] or "TASK")


def new_state(project: str) -> Dict[str, Any]:
    return {
        "schema": 1,
        "project": project,
        "updated": utc_now(),
        "tasks": [],
        "models": {},
        "runs": [],
    }


def _strip_fence(block: str) -> str:
    text = block.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _escape_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return " ".join(text.split())


def _task_sort_key(task: Dict[str, Any]):
    priority = str(task.get("priority", "P3")).upper()
    return (PRIORITY_ORDER.get(priority, 9), str(task.get("id", "")))


class StatusStore:
    """A project's AGENTS.md and the pipeline state it contains."""

    def __init__(self, project: str, path: Path):
        self.project = project
        self.path = Path(path)
        try:
            self.text = self.path.read_text(encoding="utf-8")
        except OSError:
            self.text = ""
        self.created = False
        self.state = self._parse()

    @property
    def has_state(self) -> bool:
        return not self.created

    def _parse(self) -> Dict[str, Any]:
        match = STATE_RE.search(self.text)
        if not match:
            self.created = True
            return new_state(self.project)
        block = _strip_fence(match.group(1))
        try:
            state = json.loads(block)
        except json.JSONDecodeError as exc:
            raise StatusError(
                f"{self.path}: invalid pipeline state JSON at line {exc.lineno}: {exc.msg}"
            ) from exc
        if not isinstance(state, dict):
            raise StatusError(f"{self.path}: pipeline state must be a JSON object")
        state.setdefault("schema", 1)
        state.setdefault("project", self.project)
        state.setdefault("tasks", [])
        state.setdefault("models", {})
        state.setdefault("runs", [])
        if not isinstance(state["tasks"], list):
            raise StatusError(f"{self.path}: 'tasks' must be a list")
        return state

    def reload(self) -> None:
        """Re-read from disk, e.g. after an agent has edited the file."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        previous = self.state
        self.text = text
        old_created = self.created
        try:
            self.state = self._parse()
        except StatusError:
            self.state = previous
            self.created = old_created

    # -- task operations -------------------------------------------------

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return [task for task in self.state.get("tasks", []) if isinstance(task, dict)]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        wanted = task_id.strip().upper()
        for task in self.tasks:
            if str(task.get("id", "")).upper() == wanted:
                return task
        return None

    def add_task(
        self,
        title: str,
        project: Optional[str] = None,
        priority: str = "P2",
        acceptance: str = "",
        task_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        project_name = project or self.project
        priority = str(priority).upper()
        if priority not in PRIORITY_ORDER:
            raise StatusError(f"priority must be one of {', '.join(PRIORITY_ORDER)}")
        if task_id:
            task_id = task_id.strip().upper()
            if self.get_task(task_id):
                raise StatusError(f"task {task_id} already exists")
        else:
            prefix = _project_prefix(project_name)
            index = 1
            existing = {str(task.get("id", "")).upper() for task in self.tasks}
            while f"{prefix}-{index:03d}" in existing:
                index += 1
            task_id = f"{prefix}-{index:03d}"
        task = {
            "id": task_id,
            "project": project_name,
            "priority": priority,
            "status": "pending",
            "title": title.strip(),
            "acceptance": acceptance.strip(),
            "created": utc_now(),
            "updated": utc_now(),
            "attempts": 0,
            "last_model": None,
            "notes": "",
        }
        if source:
            task["source"] = source
        self.state.setdefault("tasks", []).append(task)
        return task

    def select_task(
        self,
        task_id: Optional[str] = None,
        project: Optional[str] = None,
        retry_blocked: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if task_id:
            task = self.get_task(task_id)
            if task is None:
                raise StatusError(f"task {task_id} not found in {self.path}")
            return task
        candidates = [
            task
            for task in self.tasks
            if task.get("status") == "pending"
            or (retry_blocked and task.get("status") == "blocked")
        ]
        if project:
            candidates = [task for task in candidates if task.get("project", self.project) == project]
        candidates.sort(key=_task_sort_key)
        return candidates[0] if candidates else None

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        model: Optional[str] = None,
        increment_attempt: bool = False,
    ) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if task is None:
            raise StatusError(f"task {task_id} not found in {self.path}")
        if status:
            task["status"] = status
        if notes is not None:
            task["notes"] = notes
        if model is not None:
            task["last_model"] = model
        if increment_attempt:
            task["attempts"] = int(task.get("attempts") or 0) + 1
        task["updated"] = utc_now()
        return task

    def recover_stale(self, older_than_seconds: int = 7200) -> List[str]:
        """Move abandoned in-progress tasks back to pending."""
        recovered = []
        now = parse_utc(utc_now())
        for task in self.tasks:
            if task.get("status") != "in-progress":
                continue
            updated = parse_utc(str(task.get("updated", "")))
            if updated is None or now is None:
                continue
            if (now - updated).total_seconds() > older_than_seconds:
                task["status"] = "pending"
                task["notes"] = "Recovered after an interrupted pipeline run."
                task["updated"] = utc_now()
                recovered.append(str(task.get("id")))
        return recovered

    # -- model/run operations -------------------------------------------

    def record_models(
        self,
        working: Iterable[Dict[str, Any]],
        failed: Iterable[Dict[str, Any]],
        base_url: str,
        from_cache: bool = False,
    ) -> None:
        self.state["models"] = {
            "base_url": base_url,
            "checked_at": utc_now(),
            "from_cache": from_cache,
            "working": list(working),
            "failed": list(failed),
        }

    def record_run(self, run: Dict[str, Any]) -> None:
        runs = self.state.setdefault("runs", [])
        runs.append(run)
        del runs[:-40]

    # -- rendering/writing ----------------------------------------------

    def _render_state(self) -> str:
        payload = json.dumps(self.state, indent=2, ensure_ascii=False)
        return f"{STATE_BEGIN}\n```json\n{payload}\n```\n{STATE_END}"

    def _render_board(self) -> str:
        lines = [BOARD_BEGIN, "", "### Task board", ""]
        tasks = sorted(self.tasks, key=_task_sort_key)
        if tasks:
            lines.append("| ID | Project | Priority | Status | Task | Acceptance | Updated |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for task in tasks:
                lines.append(
                    "| {id} | {project} | {priority} | {status} | {title} | {acceptance} | {updated} |".format(
                        id=_escape_cell(task.get("id")),
                        project=_escape_cell(task.get("project", self.project)),
                        priority=_escape_cell(task.get("priority", "P2")),
                        status=_escape_cell(task.get("status", "pending")),
                        title=_escape_cell(task.get("title")),
                        acceptance=_escape_cell(task.get("acceptance")),
                        updated=_escape_cell(task.get("updated")),
                    )
                )
        else:
            lines.append("_No tasks queued. Add one with `bin/dev-pipeline add-task`._")
        lines.extend(["", "### Free model health", ""])
        models = self.state.get("models") or {}
        working = models.get("working") or []
        failed = models.get("failed") or []
        if working or failed:
            lines.append(
                "Checked `{when}` at `{base}`.".format(
                    when=_escape_cell(models.get("checked_at", "unknown")),
                    base=_escape_cell(models.get("base_url", "unknown")),
                )
            )
            lines.append("")
            lines.append("| Model | Status | Latency | Detail |")
            lines.append("| --- | --- | --- | --- |")
            for item in working:
                lines.append(
                    "| {id} | working | {latency}s | {detail} |".format(
                        id=_escape_cell(item.get("id")),
                        latency=_escape_cell(item.get("latency_s", 0)),
                        detail=_escape_cell(one_line(str(item.get("detail", "")), 120)),
                    )
                )
            for item in failed[:10]:
                lines.append(
                    "| {id} | failed | - | {detail} |".format(
                        id=_escape_cell(item.get("id")),
                        detail=_escape_cell(one_line(str(item.get("detail", "")), 120)),
                    )
                )
        else:
            lines.append("_No model health snapshot yet. Run `bin/dev-pipeline models --refresh`._")
        lines.extend(["", "### Recent runs", ""])
        runs = self.state.get("runs", [])
        recent = list(runs)[-12:][::-1]
        if recent:
            lines.append("| When | Task | Project | Model | Result | Detail |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for run in recent:
                lines.append(
                    "| {when} | {task} | {project} | {model} | {result} | {detail} |".format(
                        when=_escape_cell(run.get("finished") or run.get("started")),
                        task=_escape_cell(run.get("task")),
                        project=_escape_cell(run.get("project", self.project)),
                        model=_escape_cell(run.get("model") or "-"),
                        result=_escape_cell(run.get("result")),
                        detail=_escape_cell(one_line(str(run.get("detail", "")), 220)),
                    )
                )
        else:
            lines.append("_No pipeline runs recorded yet._")
        lines.extend(["", BOARD_END])
        return "\n".join(lines)

    def _replace(self, text: str, pattern: re.Pattern, block: str) -> str:
        match = pattern.search(text)
        if match:
            return text[: match.start()] + block + text[match.end() :]
        return text.rstrip() + "\n\n" + block + "\n"

    def save(self) -> None:
        self.state["updated"] = utc_now()
        state_block = self._render_state()
        board_block = self._render_board()

        text = self.text or ""
        if "## Development pipeline" not in text:
            text = text.rstrip() + "\n\n## Development pipeline\n\n" if text.strip() else "## Development pipeline\n\n"
        text = self._replace(text, STATE_RE, state_block)
        text = self._replace(text, BOARD_RE, board_block)
        if not text.endswith("\n"):
            text += "\n"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)
        self.text = text
        self.created = False

    def summary_rows(self) -> List[List[str]]:
        return [
            [
                str(task.get("id", "")),
                str(task.get("project", self.project)),
                str(task.get("priority", "P2")),
                str(task.get("status", "pending")),
                str(task.get("title", "")),
                str(task.get("updated", "")),
            ]
            for task in sorted(self.tasks, key=_task_sort_key)
        ]
