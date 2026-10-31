"""Orchestration: pick a task, ask a free model to do it, verify, record."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig, ProjectConfig
from .freeflow import FreeflowClient, FreeflowError, select_working_models
from .runner import (
    AgentResult,
    AgentRunner,
    checks_ok,
    format_check_results,
    git_diff_stat,
    maybe_commit,
    run_checks,
)
from .status import StatusStore, StatusError
from .util import one_line, truncate, utc_now

TODO_RE = re.compile(r"\b(TODO|FIXME)\b[:\s-]*(.{0,140})", re.IGNORECASE)
SKIP_DIRS = {
    ".git",
    ".dev-pipeline",
    ".pi",
    "__pycache__",
    "node_modules",
    "generated_media",
    ".venv",
    "venv",
    "dist",
    "build",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".sh",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".go",
    ".rs",
    ".html",
    ".css",
}


def load_stores(config: PipelineConfig) -> Dict[str, StatusStore]:
    stores: Dict[str, StatusStore] = {}
    for name, project in config.available_projects().items():
        stores[name] = StatusStore(name, project.agents_path)
    return stores


def find_task_project(
    config: PipelineConfig,
    stores: Dict[str, StatusStore],
    task: Dict[str, Any],
    fallback: str,
) -> str:
    project_name = str(task.get("project") or fallback)
    if project_name in stores:
        return project_name
    # A task may name a project that is not checked out yet.
    if project_name in config.projects:
        return project_name
    return fallback


def select_next_task(
    config: PipelineConfig,
    stores: Dict[str, StatusStore],
    task_id: Optional[str] = None,
    project: Optional[str] = None,
    retry_blocked: bool = False,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if task_id:
        wanted = task_id.upper()
        for name, store in stores.items():
            task = store.get_task(wanted)
            if task is not None:
                return name, task
        raise StatusError(f"task {task_id} not found in any available AGENTS.md")

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for name, store in stores.items():
        if project and name != project:
            continue
        task = store.select_task(
            project=project or name,
            retry_blocked=retry_blocked,
        )
        if task is not None:
            candidates.append((name, task))

    def key(item: Tuple[str, Dict[str, Any]]):
        task = item[1]
        priority = str(task.get("priority", "P3")).upper()
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        return (order.get(priority, 9), str(task.get("id", "")))

    candidates.sort(key=key)
    return candidates[0] if candidates else None


def build_prompt(
    project: ProjectConfig,
    task: Dict[str, Any],
    checks: Sequence[Sequence[str]],
    repair_context: str = "",
) -> str:
    check_lines = "\n".join("- `" + " ".join(command) + "`" for command in checks)
    if not check_lines:
        check_lines = "- No project checks are configured. Inspect the diff manually."

    parts = [
        "You are the autonomous implementer in the {name} repository.".format(name=project.name),
        "",
        "Repository: {path}".format(path=project.path),
        "Task: {id} ({priority})".format(
            id=task.get("id"), priority=task.get("priority", "P2")
        ),
        "Title: {title}".format(title=task.get("title", "")),
        "Acceptance criteria: {acceptance}".format(
            acceptance=task.get("acceptance") or "The task is implemented and checks pass."
        ),
        "",
        "Rules:",
        "- Read AGENTS.md first and follow it.",
        "- Work only inside this repository.",
        "- Do not edit the DEV-PIPELINE:STATE or DEV-PIPELINE:BOARD blocks in AGENTS.md; the pipeline owns them.",
        "- Make the smallest correct change. Add or update tests when behaviour changes.",
        "- Do not create commits, push, or install paid services.",
        "- Run these checks when you are done:",
        check_lines,
        "",
        "When finished, reply with a concise summary of what changed, files touched,",
        "checks you ran, and any remaining risk. Do not ask for confirmation.",
    ]
    if repair_context:
        parts.extend(
            [
                "",
                "The previous attempt did not pass verification. Fix only what is needed.",
                "Verification output:",
                truncate(repair_context, 6000),
            ]
        )
    return "\n".join(parts)


def discover_todos(
    config: PipelineConfig,
    stores: Dict[str, StatusStore],
    project_name: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Turn TODO/FIXME comments into backlog tasks (idempotent by source)."""
    project = config.project(project_name)
    store = stores.get(project_name)
    if store is None:
        return []
    existing_sources = {
        str(task.get("source")) for task in store.tasks if task.get("source")
    }
    created: List[Dict[str, Any]] = []
    found = 0
    for root, dirs, files in os.walk(project.path):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".git")]
        for filename in files:
            if found >= limit:
                break
            path = Path(root) / filename
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                relative = path.relative_to(project.path).as_posix()
            except ValueError:
                relative = path.name
            if relative in {"AGENTS.md", "docs/dev-pipeline.md"} or relative.startswith(
                "tools/dev_pipeline/"
            ):
                # Do not turn the pipeline's own documentation/regex into tasks.
                continue
            try:
                if path.stat().st_size > 500_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if found >= limit:
                    break
                match = TODO_RE.search(line)
                if not match:
                    continue
                source = f"{path.relative_to(project.path)}:{number}"
                if source in existing_sources:
                    continue
                description = match.group(2).strip() or line.strip()
                task = store.add_task(
                    title=f"Resolve {match.group(1).upper()} in {path.name}: {description}",
                    project=project_name,
                    priority="P3",
                    acceptance=f"Address `{source}` without breaking the configured checks.",
                    source=source,
                )
                created.append(task)
                existing_sources.add(source)
                found += 1
        if found >= limit:
            break
    if created:
        store.save()
    return created


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        reporter=print,
        refresh_models: bool = False,
        explicit_model: Optional[str] = None,
        commit: Optional[bool] = None,
        push: Optional[bool] = None,
    ):
        self.config = config
        self.reporter = reporter
        self.refresh_models = refresh_models
        self.explicit_model = explicit_model
        self.commit = config.commit if commit is None else commit
        self.push = config.push if push is None else push
        self.stores = load_stores(config)
        self.runner: Optional[AgentRunner] = None

    def log(self, message: str = "") -> None:
        self.reporter(message)

    def _runner(self) -> AgentRunner:
        if self.runner is None:
            self.runner = AgentRunner(self.config)
        return self.runner

    def recover_stale(self) -> None:
        for name, store in self.stores.items():
            recovered = store.recover_stale()
            if recovered:
                store.save()
                self.log(f"[{name}] recovered stale in-progress tasks: {', '.join(recovered)}")

    def models(
        self,
        refresh: bool = False,
        explore: bool = False,
        project_name: Optional[str] = None,
    ) -> Tuple[FreeflowClient, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        if project_name and project_name in self.stores:
            name = project_name
        elif project_name and project_name in self.config.projects:
            name = project_name
        else:
            name = next(iter(self.stores), None) or next(iter(self.config.projects))
        project = self.config.project(name)
        return select_working_models(
            self.config,
            project,
            refresh=refresh or self.refresh_models,
            explicit_model=self.explicit_model,
            explore=explore,
        )

    def status(self, refresh_models: bool = False) -> None:
        for name, store in self.stores.items():
            project = self.config.projects[name]
            self.log(f"== {name} ({project.path}) ==")
            rows = store.summary_rows()
            if not rows:
                self.log("  no tasks")
            for row in rows:
                self.log(f"  {row[0]:<12} {row[2]:<3} {row[3]:<11} {row[4]}")
            health = store.state.get("models") or {}
            working = health.get("working") or []
            if working:
                self.log(
                    "  models: "
                    + ", ".join(str(item.get("id")) for item in working[:5])
                )
            self.log("")

        if refresh_models:
            try:
                client, working, failed, cached = self.models(refresh=True)
                self.log(
                    f"freeflow: {client.base_url} — {len(working)} working, {len(failed)} failed "
                    f"({'cache' if cached else 'live probe'})"
                )
                for item in working:
                    self.log(
                        f"  ok   {item['id']}  {item.get('latency_s', 0):.2f}s"
                    )
                for item in failed[:10]:
                    self.log(f"  fail {item['id']}  {one_line(item.get('detail', ''), 120)}")
            except FreeflowError as exc:
                self.log(f"freeflow: ERROR {exc}")

    def run_once(
        self,
        task_id: Optional[str] = None,
        project: Optional[str] = None,
        dry_run: bool = False,
        retry_blocked: bool = False,
    ) -> bool:
        self.recover_stale()
        # Reload because another process or an agent may have touched the file.
        for store in self.stores.values():
            store.reload()

        selected = select_next_task(
            self.config,
            self.stores,
            task_id=task_id,
            project=project,
            retry_blocked=retry_blocked,
        )
        if selected is None:
            self.log("No pending task found. Use `bin/dev-pipeline add-task` or `plan`.")
            return False

        owner, task = selected
        task_project = str(task.get("project") or owner)
        if task_project != owner:
            if task_project in self.stores:
                owner = task_project
                store = self.stores[owner]
            else:
                store = self.stores[owner]
                store.update_task(
                    str(task.get("id")),
                    status="blocked",
                    notes=f"Project {task_project!r} is not available (missing checkout or AGENTS.md).",
                )
                store.save()
                self.log(
                    f"Task {task.get('id')} belongs to {task_project}, which is not available."
                )
                return False
        else:
            store = self.stores[owner]
        project_config = self.config.projects[owner]
        if not project_config.path.is_dir():
            self.log(f"Task {task.get('id')} belongs to {owner}, but {project_config.path} is missing.")
            return False

        checks = project_config.effective_checks()
        self.log(f"Task {task.get('id')}: {task.get('title')}")
        self.log(f"Project: {owner} ({project_config.path})")
        self.log(f"Priority: {task.get('priority')}  Attempts: {task.get('attempts', 0)}")

        freeflow_error = ""
        try:
            client, working, failed, from_cache = self.models(
                refresh=self.refresh_models, project_name=owner
            )
        except FreeflowError as exc:
            client = None
            working, failed, from_cache = [], [], False
            freeflow_error = str(exc)

        candidates = [str(item.get("id")) for item in working if item.get("id")]
        fallbacks = [model for model in self.config.fallback_models if model]
        if not candidates and not fallbacks:
            if freeflow_error:
                self.log(f"No working free model: {freeflow_error}")
            else:
                self.log("No working free model returned by the proxy.")
            if dry_run:
                return False
            store.update_task(
                str(task.get("id")),
                status="blocked",
                notes="No working freeflow model: "
                + one_line(freeflow_error or "empty proxy catalog", 300),
            )
            store.record_run(
                {
                    "task": task.get("id"),
                    "project": owner,
                    "model": None,
                    "result": "blocked",
                    "started": utc_now(),
                    "finished": utc_now(),
                    "detail": "no working free model: "
                    + one_line(freeflow_error or "empty proxy catalog", 250),
                }
            )
            store.save()
            return False

        if client is not None:
            self.log(
                f"Freeflow: {client.base_url} ({len(working)} working, "
                f"{'cached' if from_cache else 'live'})"
            )
        elif freeflow_error:
            self.log(f"Freeflow unavailable: {one_line(freeflow_error, 200)}")
        if candidates:
            self.log("Models: " + ", ".join(candidates[: self.config.max_model_attempts]))
        if fallbacks:
            self.log("Fallbacks: " + ", ".join(fallbacks))

        if dry_run:
            self.log("Checks: " + (", ".join(" ".join(c) for c in checks) or "(none)"))
            self.log("Dry run only; AGENTS.md was not modified.")
            return True

        first_model = candidates[0] if candidates else fallbacks[0]
        store.update_task(
            str(task.get("id")),
            status="in-progress",
            model=first_model,
            increment_attempt=True,
        )
        if client is not None:
            store.record_models(working, failed, client.base_url, from_cache)
        store.save()

        started = utc_now()
        # Fail before spending model time if the repository is already broken.
        baseline = run_checks(self.config, project_config, checks)
        if checks and not checks_ok(baseline):
            detail = "baseline checks failed:\n" + format_check_results(baseline)
            store.update_task(
                str(task.get("id")),
                status="blocked",
                notes="Baseline checks failed; no model was invoked.",
            )
            store.record_run(
                {
                    "task": task.get("id"),
                    "project": owner,
                    "model": None,
                    "result": "blocked",
                    "started": started,
                    "finished": utc_now(),
                    "detail": one_line(detail, 300),
                }
            )
            store.save()
            self.log(detail)
            return False

        attempt_models = list(candidates[: self.config.max_model_attempts])
        for fallback in fallbacks:
            if fallback not in attempt_models:
                attempt_models.append(fallback)
        last_detail = ""
        used_model = None
        success = False
        agent_exit = None

        try:
            for index, model_id in enumerate(attempt_models):
                if index:
                    self.log(f"Retrying with fallback model: {model_id}")
                used_model = model_id
                prompt = build_prompt(project_config, task, checks)
                if last_detail:
                    prompt += "\n\nPrevious attempt output:\n" + truncate(last_detail, 3000)
                self.log(
                    f"Running {self.runner_binary()} with {model_id} ... "
                    f"(streaming progress, timeout {self.config.task_timeout}s)"
                )
                agent: AgentResult = self._runner().run(
                    project_config, model_id, prompt, progress=self.log
                )
                agent_exit = agent.exit_code
                self.log(
                    f"Agent finished in {agent.duration:.1f}s with exit {agent.exit_code}."
                )
                if not agent.ok:
                    last_detail = f"agent exit {agent.exit_code}:\n{truncate(agent.output, 3000)}"
                    self.log("Agent failed; trying the next free model if available.")
                    continue

                results = run_checks(self.config, project_config, checks)
                if checks_ok(results):
                    success = True
                    break

                last_detail = "checks failed:\n" + format_check_results(results)
                self.log("Checks failed after the first attempt; asking the model to repair.")
                for repair in range(self.config.repair_attempts):
                    repair_prompt = build_prompt(
                        project_config,
                        task,
                        checks,
                        repair_context=last_detail,
                    )
                    repair_agent = self._runner().run(
                        project_config, model_id, repair_prompt, progress=self.log
                    )
                    agent_exit = repair_agent.exit_code
                    self.log(
                        f"Repair {repair + 1}/{self.config.repair_attempts} finished in "
                        f"{repair_agent.duration:.1f}s with exit {repair_agent.exit_code}."
                    )
                    if not repair_agent.ok:
                        last_detail = "repair agent exit {}:\n{}".format(
                            repair_agent.exit_code,
                            truncate(repair_agent.output, 3000),
                        )
                        break
                    results = run_checks(self.config, project_config, checks)
                    if checks_ok(results):
                        success = True
                        break
                    last_detail = "checks failed:\n" + format_check_results(results)
                if success:
                    break
        except KeyboardInterrupt:
            store.update_task(
                str(task.get("id")),
                status="pending",
                notes="Interrupted; task requeued by the operator.",
            )
            store.save()
            self.log("Interrupted; task requeued as pending.")
            raise

        finished = utc_now()
        if success:
            notes = f"Completed with {used_model}; checks passed."
            store.update_task(str(task.get("id")), status="done", notes=notes)
            result = "success"
            detail = notes
            self.log(f"Task {task.get('id')} completed.")
            commit_note = maybe_commit(
                self.config, project_config, task, self.commit, self.push
            )
            if commit_note:
                self.log(commit_note)
                detail += " " + commit_note
        else:
            diff = git_diff_stat(project_config)
            notes = "Blocked after verification failure: " + one_line(last_detail, 500)
            store.update_task(str(task.get("id")), status="blocked", notes=notes)
            result = "blocked"
            detail = one_line(last_detail, 700)
            if diff:
                detail += " | diff: " + one_line(diff, 300)
            self.log("Task blocked.")
            self.log(last_detail[-2500:])

        store.record_run(
            {
                "task": task.get("id"),
                "project": owner,
                "model": used_model,
                "result": result,
                "started": started,
                "finished": finished,
                "duration_s": None,
                "agent_exit": agent_exit,
                "detail": detail,
            }
        )
        store.save()
        return success

    def runner_binary(self) -> str:
        try:
            return self._runner().binary
        except Exception:
            return self.config.agent

    def run_many(
        self,
        max_tasks: int = 1,
        task_id: Optional[str] = None,
        project: Optional[str] = None,
        dry_run: bool = False,
        retry_blocked: bool = False,
        loop: bool = False,
        interval: int = 900,
        auto_plan: bool = False,
    ) -> int:
        import time

        completed = 0
        while True:
            if auto_plan:
                for name in ([project] if project else list(self.stores)):
                    created = discover_todos(self.config, self.stores, name)
                    if created:
                        self.log(
                            f"[{name}] queued {len(created)} TODO task(s): "
                            + ", ".join(item["id"] for item in created)
                        )
            ran = self.run_once(
                task_id=task_id if completed == 0 else None,
                project=project,
                dry_run=dry_run,
                retry_blocked=retry_blocked,
            )
            if not ran:
                if completed == 0:
                    return 1
                break
            completed += 1
            if dry_run or completed >= max_tasks:
                break
            if not loop:
                break
            self.log(f"Sleeping {interval}s before the next task ...")
            time.sleep(interval)
        return 0

    def doctor(self) -> int:
        failures = 0
        self.log("dev-pipeline doctor")
        self.log("===================")
        self.log(f"root: {self.config.root}")
        for name, project in self.config.projects.items():
            configured = "yes" if project.enabled else "no"
            if project.exists:
                self.log(f"  ok   project {name}: {project.path} (AGENTS.md found)")
            else:
                self.log(
                    f"  --   project {name}: {project.path} "
                    f"(enabled={configured}, checkout or AGENTS.md missing)"
                )
        if not self.stores:
            self.log("  FAIL no available project with an AGENTS.md")
            failures += 1

        try:
            agent = self.config.find_agent()
            self.log(f"  ok   agent: {agent}")
        except Exception as exc:
            self.log(f"  FAIL agent: {exc}")
            failures += 1

        try:
            client, working, failed, cached = self.models(refresh=True)
            self.log(
                f"  ok   freeflow: {client.base_url} — {len(working)} working "
                f"({'cached' if cached else 'live probe'})"
            )
            self.log("       working: " + ", ".join(str(item.get("id")) for item in working))
            if failed:
                self.log("       failed: " + ", ".join(str(item.get("id")) for item in failed[:8]))
        except Exception as exc:
            self.log(f"  FAIL freeflow: {exc}")
            failures += 1

        if failures:
            self.log(f"\n{failures} check(s) failed")
        else:
            self.log("\nAll checks passed")
        return 1 if failures else 0
