"""Command line interface for the automatic development pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, PipelineConfig, load_config
from .freeflow import FreeflowError
from .orchestrator import Pipeline, discover_todos, load_stores
from .status import StatusError, StatusStore
from .util import LockError, ensure_state_dir, file_lock, utc_now


DEFAULT_AGENTS_TEMPLATE = """# AGENTS.md

This repository is managed by the gpt4free automatic development pipeline.

## Working agreement

- Read the task board below before changing code.
- Make the smallest correct change and keep the existing style.
- Add or update tests when behaviour changes.
- Run the project checks before declaring a task done.
- Never edit the `DEV-PIPELINE:STATE` or `DEV-PIPELINE:BOARD` blocks by hand;
  use `bin/dev-pipeline add-task`, `run`, and `status` instead.
- Do not add paid services or require API keys for the free pipeline.

## Repository notes

Add project-specific build, test, and architecture notes here.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dev-pipeline",
        description=(
            "Autonomous development pipeline for gpt4free and g4f.dev. "
            "Tasks and status live in each project's AGENTS.md; models come "
            "only from the pi-freeflow proxy."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: the parent of bin/dev-pipeline)",
    )
    parser.add_argument("--config", default=None, help="path to dev-pipeline.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="show extra detail")

    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="show tasks and recorded model health")
    status.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    status.add_argument(
        "--refresh-models", action="store_true", help="probe freeflow models before printing"
    )

    models = sub.add_parser("models", help="probe and list working freeflow models")
    models.add_argument("--refresh", action="store_true", help="ignore the health cache")
    models.add_argument("--limit", type=int, default=None, help="maximum working models to keep")
    models.add_argument("--no-save", action="store_true", help="do not write health to AGENTS.md")

    run = sub.add_parser("run", help="execute the next task from AGENTS.md")
    run.add_argument("--task", default=None, help="run a specific task id")
    run.add_argument("--project", default=None, help="restrict to one project")
    run.add_argument("--model", default=None, help="use one explicit freeflow model id")
    run.add_argument("--dry-run", action="store_true", help="show the plan without editing files")
    run.add_argument("--max-tasks", type=int, default=1, help="tasks to run in this invocation")
    run.add_argument("--loop", action="store_true", help="keep running tasks")
    run.add_argument("--interval", type=int, default=900, help="seconds between loop iterations")
    run.add_argument("--retry-blocked", action="store_true", help="retry blocked tasks too")
    run.add_argument("--auto-plan", action="store_true", help="queue TODO/FIXME comments first")
    run.add_argument("--refresh-models", action="store_true", help="ignore model health cache")
    run.add_argument("--commit", action="store_true", help="git commit successful tasks")
    run.add_argument("--push", action="store_true", help="git push after a successful commit")
    run.add_argument("--no-lock", action="store_true", help="do not take the run lock")

    add = sub.add_parser("add-task", help="append a task to AGENTS.md")
    add.add_argument("--title", required=True, help="short task title")
    add.add_argument("--project", default=None, help="project that owns the task")
    add.add_argument("--priority", default="P2", choices=["P0", "P1", "P2", "P3"])
    add.add_argument("--acceptance", default="", help="definition of done")
    add.add_argument("--id", default=None, help="explicit task id")

    plan = sub.add_parser("plan", help="queue TODO/FIXME comments as tasks")
    plan.add_argument("--project", default=None, help="restrict to one project")
    plan.add_argument("--limit", type=int, default=10, help="maximum tasks to create")

    init = sub.add_parser("init", help="create an AGENTS.md for a configured project")
    init.add_argument("--project", required=True)
    init.add_argument("--path", default=None, help="override the project path")
    init.add_argument("--force", action="store_true", help="overwrite an existing AGENTS.md")

    doctor = sub.add_parser("doctor", help="check projects, agent, and freeflow health")

    return parser


def _root_from_args(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    root = _root_from_args(args)
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    return load_config(root, config_path)


def _print_models(working, failed, base_url: str, cached: bool) -> None:
    print(f"freeflow: {base_url} ({'cache' if cached else 'live probe'})")
    for item in working:
        print(
            "  ok   {id:<42} {latency:>6.2f}s  {detail}".format(
                id=item.get("id", ""),
                latency=float(item.get("latency_s") or 0),
                detail=(item.get("detail") or "")[:60],
            )
        )
    for item in failed:
        print(
            "  fail {id:<42} {detail}".format(
                id=item.get("id", ""),
                detail=(item.get("detail") or "")[:100],
            )
        )


def _save_models(stores, working, failed, base_url: str, cached: bool) -> None:
    for store in stores.values():
        store.record_models(working, failed, base_url, cached)
        store.save()


def _init_project(config: PipelineConfig, name: str, path: Optional[str], force: bool) -> int:
    project = config.projects.get(name)
    if project is None:
        raise ConfigError(
            f"project {name!r} is not configured. Add it to dev-pipeline.json first. "
            f"Known projects: {', '.join(config.projects)}"
        )
    target = Path(path).expanduser().resolve() if path else project.path
    if not target.is_dir():
        raise ConfigError(
            f"project path does not exist: {target}\n"
            "Clone g4f.dev first, for example:\n"
            "  git clone --depth 1 https://github.com/gpt4free/g4f.dev.git ../g4f.dev"
        )
    agents_path = target / project.agents_file
    if agents_path.exists() and not force:
        raise ConfigError(f"{agents_path} already exists (use --force to overwrite)")
    agents_path.write_text(
        DEFAULT_AGENTS_TEMPLATE.replace("This repository", f"The {name} repository"),
        encoding="utf-8",
    )
    store = StatusStore(name, agents_path)
    store.add_task(
        title="Initialize the automatic development pipeline",
        project=name,
        priority="P1",
        acceptance="AGENTS.md exists and `bin/dev-pipeline status` lists this task.",
    )
    task = store.tasks[-1]
    store.update_task(str(task["id"]), status="done", notes="Created by dev-pipeline init.")
    store.save()
    print(f"initialized {agents_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ConfigError as exc:
        print(f"dev-pipeline: {exc}", file=sys.stderr)
        return 2
    ensure_state_dir(config)

    try:
        if args.command == "status":
            pipeline = Pipeline(config)
            if args.json:
                payload = {
                    "updated": utc_now(),
                    "projects": {
                        name: store.state for name, store in pipeline.stores.items()
                    },
                }
                if args.refresh_models:
                    try:
                        client, working, failed, cached = pipeline.models(refresh=True)
                        payload["models"] = {
                            "base_url": client.base_url,
                            "from_cache": cached,
                            "working": working,
                            "failed": failed,
                        }
                    except FreeflowError as exc:
                        payload["models"] = {"error": str(exc)}
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                pipeline.status(refresh_models=args.refresh_models)
            return 0

        if args.command == "models":
            pipeline = Pipeline(config, explicit_model=None, refresh_models=args.refresh)
            client, working, failed, cached = pipeline.models(
                refresh=args.refresh, explore=True
            )
            if args.limit:
                working = working[: args.limit]
            _print_models(working, failed, client.base_url, cached)
            if not args.no_save:
                with file_lock(config.lock_path):
                    _save_models(pipeline.stores, working, failed, client.base_url, cached)
            return 0

        if args.command == "run":
            pipeline = Pipeline(
                config,
                refresh_models=args.refresh_models,
                explicit_model=args.model,
                commit=args.commit,
                push=args.push,
            )
            if not args.no_lock:
                with file_lock(config.lock_path):
                    return pipeline.run_many(
                        max_tasks=args.max_tasks,
                        task_id=args.task,
                        project=args.project,
                        dry_run=args.dry_run,
                        retry_blocked=args.retry_blocked,
                        loop=args.loop,
                        interval=args.interval,
                        auto_plan=args.auto_plan,
                    )
            return pipeline.run_many(
                max_tasks=args.max_tasks,
                task_id=args.task,
                project=args.project,
                dry_run=args.dry_run,
                retry_blocked=args.retry_blocked,
                loop=args.loop,
                interval=args.interval,
                auto_plan=args.auto_plan,
            )

        if args.command == "add-task":
            stores = load_stores(config)
            if not stores:
                print("dev-pipeline: no available project with AGENTS.md", file=sys.stderr)
                return 1
            project_name = args.project or next(iter(stores))
            store = stores.get(project_name)
            if store is None:
                print(
                    f"dev-pipeline: project {project_name!r} is not available",
                    file=sys.stderr,
                )
                return 1
            with file_lock(config.lock_path):
                store.reload()
                task = store.add_task(
                    title=args.title,
                    project=project_name,
                    priority=args.priority,
                    acceptance=args.acceptance,
                    task_id=args.id,
                )
                store.save()
            print(f"added {task['id']}: {task['title']}")
            return 0

        if args.command == "plan":
            with file_lock(config.lock_path):
                stores = load_stores(config)
                targets = [args.project] if args.project else list(stores)
                total = 0
                for name in targets:
                    created = discover_todos(config, stores, name, limit=args.limit)
                    total += len(created)
                    for task in created:
                        print(f"[{name}] {task['id']}: {task['title']}")
                if not total:
                    print("no new TODO/FIXME tasks found")
            return 0

        if args.command == "init":
            with file_lock(config.lock_path):
                return _init_project(config, args.project, args.path, args.force)

        if args.command == "doctor":
            return Pipeline(config).doctor()

        parser.error(f"unknown command {args.command}")
        return 2
    except (ConfigError, FreeflowError, StatusError, LockError) as exc:
        print(f"dev-pipeline: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ndev-pipeline: interrupted", file=sys.stderr)
        return 130
