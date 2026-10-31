"""Command line interface for the automatic development pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

from .config import ConfigError, PipelineConfig, load_config
from .freeflow import FreeflowError
from .orchestrator import Pipeline, discover_todos, load_stores
from .status import StatusError, StatusStore
from .util import LockError, ensure_state_dir, file_lock, one_line, utc_now


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
    run.add_argument(
        "--fallback-model",
        action="append",
        default=None,
        metavar="MODEL",
        help=(
            "provider-qualified model tried after freeflow models fail "
            "(repeatable; added before configured fallback_models)"
        ),
    )
    run.add_argument(
        "--fallback-only",
        action="store_true",
        help="skip freeflow discovery and run with fallback models only",
    )
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

    pr = sub.add_parser(
        "pr", help="create or update a PR from the working tree with gh"
    )
    pr.add_argument(
        "--remote",
        default=None,
        help="remote to push to (default: origin, then any other GitHub remote)",
    )

    push = sub.add_parser("push", help="push the current branch without opening a PR")
    push.add_argument(
        "--remote",
        default=None,
        help="remote to push to (default: the tracked remote, then origin, then others)",
    )

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
                fallback_models=args.fallback_model,
                fallback_only=args.fallback_only,
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

        if args.command == "pr":
            return _pr_command(config, args.remote)

        if args.command == "push":
            return _push_command(config, args.remote)

        parser.error(f"unknown command {args.command}")
        return 2
    except (ConfigError, FreeflowError, StatusError, LockError) as exc:
        print(f"dev-pipeline: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ndev-pipeline: interrupted", file=sys.stderr)
        return 130


def _github_slug(url: str) -> Optional[str]:
    """Return ``owner/repo`` for a GitHub remote URL, if it is one.

    The host is compared exactly so that lookalike domains and paths that
    merely contain ``github.com`` are not treated as GitHub remotes.
    """
    value = url.strip()
    # Backslashes make URL parsers disagree about where the authority ends.
    if not value or "\\" in value:
        return None
    try:
        if "://" in value:
            parsed = urlsplit(value)
            host = parsed.hostname
            path = parsed.path
        else:
            # scp-like syntax: git@github.com:owner/repo.git
            host_part, separator, path = value.partition(":")
            if not separator:
                return None
            host = host_part.rsplit("@", 1)[-1]
    except ValueError:
        return None
    if not host or host.lower() != "github.com":
        return None
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [part for part in path.split("/") if part]
    if len(parts) != 2:
        return None
    owner, repo = parts
    return f"{owner}/{repo}"


_PROJECTS_CLASSIC_MARKERS = (
    "projectCards",
    "Projects (classic) is being deprecated",
)


def _projects_classic_error(stderr: str) -> bool:
    """True when gh failed while querying the deprecated Projects (classic) API."""
    return any(marker in stderr for marker in _PROJECTS_CLASSIC_MARKERS)


def _pr_number(url: str) -> Optional[str]:
    """Extract the pull request number from a GitHub PR URL."""
    path = urlsplit(url.strip()).path.rstrip("/")
    number = path.rsplit("/", 1)[-1]
    return number if number.isdigit() else None


def _edit_pr_via_api(
    gh: str, cwd: Path, slug: str, url: str, title: str, body: str
) -> Optional[subprocess.CompletedProcess]:
    """Update a PR through the REST API when ``gh pr edit`` cannot.

    Older gh releases query the removed Projects (classic) field while editing
    a PR and fail with ``repository.pullRequests.nodes.0.projectCards``.  The
    REST endpoint edits title/body without touching project metadata.
    """
    number = _pr_number(url)
    if not slug or not number:
        return None
    return subprocess.run(
        [
            gh,
            "api",
            f"repos/{slug}/pulls/{number}",
            "-X",
            "PATCH",
            "-f",
            f"title={title}",
            "-f",
            f"body={body}",
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
    )


def _push_remote_order(remotes: List[str], requested: Optional[str]) -> List[str]:
    """Order remotes so a read-only origin can fall back to a writable fork."""
    if requested:
        return [requested]
    if "origin" in remotes:
        return ["origin"] + [name for name in remotes if name != "origin"]
    return sorted(remotes)


def _default_branch(git, remote: str) -> str:
    """Best-effort lookup of a remote's default branch (for ``--base``)."""
    result = git("symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD")
    value = result.stdout.strip()
    if result.returncode == 0 and value:
        return value.split("/", 1)[-1]
    for candidate in ("main", "master"):
        check = git("rev-parse", "--verify", f"refs/remotes/{remote}/{candidate}")
        if check.returncode == 0:
            return candidate
    return ""


def _push_command(config: PipelineConfig, remote: Optional[str] = None) -> int:
    """Push the current branch without creating a commit or a pull request."""
    root = config.root

    def fail(message: str) -> int:
        print(f"dev-pipeline: push: {message}", file=sys.stderr)
        return 1

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
        )

    if not (root / ".git").exists():
        return fail("not a git repository (no .git directory)")

    remotes_result = git("remote")
    if remotes_result.returncode != 0:
        return fail("could not list git remotes")
    remotes = [
        entry.strip() for entry in remotes_result.stdout.splitlines() if entry.strip()
    ]
    if not remotes:
        return fail(
            "no git remote configured; add one with: "
            "git remote add origin https://github.com/<owner>/<repo>.git"
        )
    if remote is not None and remote not in remotes:
        return fail(
            f"remote {remote!r} is not configured; choose one of: "
            + ", ".join(sorted(remotes))
        )

    branch_result = git("branch", "--show-current")
    branch = branch_result.stdout.strip()
    if branch_result.returncode != 0 or not branch:
        return fail("could not determine the current branch (detached HEAD?)")

    push_order = _push_remote_order(sorted(remotes), remote)
    if remote is None:
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream.returncode == 0:
            tracked = upstream.stdout.strip().split("/", 1)[0]
            if tracked in remotes:
                push_order = [tracked] + [
                    name for name in push_order if name != tracked
                ]

    push_error = ""
    for name in push_order:
        attempt = git("push", "-u", name, branch)
        if attempt.returncode == 0:
            print(f"dev-pipeline: push: pushed {branch} to {name}")
            return 0
        push_error = one_line(attempt.stderr, 300)
        if name != push_order[-1]:
            print(
                f"dev-pipeline: push: {name} is not writable, trying the next remote",
                file=sys.stderr,
            )
    return fail(
        f"failed to push {branch} to any remote ({', '.join(push_order)}): "
        + push_error
    )


def _pr_command(config: PipelineConfig, remote: Optional[str] = None) -> int:
    """Create a branch, commit the working tree, and open a PR with gh."""
    root = config.root

    def fail(message: str) -> int:
        print(f"dev-pipeline: pr: {message}", file=sys.stderr)
        return 1

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
        )

    if not (root / ".git").exists():
        return fail("not a git repository (no .git directory)")

    remotes = git("remote", "-v")
    if remotes.returncode != 0:
        return fail("could not list git remotes")
    remote_lines = [line for line in remotes.stdout.splitlines() if line.strip()]
    if not remote_lines:
        return fail(
            "no git remote configured; add one with: "
            "git remote add origin https://github.com/<owner>/<repo>.git"
        )
    github_remotes = {}
    for line in remote_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        if _github_slug(parts[1]) is not None:
            github_remotes.setdefault(parts[0], parts[1])
    if not github_remotes:
        return fail("no GitHub remote configured; `gh pr create` needs a github.com remote")
    if remote is not None and remote not in github_remotes:
        return fail(
            f"remote {remote!r} is not a configured GitHub remote; choose one of: "
            + ", ".join(sorted(github_remotes))
        )
    push_order = _push_remote_order(sorted(github_remotes), remote)

    gh = shutil.which("gh")
    if not gh:
        return fail(
            "gh (GitHub CLI) not found; install it from https://cli.github.com/ "
            "and run `gh auth login`"
        )

    auth = subprocess.run(
        [gh, "auth", "status"], cwd=root, text=True, capture_output=True
    )
    if auth.returncode != 0:
        return fail(
            "gh is not authenticated; run `gh auth login` or set "
            "GH_TOKEN/GITHUB_TOKEN"
        )

    branch_result = git("branch", "--show-current")
    branch = branch_result.stdout.strip()
    if branch_result.returncode != 0 or not branch:
        return fail("could not determine the current branch (detached HEAD?)")

    status = git("status", "--porcelain")
    if status.returncode != 0:
        return fail("could not read git status")
    has_changes = bool(status.stdout.strip())
    reusing_branch = branch.startswith("pipeline/pr-")
    if not has_changes and not reusing_branch:
        return fail("no changes to commit")

    if reusing_branch:
        # Already on a pipeline PR branch: update that PR in place instead of
        # opening another one for the same change.
        pr_branch = branch
        print(f"dev-pipeline: pr: updating branch {pr_branch}")
    else:
        pr_branch = "pipeline/pr-" + utc_now().replace("-", "").replace(":", "").replace("Z", "")
        created = git("switch", "-c", pr_branch)
        if created.returncode != 0:
            return fail("failed to create branch: " + one_line(created.stderr, 300))
        print(f"dev-pipeline: pr: created branch {pr_branch}")

    if has_changes:
        added = git("add", "-A")
        if added.returncode != 0:
            if not reusing_branch:
                git("switch", branch)
            return fail("failed to stage changes: " + one_line(added.stderr, 300))

        message = "pipeline: prepare PR"
        committed = git("commit", "-m", message)
        if committed.returncode != 0:
            if not reusing_branch:
                git("switch", branch)
            return fail("failed to commit: " + one_line(committed.stderr, 300))
        print(f"dev-pipeline: pr: committed {message}")

    # Keep pushing to the remote that already tracks the existing PR branch
    # before trying the default origin-first order.
    if reusing_branch and remote is None:
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if upstream.returncode == 0:
            tracked = upstream.stdout.strip().split("/", 1)[0]
            if tracked in github_remotes:
                push_order = [tracked] + [
                    name for name in push_order if name != tracked
                ]

    pushed = None
    pushed_remote = None
    push_error = ""
    for name in push_order:
        attempt = git("push", "-u", name, pr_branch)
        if attempt.returncode == 0:
            pushed = attempt
            pushed_remote = name
            break
        push_error = one_line(attempt.stderr, 300)
        if name != push_order[-1]:
            print(
                f"dev-pipeline: pr: {name} is not writable, trying the next remote "
                f"({_github_slug(github_remotes[name]) or name})",
                file=sys.stderr,
            )
    if pushed is None or pushed_remote is None:
        if not reusing_branch:
            git("switch", branch)
        return fail(
            f"failed to push {pr_branch} to any GitHub remote "
            f"({', '.join(push_order)}): {push_error}"
        )
    print(f"dev-pipeline: pr: pushed {pr_branch} to {pushed_remote}")

    # A read-only upstream is common: push to the fork, but open the PR against
    # the upstream repository.  Cross-repo PRs need ``owner:branch`` as head.
    base_remote = next(
        (name for name in push_order if name != pushed_remote and name == "origin"),
        None,
    )
    if base_remote is None:
        base_remote = next(
            (name for name in push_order if name != pushed_remote), pushed_remote
        )
    base_slug = _github_slug(github_remotes[base_remote])
    push_slug = _github_slug(github_remotes[pushed_remote])
    head = pr_branch
    if base_slug and push_slug and base_slug != push_slug:
        head = f"{push_slug.split('/', 1)[0]}:{pr_branch}"
    base_branch = _default_branch(git, base_remote) or branch
    repo_args = ["--repo", base_slug] if base_slug else []

    title = "Automated pipeline changes"
    body = "Created by `bin/dev-pipeline pr`."
    existing = subprocess.run(
        [gh, "pr", "view", head, *repo_args, "--json", "url", "--jq", ".url"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if existing.returncode == 0:
        result = subprocess.run(
            [gh, "pr", "edit", head, *repo_args, "--title", title, "--body", body],
            cwd=root,
            text=True,
            capture_output=True,
        )
        action = "updated"
        if result.returncode != 0 and _projects_classic_error(result.stderr):
            fallback = _edit_pr_via_api(
                gh,
                root,
                base_slug or "",
                existing.stdout.strip(),
                title,
                body,
            )
            if fallback is not None:
                print(
                    "dev-pipeline: pr: gh pr edit hit the removed Projects "
                    "(classic) query; updated the PR through the REST API"
                )
                result = fallback
                if fallback.returncode == 0:
                    # gh pr edit is quiet; do not dump the API response.
                    result = subprocess.CompletedProcess(
                        fallback.args, fallback.returncode, "", fallback.stderr
                    )
    else:
        result = subprocess.run(
            [
                gh,
                "pr",
                "create",
                "--title",
                title,
                "--body",
                body,
                "--base",
                base_branch,
                "--head",
                head,
                *repo_args,
            ],
            cwd=root,
            text=True,
            capture_output=True,
        )
        action = "created"

    if result.returncode != 0:
        git("switch", branch)
        print(
            f"dev-pipeline: pr: gh pr {action} failed: "
            + one_line(result.stderr, 400),
            file=sys.stderr,
        )
        print(
            f"  Branch {pr_branch} was pushed; switch back with "
            f"`git switch {branch}` and retry the gh command.",
            file=sys.stderr,
        )
        return 1

    print(f"dev-pipeline: pr: PR {action} successfully!")
    print(result.stdout.strip() or f"PR {action}")
    return 0
