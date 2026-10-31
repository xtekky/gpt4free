# Automatic development pipeline

`bin/dev-pipeline` is a small, dependency-free orchestrator that turns the task
board in each project's `AGENTS.md` into an autonomous development loop:

1. read tasks and status from `AGENTS.md`;
2. discover free models from the local [pi-freeflow](https://github.com/trefeon/pi-freeflow) proxy;
3. probe each candidate with a real chat completion and keep only working models;
4. pick the highest-priority pending task;
5. ask the selected model (through Pi or OMP) to implement it;
6. run the project checks, optionally ask the same model to repair failures;
7. write the result, model health, and run log back to `AGENTS.md`.

Only free models are used by default. The pipeline never falls back to a paid
provider on its own: if the freeflow proxy is unreachable or no probed free
model answers, the run stops and records the failure. An operator can opt in to
provider-qualified `fallback_models` (for example a model served by Pollinations)
when the agent is already configured for that provider; those models are tried
after the freeflow candidates.

## Files

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Agent instructions plus the canonical pipeline state, task board, and run log |
| `dev-pipeline.json` | Projects, checks, model preferences, timeouts, and safety switches |
| `bin/dev-pipeline` | CLI entry point |
| `tools/dev_pipeline/` | Implementation (standard library only) |
| `tools/dev_pipeline/freeflow.py` | Model discovery, health probing, ranking, cache |
| `tools/dev_pipeline/status.py` | `AGENTS.md` parsing, rendering, and atomic writes |
| `tools/dev_pipeline/orchestrator.py` | Task selection, prompt building, run loop |
| `tools/dev_pipeline/runner.py` | Pi/OMP invocation, checks, git commit helper |
| `tools/dev_pipeline/tests/` | Unit tests for the state store and model ranking |
| `.dev-pipeline/` | Local run lock and model-health cache (git-ignored) |

## Prerequisites

- Python 3.8+ (the repository's normal runtime).
- Pi or OMP on `PATH`, with the `pi-freeflow` extension installed.
- The freeflow proxy running on `127.0.0.1:28180` (or a compatible base URL).

```bash
# Install Pi + freeflow (the sandbox image does this already)
curl -fsSL https://pi.dev/install.sh | sh
pi install npm:pi-freeflow

# Start the shared proxy
sandbox-cli start-freeflow-api
```

The proxy writes its bearer token to `~/.pi/agent/token`. The pipeline checks,
in order:

1. `FREEFLOW_TOKEN`, `FREEFLOW_API_KEY`, or `G4F_FREEFLOW_TOKEN`;
2. `<project>/.pi/agent/token`;
3. `<repository>/.pi/agent/token`;
4. `~/.pi/agent/token`;
5. no token (in case a local relay allows it).

Set `FREEFLOW_BASE_URL` (or `base_url` in `dev-pipeline.json`) to point at a
different proxy.

## Enabling g4f.dev

The gpt4free checkout is configured as `gpt4free`; the website checkout is
configured as `g4f.dev` at `../g4f.dev`. Missing projects are skipped, so the
pipeline works with only one checkout.

```bash
git clone --depth 1 https://github.com/gpt4free/g4f.dev.git ../g4f.dev
bin/dev-pipeline init --project g4f.dev
```

`init` creates an `AGENTS.md` in that checkout with a bootstrap task and the
same machine-readable state block. Add website-specific checks to
`dev-pipeline.json` when you know the command, for example:

```json
{
  "projects": {
    "g4f.dev": {
      "path": "../g4f.dev",
      "checks": [["node", "scripts/check-js.mjs"]]
    }
  }
}
```

For git checkouts the pipeline always adds `git diff --check` to the checks so
whitespace errors are caught even when no test command is configured.

## Commands

```bash
# Inspect tasks and recorded model health
bin/dev-pipeline status
bin/dev-pipeline status --json
bin/dev-pipeline status --refresh-models

# Probe freeflow models and persist the health snapshot
bin/dev-pipeline models --refresh
bin/dev-pipeline models --refresh --limit 5 --no-save

# Show what would run without editing files or invoking an agent
bin/dev-pipeline run --dry-run
bin/dev-pipeline run --dry-run --task G4F-003

# Run one task (default), or several
bin/dev-pipeline run --max-tasks 1
bin/dev-pipeline run --max-tasks 3 --retry-blocked

# Try an extra fallback model after the freeflow candidates fail
bin/dev-pipeline run --fallback-model pollinations/deepseek/deepseek-v4.1-flash-1017

# Skip freeflow discovery and run with fallbacks only (requires a fallback)
bin/dev-pipeline run --fallback-only \
  --fallback-model pollinations/deepseek/deepseek-v4.1-flash-1017

# Run forever, one task every 15 minutes
bin/dev-pipeline run --loop --max-tasks 1000 --interval 900

# Queue TODO/FIXME comments as P3 tasks, then run
bin/dev-pipeline plan --limit 10
bin/dev-pipeline run --auto-plan

# Task management
bin/dev-pipeline add-task --title "Fix streaming retry" --project gpt4free \
  --priority P2 --acceptance "Streaming retries once on 429 and tests pass"

# Health check: projects, agent, freeflow, working models
bin/dev-pipeline doctor

# Create/update a PR from the current working tree
bin/dev-pipeline pr
bin/dev-pipeline pr --project g4f.dev

# Push the current branch without creating or updating a PR
bin/dev-pipeline push
bin/dev-pipeline push --remote fork
```

`push` pushes the current branch to a git remote without committing anything or
requiring `gh`. It prefers the branch's tracked remote, then `origin`, and falls
back to the other configured remotes when one is not writable. Use
`--remote <name>` to force a specific remote.

`pr` creates `pipeline/pr-<timestamp>`, commits the working tree, pushes it to
a GitHub remote, then creates or updates the PR with `gh`. By default it
operates on the pipeline root checkout; use `--project <name>` to open the PR
from another configured checkout such as `g4f.dev`. When the current branch is
already a `pipeline/pr-*` branch, it commits and pushes that branch in place, so
the existing PR is updated instead of a second PR being opened (a clean branch
is allowed in that case). It tries `origin` first and falls back to another
configured GitHub remote (typically a fork) when `origin` is read-only;
cross-repo PRs use `owner:branch` as the head, and an existing branch keeps
pushing to its tracked remote first. Use `--remote <name>` to force a specific
remote. Install `gh` and run `gh auth login`, or set `GH_TOKEN`/`GITHUB_TOKEN`
(for example in CI). The command validates the repository, remote, `gh`
availability, and authentication before it changes any branch, so a missing
tool never leaves a half-made commit.

During `run`, the agent is invoked in Pi's JSON event mode. The pipeline prints
each tool call as it happens and a heartbeat every 30 seconds, so a long agent
run shows progress instead of appearing idle:

```
Running /home/pn/.local/bin/pi with cohere/north-mini-code:free ... (streaming progress, timeout 1800s)
  -> bash: pwd
  <- bash (ok, 0.1s)
```

Pressing Ctrl-C requeues the current task as `pending` instead of leaving it
`in-progress`.

Global options (`--root`, `--config`) must come before the subcommand:

```bash
bin/dev-pipeline --config /path/to/dev-pipeline.json status
```

## Status format in AGENTS.md

`AGENTS.md` contains two generated blocks:

````markdown
<!-- DEV-PIPELINE:STATE:BEGIN -->
```json
{
  "schema": 1,
  "project": "gpt4free",
  "tasks": [
    {
      "id": "G4F-003",
      "project": "gpt4free",
      "priority": "P3",
      "status": "pending",
      "title": "...",
      "acceptance": "...",
      "attempts": 0,
      "last_model": null,
      "notes": ""
    }
  ],
  "models": { "working": [], "failed": [] },
  "runs": []
}
```
<!-- DEV-PIPELINE:STATE:END -->

<!-- DEV-PIPELINE:BOARD:BEGIN -->
### Task board
| ID | Project | Priority | Status | Task | Acceptance | Updated |
...
### Free model health
| Model | Status | Latency | Detail |
...
### Recent runs
| When | Task | Project | Model | Result | Detail |
...
<!-- DEV-PIPELINE:BOARD:END -->
````

The JSON block is canonical. The board and recent-runs tables are regenerated
from it every time the pipeline writes the file, so they are safe to read but
not safe to edit. Human edits to the JSON block are picked up on the next run;
prefer `add-task` and `plan` for changes.

Task statuses:

| Status | Meaning |
| --- | --- |
| `pending` | ready to run |
| `in-progress` | currently owned by a run; stale entries are recovered after two hours |
| `done` | agent exit 0 and all checks passed |
| `blocked` | baseline/verification failed, or no free model worked |

Blocked tasks are not retried by default. Use `--retry-blocked` after fixing the
environment or reviewing the diff.

## Free-model policy

All 32 models exposed by pi-freeflow are free. The pipeline still verifies each
one because free lanes are rate-limited and occasionally return empty streams:

1. `GET /v1/models` lists the catalog.
2. Candidates are ranked by `preferred_models` (substring matching accepts both
   short aliases and full ids, e.g. `qwen3.5-397b` matches
   `ovh-free/Qwen3.5-397B-A17B`).
3. Each candidate is probed with a small non-streaming chat completion.
4. A model is usable only if the response has choices and non-empty content or
   reasoning. Empty 200 responses are failures.
5. Entries in `disabled_models` are removed before probing, even when the
   proxy still lists them; short aliases and full ids both match. Health is
   cached in `.dev-pipeline/models.json` for `model_cache_ttl` seconds
   (default 300). `--refresh-models` bypasses the cache.
6. On an agent failure the next working model is tried, up to
   `max_model_attempts`.
7. If every freeflow attempt fails, optional `fallback_models` are tried next.
   Entries are provider-qualified ids understood by the agent, such as
   `pollinations/deepseek/deepseek-v4.1-flash-1017`. Configure the agent
   provider first, for example:

   ```bash
   polli harness pi on --model deepseek/deepseek-v4.1-flash-1017 --no-mcp
   ```

   Pollinations models can be paid. Fallbacks are opt-in and empty by default,
   so the free-only guarantee holds unless an operator adds them.

   `run --fallback-model <id>` adds a fallback for one invocation (repeatable);
   these are tried before the configured `fallback_models`. `run
   --fallback-only` skips freeflow discovery entirely and uses only fallback
   models, which is useful when the proxy is down or rate-limited.

Probing stops as soon as `min_models` working models are found, to conserve free
quotas. The default preferred order starts with OpenCode Zen and Kilo coding
models, then OVHcloud models as keyless fallbacks.

## Configuration reference

`dev-pipeline.json`:

| Key | Default | Meaning |
| --- | --- | --- |
| `projects` | gpt4free + g4f.dev | map of project name to `path`, `agents_file`, `checks`, `enabled` |
| `preferred_models` | coding-first list | ranking hints for free models |
| `disabled_models` | empty | models or aliases never probed or used (substring matching) |
| `fallback_models` | `[]` | optional provider-qualified models tried after freeflow failures |
| `base_url` | `http://127.0.0.1:28180/v1` | freeflow-compatible endpoint |
| `probe_limit` | `10` | maximum candidates probed per refresh |
| `min_models` | `2` | stop probing after this many working models |
| `model_cache_ttl` | `300` | seconds before cached health is stale |
| `max_model_attempts` | `3` | fallback models per task |
| `repair_attempts` | `1` | repair prompts after check failures |
| `agent` | `pi` | agent executable (`pi` or `omp`) |
| `thinking` | `high` | Pi thinking level |
| `task_timeout` | `1800` | seconds per agent invocation |
| `check_timeout` | `300` | seconds per check command |
| `commit` | `false` | git commit successful tasks |
| `push` | `false` | git push after a successful commit |
| `enabled_projects` | all | optional allowlist |

Checks are argv lists, not shell strings, so quoting is predictable:

```json
"checks": [
  ["python3", "-m", "compileall", "-q", "g4f"],
  ["python3", "-m", "unittest", "tools.dev_pipeline.tests.test_status"]
]
```

## Scheduling

### systemd user timer

`~/.config/systemd/user/dev-pipeline.service`:

```ini
[Unit]
Description=gpt4free automatic development pipeline
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/gpt4free
ExecStart=%h/gpt4free/bin/dev-pipeline run --max-tasks 1 --auto-plan
TimeoutStartSec=3h
```

`~/.config/systemd/user/dev-pipeline.timer`:

```ini
[Unit]
Description=Run the gpt4free dev pipeline every 30 minutes

[Timer]
OnBootSec=10min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now dev-pipeline.timer
journalctl --user -u dev-pipeline -f
```

### cron

```cron
*/30 * * * * cd /path/to/gpt4free && ./bin/dev-pipeline run --max-tasks 1 --auto-plan >> ~/.dev-pipeline.log 2>&1
```

The run lock prevents overlapping invocations.

## Safety notes

- `--dry-run` does not edit `AGENTS.md`, does not run checks, and does not invoke
  the agent. It does probe/list models unless a fresh cache exists, or unless
  `--fallback-only` is set.
- Baseline checks run before the model is invoked. A broken checkout is marked
  `blocked` instead of burning model attempts on an unrelated failure.
- Commits are opt-in (`--commit`); pushes require both `--commit` and `--push`.
- The agent prompt forbids commits, pushes, paid services, and edits to the
  pipeline state blocks.
- A failed run leaves the working tree intact and records the failure for human
  review. There is no automatic revert.
- Free quotas are shared. Keep `probe_limit` low and avoid tight loop intervals.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pi-freeflow is not reachable` | Run `sandbox-cli start-freeflow-api`; check `~/.pi/agent/pi-freeflow.log` |
| `all probed freeflow models failed` | Run `bin/dev-pipeline models --refresh`; free lanes may be rate-limited, retry later |
| `no agent executable found` | Install Pi/OMP and `pi install npm:pi-freeflow` |
| `project g4f.dev path does not exist` | Clone g4f.dev or remove/disable it in `dev-pipeline.json` |
| `baseline checks failed` | Fix the checkout, or run the check command directly and inspect the output |
| `another dev-pipeline run holds ...` | Wait for the other run, or remove a stale `.dev-pipeline/run.lock` after confirming no process is active |
| Task stuck `in-progress` | It is recovered after two hours, or edit the status to `pending` and run again |

## Tests

```bash
PYTHONPATH=. python3 -m unittest tools.dev_pipeline.tests.test_status -v
```

The pipeline's own `gpt4free` check list includes this test, so every
autonomous task also verifies that the status store still round-trips.
