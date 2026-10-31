# AGENTS.md

This file is the shared context file and status ledger for everyone working in
this repository, including the automatic development pipeline. Read it before
you change code.

## Project

**gpt4free** is a Python library, CLI, OpenAI-compatible API server, and desktop
sandbox that unifies access to many LLM and media providers. The public site and
documentation live at <https://g4f.dev> (source: `gpt4free/g4f.dev`).

The pipeline is designed to work on both workspaces:

| Workspace | Default path | Status file |
| --- | --- | --- |
| gpt4free | `.` | `AGENTS.md` |
| g4f.dev | `../g4f.dev` | `AGENTS.md` |

`g4f.dev` is optional. If it is not checked out, the pipeline skips it and
continues with gpt4free. To enable it:

```bash
git clone --depth 1 https://github.com/gpt4free/g4f.dev.git ../g4f.dev
bin/dev-pipeline init --project g4f.dev
```

## Working agreement

- Read the task board at the bottom of this file before starting.
- Keep changes small, focused, and consistent with the surrounding code.
- Add or update tests when behaviour changes. Prefer the existing `unittest`
  style under `etc/unittest/` and `tools/dev_pipeline/tests/`.
- Run the checks relevant to the files you touched. The pipeline runs:
  - `python3 -m compileall -q g4f`
  - `python3 -m unittest tools.dev_pipeline.tests.test_status`
- Do not add paid services, hard-coded API keys, or dependencies that require
  an account for the pipeline to run.
- Do not rewrite unrelated files. Do not commit generated media or caches.
- The pipeline owns the `DEV-PIPELINE:STATE` and `DEV-PIPELINE:BOARD` blocks.
  Agents must not edit those blocks; the pipeline updates them after each run.

## Repository map

| Path | Purpose |
| --- | --- |
| `g4f/` | Python package: providers, API, client, CLI, MCP, media tools |
| `g4f/Provider/` | Provider implementations and provider routing |
| `g4f/api/` | FastAPI server, OpenAI-compatible routes, sandbox apps |
| `bin/` | Shell and Python helpers, including `bin/dev-pipeline` |
| `etc/unittest/` | Main unit/integration test suite |
| `docs/` | Documentation for APIs, routing, builds, and the pipeline |
| `projects/` | Companion apps (Android, browser extension, Discord bot, ...) |
| `tools/dev_pipeline/` | Automatic development pipeline (standard library only) |

## Commands

```bash
# API server and CLI
python3 -m g4f --port 8080
g4f client "Hello" --model openai

# Quick checks used by the pipeline
python3 -m compileall -q g4f
PYTHONPATH=. python3 -m unittest tools.dev_pipeline.tests.test_status

# Pipeline
bin/dev-pipeline status
bin/dev-pipeline doctor
bin/dev-pipeline models --refresh
bin/dev-pipeline run --dry-run
bin/dev-pipeline run --max-tasks 1
```

## Automatic development pipeline

`bin/dev-pipeline` reads tasks and writes status from this file. It selects the
next pending task, asks a **working free model from the pi-freeflow proxy** to
implement it, runs the project checks, and records the result in the task board
and run log below.

Only free models are allowed. The pipeline discovers them from the local
`pi-freeflow` OpenAI-compatible proxy (`http://127.0.0.1:28180/v1` by default),
probes each candidate with a real chat completion, and refuses to run if none
answer. It never falls back to a paid provider.

Model credentials are read from `FREEFLOW_TOKEN`/`FREEFLOW_API_KEY` or the
standard `~/.pi/agent/token` file. Start the proxy with:

```bash
sandbox-cli start-freeflow-api
```

Configuration lives in `dev-pipeline.json`; full documentation is in
`docs/dev-pipeline.md`.

## Development pipeline status

The blocks below are maintained automatically. Do not edit them by hand.

<!-- DEV-PIPELINE:STATE:BEGIN -->
```json
{
  "schema": 1,
  "project": "gpt4free",
  "updated": "2026-10-31T04:03:11Z",
  "tasks": [
    {
      "id": "G4F-001",
      "project": "gpt4free",
      "priority": "P1",
      "status": "done",
      "title": "Bootstrap the automatic development pipeline",
      "acceptance": "bin/dev-pipeline status/doctor work, freeflow models are probed, and AGENTS.md is read/written.",
      "created": "2026-10-31T03:50:46Z",
      "updated": "2026-10-31T03:50:46Z",
      "attempts": 0,
      "last_model": null,
      "notes": "Implemented tools/dev_pipeline, bin/dev-pipeline, dev-pipeline.json, docs, and tests."
    },
    {
      "id": "G4F-002",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "Document pipeline operations and free-model policy",
      "acceptance": "docs/dev-pipeline.md explains setup, status format, freeflow model probing, and scheduling.",
      "created": "2026-10-31T03:50:46Z",
      "updated": "2026-10-31T03:50:46Z",
      "attempts": 0,
      "last_model": null,
      "notes": "Documented in docs/dev-pipeline.md."
    },
    {
      "id": "G4F-003",
      "project": "gpt4free",
      "priority": "P3",
      "status": "pending",
      "title": "Add a dev-pipeline pr command for GitHub checkouts",
      "acceptance": "The command creates a branch and commit then opens a PR with gh; it fails clearly when no GitHub remote/gh exists.",
      "created": "2026-10-31T03:50:46Z",
      "updated": "2026-10-31T03:50:46Z",
      "attempts": 0,
      "last_model": null,
      "notes": ""
    }
  ],
  "models": {
    "base_url": "http://127.0.0.1:28180/v1",
    "checked_at": "2026-10-31T04:03:11Z",
    "from_cache": false,
    "working": [
      {
        "id": "nvidia/nemotron-3.5-lightning:free",
        "ok": true,
        "latency_s": 1.67,
        "detail": "reasoning-only response",
        "checked_at": "2026-10-31T04:02:31Z"
      },
      {
        "id": "cohere/north-mini-code:free",
        "ok": true,
        "latency_s": 0.72,
        "detail": "OK",
        "checked_at": "2026-10-31T04:02:32Z"
      },
      {
        "id": "ovh-free/gpt-oss-120b",
        "ok": true,
        "latency_s": 1.52,
        "detail": "OK",
        "checked_at": "2026-10-31T04:02:41Z"
      }
    ],
    "failed": [
      {
        "id": "deepseek-v4.1-flash-free",
        "ok": false,
        "latency_s": 4.76,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T04:02:28Z"
      },
      {
        "id": "glm-5.3-flash-free",
        "ok": false,
        "latency_s": 0.21,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T04:02:29Z"
      },
      {
        "id": "mimo-v2.6-flash-free",
        "ok": false,
        "latency_s": 0.25,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T04:02:29Z"
      },
      {
        "id": "nemotron-3.5-lightning-free",
        "ok": false,
        "latency_s": 0.21,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T04:02:29Z"
      },
      {
        "id": "big-pickle",
        "ok": false,
        "latency_s": 0.21,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T04:02:31Z"
      },
      {
        "id": "ovh-free/Qwen3.6-27B",
        "ok": false,
        "latency_s": 6.79,
        "detail": "HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {\"error\":\"upstream error\",\"detail\":\"upstream returned an empty stream\"}",
        "checked_at": "2026-10-31T04:02:38Z"
      },
      {
        "id": "ovh-free/Qwen3.5-397B-A17B",
        "ok": false,
        "latency_s": 1.26,
        "detail": "HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {\"error\":\"upstream error\",\"detail\":\"upstream returned an empty stream\"}",
        "checked_at": "2026-10-31T04:02:40Z"
      }
    ]
  },
  "runs": []
}
```
<!-- DEV-PIPELINE:STATE:END -->

<!-- DEV-PIPELINE:BOARD:BEGIN -->

### Task board

| ID | Project | Priority | Status | Task | Acceptance | Updated |
| --- | --- | --- | --- | --- | --- | --- |
| G4F-001 | gpt4free | P1 | done | Bootstrap the automatic development pipeline | bin/dev-pipeline status/doctor work, freeflow models are probed, and AGENTS.md is read/written. | 2026-10-31T03:50:46Z |
| G4F-002 | gpt4free | P2 | done | Document pipeline operations and free-model policy | docs/dev-pipeline.md explains setup, status format, freeflow model probing, and scheduling. | 2026-10-31T03:50:46Z |
| G4F-003 | gpt4free | P3 | pending | Add a dev-pipeline pr command for GitHub checkouts | The command creates a branch and commit then opens a PR with gh; it fails clearly when no GitHub remote/gh exists. | 2026-10-31T03:50:46Z |

### Free model health

Checked `2026-10-31T04:03:11Z` at `http://127.0.0.1:28180/v1`.

| Model | Status | Latency | Detail |
| --- | --- | --- | --- |
| nvidia/nemotron-3.5-lightning:free | working | 1.67s | reasoning-only response |
| cohere/north-mini-code:free | working | 0.72s | OK |
| ovh-free/gpt-oss-120b | working | 1.52s | OK |
| deepseek-v4.1-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| glm-5.3-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| mimo-v2.6-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| nemotron-3.5-lightning-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| big-pickle | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| ovh-free/Qwen3.6-27B | failed | - | HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {"error":"upstream error","detail":"upstream returned an em... |
| ovh-free/Qwen3.5-397B-A17B | failed | - | HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {"error":"upstream error","detail":"upstream returned an em... |

### Recent runs

_No pipeline runs recorded yet._

<!-- DEV-PIPELINE:BOARD:END -->
