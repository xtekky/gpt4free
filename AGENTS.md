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

Only free models are used by default. The pipeline discovers them from the local
`pi-freeflow` OpenAI-compatible proxy (`http://127.0.0.1:28180/v1` by default),
probes each candidate with a real chat completion, and refuses to run if none
answer. It never falls back to a paid provider on its own; an operator can opt
in to provider-qualified `fallback_models` in `dev-pipeline.json` (for example a
Pollinations model configured through `polli harness pi`).

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
  "updated": "2026-10-31T06:09:09Z",
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
      "status": "done",
      "title": "Add a dev-pipeline pr command for GitHub checkouts",
      "acceptance": "The command creates a branch and commit then opens a PR with gh; it fails clearly when no GitHub remote/gh exists.",
      "created": "2026-10-31T03:50:46Z",
      "updated": "2026-10-31T05:05:27Z",
      "attempts": 2,
      "last_model": "nvidia/nemotron-3.5-lightning:free",
      "notes": "Completed with nvidia/nemotron-3.5-lightning:free; checks passed."
    },
    {
      "id": "G4F-004",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "push/create/updated pr",
      "acceptance": "",
      "created": "2026-10-31T05:11:29Z",
      "updated": "2026-10-31T05:23:35Z",
      "attempts": 2,
      "last_model": "nvidia/nemotron-3.5-lightning:free",
      "notes": "Completed with nvidia/nemotron-3.5-lightning:free; checks passed."
    },
    {
      "id": "G4F-005",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "create new pr #3553",
      "acceptance": "",
      "created": "2026-10-31T05:25:50Z",
      "updated": "2026-10-31T05:53:04Z",
      "attempts": 4,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "id": "G4F-006",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "github codeql check failed, update pr",
      "acceptance": "",
      "created": "2026-10-31T05:45:04Z",
      "updated": "2026-10-31T05:54:37Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "id": "G4F-007",
      "project": "gpt4free",
      "priority": "P3",
      "status": "done",
      "title": "Resolve TODO in _youtube_converter.py: Fix json type",
      "acceptance": "Address `g4f/integration/markitdown/_youtube_converter.py:288` without breaking the configured checks.",
      "created": "2026-10-31T05:51:24Z",
      "updated": "2026-10-31T05:55:38Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed.",
      "source": "g4f/integration/markitdown/_youtube_converter.py:288"
    },
    {
      "id": "G4F-008",
      "project": "gpt4free",
      "priority": "P3",
      "status": "done",
      "title": "Resolve TODO in optimize_request.py: list with specific, actionable \"",
      "acceptance": "Address `g4f/tools/optimize_request.py:328` without breaking the configured checks.",
      "created": "2026-10-31T05:51:24Z",
      "updated": "2026-10-31T05:57:27Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed.",
      "source": "g4f/tools/optimize_request.py:328"
    },
    {
      "id": "G4F-009",
      "project": "gpt4free",
      "priority": "P3",
      "status": "done",
      "title": "Resolve TODO in optimize_request.py: as in-progress before starting work\\s*\\n3\\. Complete the \"",
      "acceptance": "Address `g4f/tools/optimize_request.py:329` without breaking the configured checks.",
      "created": "2026-10-31T05:51:24Z",
      "updated": "2026-10-31T05:59:39Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed.",
      "source": "g4f/tools/optimize_request.py:329"
    },
    {
      "id": "G4F-010",
      "project": "gpt4free",
      "priority": "P3",
      "status": "pending",
      "title": "Resolve TODO in optimize_request.py: \\s*\\n4\\. Mark that todo as completed IMMEDIATELY\\s*\\n5\\. \"",
      "acceptance": "Address `g4f/tools/optimize_request.py:330` without breaking the configured checks.",
      "created": "2026-10-31T05:51:24Z",
      "updated": "2026-10-31T05:51:24Z",
      "attempts": 0,
      "last_model": null,
      "notes": "",
      "source": "g4f/tools/optimize_request.py:330"
    },
    {
      "id": "G4F-011",
      "project": "gpt4free",
      "priority": "P3",
      "status": "pending",
      "title": "Resolve TODO in optimize_request.py: and repeat\"",
      "acceptance": "Address `g4f/tools/optimize_request.py:331` without breaking the configured checks.",
      "created": "2026-10-31T05:51:24Z",
      "updated": "2026-10-31T05:51:24Z",
      "attempts": 0,
      "last_model": null,
      "notes": "",
      "source": "g4f/tools/optimize_request.py:331"
    },
    {
      "id": "G4F-012",
      "project": "gpt4free",
      "priority": "P3",
      "status": "pending",
      "title": "Resolve TODO in test_optimize_request.py: \", \"to-do\", \"to do\"):",
      "acceptance": "Address `etc/unittest/test_optimize_request.py:33` without breaking the configured checks.",
      "created": "2026-10-31T05:58:21Z",
      "updated": "2026-10-31T05:58:21Z",
      "attempts": 0,
      "last_model": null,
      "notes": "",
      "source": "etc/unittest/test_optimize_request.py:33"
    },
    {
      "id": "G4F-013",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "#3545 \u001b[200~The string github.com may be at an arbitrary position in the sanitized URL.~",
      "acceptance": "",
      "created": "2026-10-31T06:01:41Z",
      "updated": "2026-10-31T06:03:56Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "id": "G4F-014",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "update pr",
      "acceptance": "",
      "created": "2026-10-31T06:04:35Z",
      "updated": "2026-10-31T06:06:35Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "id": "G4F-015",
      "project": "gpt4free",
      "priority": "P2",
      "status": "done",
      "title": "push branch",
      "acceptance": "",
      "created": "2026-10-31T06:07:13Z",
      "updated": "2026-10-31T06:09:09Z",
      "attempts": 1,
      "last_model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "notes": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    }
  ],
  "models": {
    "base_url": "http://127.0.0.1:28180/v1",
    "checked_at": "2026-10-31T05:45:31Z",
    "from_cache": false,
    "working": [
      {
        "id": "cohere/north-mini-code:free",
        "ok": true,
        "latency_s": 3.84,
        "detail": "OK",
        "checked_at": "2026-10-31T05:45:19Z"
      },
      {
        "id": "ovh-free/gpt-oss-120b",
        "ok": true,
        "latency_s": 1.09,
        "detail": "OK",
        "checked_at": "2026-10-31T05:45:31Z"
      }
    ],
    "failed": [
      {
        "id": "deepseek-v4.1-flash-free",
        "ok": false,
        "latency_s": 2.03,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T05:45:14Z"
      },
      {
        "id": "glm-5.3-flash-free",
        "ok": false,
        "latency_s": 0.39,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T05:45:14Z"
      },
      {
        "id": "mimo-v2.6-flash-free",
        "ok": false,
        "latency_s": 0.46,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T05:45:15Z"
      },
      {
        "id": "big-pickle",
        "ok": false,
        "latency_s": 0.47,
        "detail": "HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {\"type\":\"error\",\"error\":{\"type\":\"FreeUsageLimitError\",\"message\":\"Free usage limit reached. Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go to keep going: https://opencode.ai/go\"}}",
        "checked_at": "2026-10-31T05:45:15Z"
      },
      {
        "id": "ovh-free/Qwen3.6-27B",
        "ok": false,
        "latency_s": 8.56,
        "detail": "HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {\"error\":\"upstream error\",\"detail\":\"upstream returned an empty stream\"}",
        "checked_at": "2026-10-31T05:45:28Z"
      },
      {
        "id": "ovh-free/Qwen3.5-397B-A17B",
        "ok": false,
        "latency_s": 2.1,
        "detail": "HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {\"error\":\"upstream error\",\"detail\":\"upstream returned an empty stream\"}",
        "checked_at": "2026-10-31T05:45:30Z"
      }
    ]
  },
  "runs": [
    {
      "task": "G4F-003",
      "project": "gpt4free",
      "model": "nvidia/nemotron-3.5-lightning:free",
      "result": "success",
      "started": "2026-10-31T05:02:03Z",
      "finished": "2026-10-31T05:05:27Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with nvidia/nemotron-3.5-lightning:free; checks passed."
    },
    {
      "task": "G4F-004",
      "project": "gpt4free",
      "model": "nvidia/nemotron-3.5-lightning:free",
      "result": "success",
      "started": "2026-10-31T05:21:42Z",
      "finished": "2026-10-31T05:23:35Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with nvidia/nemotron-3.5-lightning:free; checks passed."
    },
    {
      "task": "G4F-005",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T05:51:24Z",
      "finished": "2026-10-31T05:53:04Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-006",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T05:53:16Z",
      "finished": "2026-10-31T05:54:37Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-007",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T05:54:46Z",
      "finished": "2026-10-31T05:55:38Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-008",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T05:55:47Z",
      "finished": "2026-10-31T05:57:27Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-009",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T05:58:22Z",
      "finished": "2026-10-31T05:59:39Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-013",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T06:01:47Z",
      "finished": "2026-10-31T06:03:56Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-014",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T06:04:39Z",
      "finished": "2026-10-31T06:06:35Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    },
    {
      "task": "G4F-015",
      "project": "gpt4free",
      "model": "pollinations/deepseek/deepseek-v4.1-flash-1017",
      "result": "success",
      "started": "2026-10-31T06:07:16Z",
      "finished": "2026-10-31T06:09:09Z",
      "duration_s": null,
      "agent_exit": 0,
      "detail": "Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed."
    }
  ]
}
```
<!-- DEV-PIPELINE:STATE:END -->

<!-- DEV-PIPELINE:BOARD:BEGIN -->

### Task board

| ID | Project | Priority | Status | Task | Acceptance | Updated |
| --- | --- | --- | --- | --- | --- | --- |
| G4F-001 | gpt4free | P1 | done | Bootstrap the automatic development pipeline | bin/dev-pipeline status/doctor work, freeflow models are probed, and AGENTS.md is read/written. | 2026-10-31T03:50:46Z |
| G4F-002 | gpt4free | P2 | done | Document pipeline operations and free-model policy | docs/dev-pipeline.md explains setup, status format, freeflow model probing, and scheduling. | 2026-10-31T03:50:46Z |
| G4F-004 | gpt4free | P2 | done | push/create/updated pr |  | 2026-10-31T05:23:35Z |
| G4F-005 | gpt4free | P2 | done | create new pr #3553 |  | 2026-10-31T05:53:04Z |
| G4F-006 | gpt4free | P2 | done | github codeql check failed, update pr |  | 2026-10-31T05:54:37Z |
| G4F-013 | gpt4free | P2 | done | #3545 [200~The string github.com may be at an arbitrary position in the sanitized URL.~ |  | 2026-10-31T06:03:56Z |
| G4F-014 | gpt4free | P2 | done | update pr |  | 2026-10-31T06:06:35Z |
| G4F-015 | gpt4free | P2 | done | push branch |  | 2026-10-31T06:09:09Z |
| G4F-003 | gpt4free | P3 | done | Add a dev-pipeline pr command for GitHub checkouts | The command creates a branch and commit then opens a PR with gh; it fails clearly when no GitHub remote/gh exists. | 2026-10-31T05:05:27Z |
| G4F-007 | gpt4free | P3 | done | Resolve TODO in _youtube_converter.py: Fix json type | Address `g4f/integration/markitdown/_youtube_converter.py:288` without breaking the configured checks. | 2026-10-31T05:55:38Z |
| G4F-008 | gpt4free | P3 | done | Resolve TODO in optimize_request.py: list with specific, actionable " | Address `g4f/tools/optimize_request.py:328` without breaking the configured checks. | 2026-10-31T05:57:27Z |
| G4F-009 | gpt4free | P3 | done | Resolve TODO in optimize_request.py: as in-progress before starting work\\s*\\n3\\. Complete the " | Address `g4f/tools/optimize_request.py:329` without breaking the configured checks. | 2026-10-31T05:59:39Z |
| G4F-010 | gpt4free | P3 | pending | Resolve TODO in optimize_request.py: \\s*\\n4\\. Mark that todo as completed IMMEDIATELY\\s*\\n5\\. " | Address `g4f/tools/optimize_request.py:330` without breaking the configured checks. | 2026-10-31T05:51:24Z |
| G4F-011 | gpt4free | P3 | pending | Resolve TODO in optimize_request.py: and repeat" | Address `g4f/tools/optimize_request.py:331` without breaking the configured checks. | 2026-10-31T05:51:24Z |
| G4F-012 | gpt4free | P3 | pending | Resolve TODO in test_optimize_request.py: ", "to-do", "to do"): | Address `etc/unittest/test_optimize_request.py:33` without breaking the configured checks. | 2026-10-31T05:58:21Z |

### Free model health

Checked `2026-10-31T05:45:31Z` at `http://127.0.0.1:28180/v1`.

| Model | Status | Latency | Detail |
| --- | --- | --- | --- |
| cohere/north-mini-code:free | working | 3.84s | OK |
| ovh-free/gpt-oss-120b | working | 1.09s | OK |
| deepseek-v4.1-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| glm-5.3-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| mimo-v2.6-flash-free | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| big-pickle | failed | - | HTTP 429 from http://127.0.0.1:28180/v1/chat/completions: {"type":"error","error":{"type":"FreeUsageLimitError","mess... |
| ovh-free/Qwen3.6-27B | failed | - | HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {"error":"upstream error","detail":"upstream returned an em... |
| ovh-free/Qwen3.5-397B-A17B | failed | - | HTTP 502 from http://127.0.0.1:28180/v1/chat/completions: {"error":"upstream error","detail":"upstream returned an em... |

### Recent runs

| When | Task | Project | Model | Result | Detail |
| --- | --- | --- | --- | --- | --- |
| 2026-10-31T06:09:09Z | G4F-015 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T06:06:35Z | G4F-014 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T06:03:56Z | G4F-013 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:59:39Z | G4F-009 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:57:27Z | G4F-008 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:55:38Z | G4F-007 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:54:37Z | G4F-006 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:53:04Z | G4F-005 | gpt4free | pollinations/deepseek/deepseek-v4.1-flash-1017 | success | Completed with pollinations/deepseek/deepseek-v4.1-flash-1017; checks passed. |
| 2026-10-31T05:23:35Z | G4F-004 | gpt4free | nvidia/nemotron-3.5-lightning:free | success | Completed with nvidia/nemotron-3.5-lightning:free; checks passed. |
| 2026-10-31T05:05:27Z | G4F-003 | gpt4free | nvidia/nemotron-3.5-lightning:free | success | Completed with nvidia/nemotron-3.5-lightning:free; checks passed. |

<!-- DEV-PIPELINE:BOARD:END -->
