"""Install-and-run hook for the sandbox apps behind the domain router.

The sandbox desktop image ships ``sandbox-cli`` (see ``docker/sandbox-menu/``),
which installs a tool on first use and then runs it. The domain router
(:mod:`g4f.api.domain_router`) forwards ``<app>.g4f.local`` to the app's local
port, so opening a hostname for an app that was never installed would
otherwise only show the proxy's 502 "is not running".

This module closes that gap: when a routed app is not reachable, the proxy
asks :func:`ensure_app_running` to start it — installing it first when needed —
and answers with a 503 "starting" payload that tells the browser to retry until
the app is up. On the sandbox desktop the CLI runs in a visible terminal
window, so the install and the server console can be watched and closing the
window stops the app; without an X display it runs detached with its output in
a log file.

Only apps that ``sandbox-cli`` knows how to install and run are started, and
only when the CLI is present, so the hook is inert outside the sandbox image.
Set ``G4F_DOMAIN_AUTOSTART=0`` to disable it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Callable, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

#: Where the sandbox image installs the CLI (see docker/Dockerfile-sandbox-desktop).
CLI_PATH = "/usr/local/bin/sandbox-cli"
CLI_NAME = "sandbox-cli"
#: Desktop terminal used to show the install & run flow on the sandbox desktop.
TERMINAL_NAME = "xfce4-terminal"

#: How long a request waits for the app to come up before answering "starting".
START_TIMEOUT = 5.0
#: Seconds the browser should wait before retrying a "starting" response.
RETRY_AFTER = 3
#: TCP connect timeout for the reachability probe.
PROBE_TIMEOUT = 0.25

#: Routed apps that ``sandbox-cli`` can install and run, i.e. the ``run_tool``
#: cases in bin/sandbox-cli. The route names match the tool
#: names; ``g4f`` (the API itself) and ``desktop`` (the noVNC client, part of
#: the image) have no tool and are never started.
_TOOLS = frozenset(
    {
        "omniroute",
        "ollama",
        "litellm",
        "librechat",
        "sillytavern",
        "cptr",
        "code-server",
        "assistant-ui",
        "llamacpp",
        "lmstudio",
        "pi-freeflow",
        "remote-desktop",
    }
)

#: In-flight ``sandbox-cli`` processes, keyed by app name. A server app runs in
#: the foreground, so the process stays alive while the app does; a GUI app
#: exits right after starting the app detached.
_processes: Dict[str, subprocess.Popen] = {}

def autostart_enabled() -> bool:
    """Return whether apps are started on demand (``G4F_DOMAIN_AUTOSTART``)."""
    return os.environ.get("G4F_DOMAIN_AUTOSTART", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

def cli_path() -> Optional[str]:
    """Return the path of ``sandbox-cli``, or ``None`` when it is not installed."""
    if os.path.isfile(CLI_PATH) and os.access(CLI_PATH, os.X_OK):
        return CLI_PATH
    return shutil.which(CLI_NAME)

def tool_for(name: str) -> Optional[str]:
    """Return the ``sandbox-cli`` tool that serves ``name``, if there is one."""
    return name if name in _TOOLS else None

def is_running(root_url: str) -> bool:
    """Return whether something accepts TCP connections at ``root_url``."""
    parsed = urlsplit(root_url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), PROBE_TIMEOUT):
            return True
    except OSError:
        return False

def is_starting(name: str) -> bool:
    """Return whether an install/start attempt for ``name`` is still running."""
    process = _processes.get(name)
    if process is None:
        return False
    if process.poll() is None:
        return True
    _processes.pop(name, None)
    return False

def terminal_argv(path: str, tool: str, label: str) -> Optional[List[str]]:
    """Return the desktop-terminal argv for ``path tool``, if one is available.

    Auto-started apps used to run detached with their output in a log file, so
    neither the install nor the server console was visible on the sandbox
    desktop and there was no window to close. When the desktop is up (an X
    display and ``xfce4-terminal``), the CLI runs in a visible terminal window
    instead: the install and the app's console stay on screen, and closing the
    window stops the app.

    ``--disable-server`` keeps xfce4-terminal from handing the window to an
    already running instance. The spawned process must stay alive for as long
    as the window does, so :func:`is_starting` keeps tracking the install.
    """
    if not os.environ.get("DISPLAY"):
        return None
    terminal = shutil.which(TERMINAL_NAME)
    if terminal is None:
        return None
    return [
        terminal,
        "--disable-server",
        "--maximize",
        "--hide-borders",
        "--hide-menubar",
        f"--title={label}",
        f"--icon=g4f-sandbox-{tool}",
        f"--command={path} {tool}",
    ]

def start_app(name: str, label: str) -> bool:
    """Start ``sandbox-cli <name>`` in a terminal window; return whether it was spawned.

    The CLI installs the tool when it is missing and then runs it, so this is
    the whole "install & run" flow. On the sandbox desktop it is spawned in a
    visible terminal window, so the user can watch the install and close the
    window to stop the app. Without a desktop (or without xfce4-terminal) it
    falls back to a detached process with its output in a log file. Either way
    it runs in its own session, so it outlives the request and never blocks it.
    """
    tool = tool_for(name)
    path = cli_path()
    if tool is None or path is None:
        return False
    if is_starting(name):
        return False
    log_path = os.path.join(tempfile.gettempdir(), f"g4f-autostart-{name}.log")
    try:
        log = open(log_path, "ab")
    except OSError as e:
        logger.warning("Autostart: cannot open %s: %s", log_path, e)
        return False
    argv = terminal_argv(path, tool, label)
    on_desktop = argv is not None
    if argv is None:
        argv = [path, tool]
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=os.path.expanduser("~"),
        )
    except OSError as e:
        logger.warning("Autostart: cannot run %s %s: %s", path, tool, e)
        return False
    finally:
        log.close()
    _processes[name] = process
    logger.info(
        "Autostart: %s is not running, started \"%s %s\" %s (pid %s, log %s)",
        label,
        path,
        tool,
        "in a desktop terminal" if on_desktop else "detached",
        process.pid,
        log_path,
    )
    return True

async def ensure_app_running(
    name: str,
    label: str,
    root_url: Callable[[], str],
) -> bool:
    """Start the app behind ``name`` when it is down; return whether it is up.

    Returns ``True`` as soon as the app accepts connections, so the caller can
    retry the request it was serving. Returns ``False`` when the app is not
    startable (autostart off, unknown tool, no CLI) or did not come up within
    :data:`START_TIMEOUT` — the caller then answers "starting" while the
    install keeps running in the terminal window (or in the background).
    """
    if is_running(root_url()):
        return True
    if not autostart_enabled() or tool_for(name) is None:
        return False
    if not is_starting(name) and not start_app(name, label):
        # No sandbox-cli here (or it could not be spawned): nothing to wait for.
        return False
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        if is_running(root_url()):
            return True
    return False

def starting_response(name: str, label: str) -> JSONResponse:
    """Return the 503 payload shown while an app is being installed and started."""
    tool = tool_for(name) or name
    return JSONResponse(
        {
            "error": {
                "message": f"{label} is starting",
                "type": f"{name}_starting",
                "detail": (
                    f"\"sandbox-cli {tool}\" is installing and starting {label} "
                    f"on the desktop; retry in a few seconds."
                ),
            }
        },
        status_code=503,
        headers={"retry-after": str(RETRY_AFTER)},
    )