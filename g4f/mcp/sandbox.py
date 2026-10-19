"""Sandbox / container detection for the g4f MCP server.

The MCP server exposes powerful tools: sandboxed Python execution, workspace
file management and — on a normal host install — a shell / terminal command
execution tool.  Whether those tools may exist depends on the environment the
server runs in:

- **Sandboxed / containerized** (Docker, Podman, Kubernetes, LXC, Flatpak,
  Snap, …): the deployment is typically shared or publicly reachable, so the
  server switches to *safe mode* automatically and the shell/terminal
  execution tool stays **disabled**.
- **Normal host install**: no sandbox detected, safe mode stays off and the
  ``shell_execute`` tool is **activated**.

All detection is passive (marker files, ``/proc`` cgroup data and well-known
environment variables) — no subprocesses are spawned.

Environment overrides:

- ``G4F_SANDBOX=1|0``       — force the sandbox detection result
- ``G4F_MCP_SAFE_MODE=1|0`` — force safe mode on / off
- ``G4F_MCP_SHELL=1|0``     — force the ``shell_execute`` tool on / off
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

#: Marker files that indicate a container / OS-level sandbox.
SANDBOX_MARKER_FILES: Tuple[str, ...] = (
    "/.dockerenv",  # Docker
    "/run/.containerenv",  # Podman
    "/.flatpak-info",  # Flatpak
)

#: Substrings looked for in /proc cgroup data.
CGROUP_HINTS: Tuple[str, ...] = (
    "docker",
    "containerd",
    "kubepods",
    "kubecri",
    "lxc",
    "podman",
    "garden",  # Cloud Foundry containers
    "scaleway-container",
    "libpod",
    "buildkit",
)

#: Environment variables whose presence indicates a sandbox / container.
SANDBOX_ENV_VARS: Tuple[str, ...] = (
    "container",  # OCI convention: container=docker|podman|oci
    "KUBERNETES_SERVICE_HOST",  # Kubernetes pod
    "SNAP",  # Snap package sandbox
    "SNAP_NAME",
    "FLATPAK_ID",  # Flatpak sandbox
)

#: Cached detection result (``None`` until the first call).
_sandboxed: Optional[bool] = None
_sandbox_reasons: List[str] = []


def _env_flag(name: str) -> Optional[bool]:
    """Return the boolean value of an environment override, or ``None``."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None


def _read_text(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            return file.read(limit)
    except OSError:
        return ""


def get_sandbox_reasons(refresh: bool = False) -> List[str]:
    """Return human-readable reasons why the environment counts as sandboxed.

    Returns an empty list when no sandbox was detected.  The result is cached;
    pass ``refresh=True`` to re-scan.
    """
    global _sandboxed, _sandbox_reasons

    override = _env_flag("G4F_SANDBOX")
    if override is not None:
        return ["G4F_SANDBOX override"] if override else []

    if _sandboxed is not None and not refresh:
        return list(_sandbox_reasons)

    reasons: List[str] = []

    for marker in SANDBOX_MARKER_FILES:
        if os.path.exists(marker):
            reasons.append(f"marker file {marker}")

    for var in SANDBOX_ENV_VARS:
        if os.environ.get(var):
            reasons.append(f"environment variable {var}")

    for cgroup_path in ("/proc/self/cgroup", "/proc/1/cgroup"):
        content = _read_text(cgroup_path).lower()
        if not content:
            continue
        for hint in CGROUP_HINTS:
            if hint in content:
                reasons.append(f"cgroup hint '{hint}' in {cgroup_path}")
                break

    # De-duplicate while preserving order (self and pid 1 cgroups often match).
    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    _sandboxed = bool(deduped)
    _sandbox_reasons = deduped
    return list(deduped)


def detect_sandbox(refresh: bool = False) -> bool:
    """Return ``True`` when the process runs inside a sandbox / container.

    Detection is passive and cached.  ``G4F_SANDBOX`` overrides the result.
    """
    override = _env_flag("G4F_SANDBOX")
    if override is not None:
        return override
    if _sandboxed is None or refresh:
        get_sandbox_reasons(refresh=refresh)
    return bool(_sandboxed)


def safe_mode_enabled() -> bool:
    """Return ``True`` when the MCP server should run in safe mode.

    Safe mode is auto-enabled inside a detected sandbox; ``G4F_MCP_SAFE_MODE``
    overrides the automatic decision.
    """
    override = _env_flag("G4F_MCP_SAFE_MODE")
    if override is not None:
        return override
    return detect_sandbox()


def shell_tool_enabled() -> bool:
    """Return ``True`` when the shell / terminal execution tool may be active.

    The ``shell_execute`` tool is only activated outside sandboxes (normal
    host installs); ``G4F_MCP_SHELL`` overrides the automatic decision.
    """
    override = _env_flag("G4F_MCP_SHELL")
    if override is not None:
        return override
    return not safe_mode_enabled()
