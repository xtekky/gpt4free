"""PA Provider Downloader

Downloads ``*.pa.py`` files from a GitHub repository (default:
``gpt4free/pa-providers``) into the local g4f workspace directory
(``~/.g4f/workspace``).

Public API
----------
- :func:`run_pa_download`  — download (or update) PA providers from GitHub.
- :func:`run_pa_list`      — list installed PA providers in the workspace.
- :func:`run_pa_remove`    — remove a PA provider by filename.
- :func:`auto_download_pa_providers` — best-effort startup auto-download.

The downloader uses the public GitHub REST API
(https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<ref>)
which does not require authentication for public repositories.  It respects
the ``G4F_PROXY`` environment variable and is fully network-failure tolerant:
any error is logged via :mod:`g4f.debug` and never raised, so a failed
download can never break server startup.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .. import debug
from .pa_provider import get_workspace_dir, list_pa_providers


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default GitHub repository (owner/repo) to download PA providers from.
DEFAULT_REPO = "gpt4free/pa-providers"

#: Default git ref (branch / tag / commit) to download from.
DEFAULT_REF = "main"

#: Public GitHub REST API root.
GITHUB_API = "https://api.github.com"

#: Per-request timeout (seconds) for GitHub API calls and raw downloads.
DEFAULT_TIMEOUT: float = 30.0

#: Marker file written into the workspace after a successful auto-download.
#: Its mtime is used to decide when the next auto-download is allowed, so we
#: do not hit GitHub on every single server start.
AUTO_DOWNLOAD_MARKER = ".pa_auto_downloaded"

#: Minimum seconds between two automatic downloads.
AUTO_DOWNLOAD_INTERVAL: float = 6 * 60 * 60  # 6 hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_proxy() -> Optional[str]:
    """Return the configured proxy URL, if any."""
    return os.environ.get("G4F_PROXY") or os.environ.get("HTTPS_PROXY") or None


def _github_request(url: str, timeout: float, accept: str = "application/json") -> bytes:
    """Perform an HTTP GET against *url* and return the raw body bytes.

    Raises:
        URLError / HTTPError: on network or HTTP failure.
    """
    headers = {
        "Accept": accept,
        "User-Agent": "g4f-pa-downloader/1.0",
    }
    req = Request(url, headers=headers)
    kwargs = {"timeout": timeout}
    proxy = _get_proxy()
    if proxy:
        # urllib supports HTTPS proxies via environment variables only; set them
        # for the duration of the call.
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("HTTP_PROXY", proxy)
    return urlopen(req, **kwargs).read()


def _list_repo_files(repo: str, ref: str, timeout: float) -> List[str]:
    """Return the list of ``*.pa.py`` paths in the root of *repo* at *ref*.

    Uses the GitHub contents API.  Subdirectories are not recursed — the
    pa-providers repo is flat by convention.
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/?ref={ref}"
    body = _github_request(url, timeout, accept="application/vnd.github+json")
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, list):
        return []
    files: List[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        if name.endswith(".pa.py") and entry.get("type") == "file":
            files.append(name)
    return files


def _download_raw(repo: str, ref: str, name: str, timeout: float) -> bytes:
    """Download a single file's raw content from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{name}"
    return _github_request(url, timeout, accept="text/plain")


def _workspace_target(directory: Optional[str] = None) -> Path:
    """Resolve the target workspace directory, creating it if needed."""
    target = Path(directory).expanduser() if directory else (get_workspace_dir() / "pa-providers")
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pa_download(
    repo: str = DEFAULT_REPO,
    ref: str = DEFAULT_REF,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    directory: Optional[str] = None,
    only: Optional[str] = None,
) -> List[Path]:
    """Download ``*.pa.py`` files from *repo*/*ref* into the workspace.

    Args:
        repo:  ``owner/repo`` GitHub repository (default: ``gpt4free/pa-providers``).
        ref:   Branch / tag / commit (default: ``main``).
        force: If ``True``, overwrite existing files.  If ``False``, only
               download files that do not already exist locally.
        timeout: Per-request timeout in seconds.
        directory: Override target directory (default: ``~/.g4f/workspace``).
        only: If given, download only this filename (e.g. ``koala.pa.py``).

    Returns:
        List of paths that were downloaded (and written) this call.
    """
    target = _workspace_target(directory)
    written: List[Path] = []

    try:
        if only:
            names = [only]
        else:
            names = _list_repo_files(repo, ref, timeout)
            debug.log(f"pa-providers: found {len(names)} file(s) in {repo}@{ref}")
    except (URLError, HTTPError, ValueError, OSError) as e:
        debug.error(f"pa-providers: failed to list repository {repo}@{ref}:", e)
        return written

    for name in names:
        if not name.endswith(".pa.py"):
            continue
        dest = target / name
        if dest.exists() and not force:
            debug.log(f"pa-providers: skip existing {name}")
            continue
        try:
            content = _download_raw(repo, ref, name, timeout)
        except (URLError, HTTPError, OSError) as e:
            debug.error(f"pa-providers: failed to download {name}:", e)
            continue
        try:
            dest.write_bytes(content)
            written.append(dest)
            print(f"pa-providers: downloaded {name} -> {dest}")
        except OSError as e:
            debug.error(f"pa-providers: failed to write {name}:", e)

    if written:
        # Touch the auto-download marker so the startup path does not re-download
        # immediately after an explicit `g4f pa download`.
        try:
            (target / AUTO_DOWNLOAD_MARKER).touch()
        except OSError:
            pass

    return written


def run_pa_list(directory: Optional[str] = None) -> List[Path]:
    """Print and return the list of installed PA providers."""
    target = _workspace_target(directory)
    _, pa_paths = list_pa_providers(target)
    if not pa_paths:
        print(f"No PA providers installed in {target}")
        return []
    print(f"PA providers in {target}:")
    for p in pa_paths:
        try:
            rel = p.relative_to(target)
        except ValueError:
            rel = p
        print(f"  - {rel}")
    return pa_paths


def run_pa_remove(filename: str, directory: Optional[str] = None) -> bool:
    """Remove a PA provider by filename.  Returns ``True`` on success."""
    if not filename:
        print("Error: filename is required.")
        return False
    target = _workspace_target(directory)
    # Resolve safely inside the workspace.
    candidate = (target / filename).resolve()
    try:
        candidate.relative_to(target.resolve())
    except ValueError:
        print(f"Error: {filename} escapes the workspace directory.")
        return False
    if not candidate.name.endswith(".pa.py"):
        print(f"Error: {filename} is not a .pa.py file.")
        return False
    if not candidate.exists():
        print(f"Error: {filename} not found in {target}")
        return False
    try:
        candidate.unlink()
        print(f"Removed {candidate}")
        return True
    except OSError as e:
        print(f"Error removing {filename}: {e}")
        return False


# ---------------------------------------------------------------------------
# Startup auto-download
# ---------------------------------------------------------------------------

def _should_auto_download(workspace: Path) -> bool:
    """Return ``True`` if an automatic download should run now."""
    marker = workspace / AUTO_DOWNLOAD_MARKER
    if not marker.exists():
        return True
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        return True
    return age >= AUTO_DOWNLOAD_INTERVAL


def auto_download_pa_providers(
    repo: str = DEFAULT_REPO,
    ref: str = DEFAULT_REF,
    timeout: float = DEFAULT_TIMEOUT,
    force: bool = False,
) -> List[Path]:
    """Best-effort automatic download of PA providers.

    Designed to be called from server startup.  Never raises — all errors are
    logged via :mod:`g4f.debug` and swallowed so a network failure can never
    prevent the server from starting.

    When *force* is ``False`` (the default), the download is skipped if the
    auto-download marker is fresher than :data:`AUTO_DOWNLOAD_INTERVAL`.
    """
    try:
        workspace = get_workspace_dir()
    except Exception as e:
        debug.error("pa-providers: workspace unavailable, skipping auto-download:", e)
        return []

    if not force and not _should_auto_download(workspace):
        debug.log("pa-providers: auto-download skipped (recent)")
        return []

    debug.log(f"pa-providers: auto-downloading from {repo}@{ref}")
    try:
        written = run_pa_download(repo=repo, ref=ref, force=force, timeout=timeout)
    except Exception as e:
        # run_pa_download already swallows most errors, but be defensive.
        debug.error("pa-providers: auto-download failed:", e)
        return []

    if written:
        print(f"pa-providers: auto-downloaded {len(written)} provider(s) from {repo}")
    else:
        debug.log("pa-providers: auto-download found nothing new to install")
    return written