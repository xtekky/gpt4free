"""Serve historical GUI snapshots under ``/v<version>/``.

Each g4f release is tagged in ``xtekky/gpt4free`` while the GUI itself lives in
``gpt4free/g4f.dev``. The latter has no per-release tags, so a version is mapped
to the first ``g4f.dev`` commit at or after the release date. This keeps the
HTML, JavaScript, CSS, and images of an old GUI together instead of mixing old
markup with current assets.

The resolver prefers the local ``g4f.dev`` checkout (useful for development and
offline tests) and falls back to the public GitHub API and jsDelivr/raw GitHub
for deployments such as ``g4f.space`` where no repository is checked out.
"""

from __future__ import annotations

import mimetypes
import os
import posixpath
import re
import subprocess
import time
from urllib.parse import quote, unquote, urlsplit

import requests

from ...config import ORGANIZATION, STATIC_DOMAIN, get_cache_dir
from ...tools.files import secure_filename

GITHUB_API = "https://api.github.com"
SITE_REPOSITORY = f"{ORGANIZATION}/{STATIC_DOMAIN}"
SOURCE_REPOSITORY = f"xtekky/gpt4free"
_JSDELIVR_BASE = f"https://cdn.jsdelivr.net/gh/{SITE_REPOSITORY}"
_RAW_BASE = f"https://raw.githubusercontent.com/{SITE_REPOSITORY}"

# ``v1`` and ``v1.2`` are accepted as well, matching the tags that have existed
# in the wild. The version is only used after strict validation, so it has to
# stay a plain release identifier.
_VERSION_RE = re.compile(
    r"^v?(?P<version>\d+(?:\.\d+){0,3}(?:[-+][0-9A-Za-z.-]+)?)$"
)

_REF_CACHE: dict[str, tuple[float, str]] = {}
_RELEASE_DATE_CACHE: dict[str, tuple[float, str]] = {}
_REF_TTL = 3600.0
_RELEASE_DATE_TTL = 86400.0

_SESSION = requests.Session()

# Root-relative URLs owned by the static site. Everything else (the backend
# API, auth, screenshots, ...) must keep pointing at the live server so an old
# GUI can still talk to the current backend.
_EXCLUDED_URL_PREFIXES = (
    "/backend-api",
    "/v1",
    "/v2",
    "/api",
    "/auth",
    "/screenshot",
    "/images",
    "/media",
    "/thumbnail",
    "/.well-known",
    "/cdn-cgi",
    "/v",
)
_URL_ATTR_RE = re.compile(
    r"(\s(?:href|src|action|poster|data-src)\s*=\s*)([\"'])(/[^\"']*)\2",
    re.IGNORECASE,
)


class VersionedError(Exception):
    """Raised when a version or one of its files cannot be resolved."""


def normalize_version(value: str) -> str:
    """Validate a URL version and return it without a leading ``v``."""
    match = _VERSION_RE.fullmatch((value or "").strip())
    if not match:
        raise ValueError(f"Invalid version: {value!r}")
    version = match.group("version")
    if version.startswith("v"):
        version = version[1:]
    return version


def normalize_site_path(filename: str) -> str:
    """Return a safe repository-relative path or raise ``VersionedError``."""
    filename = unquote((filename or "").replace("\\", "/").strip())
    filename = filename.lstrip("/")
    if not filename:
        raise VersionedError("Missing file name")
    if "\x00" in filename:
        raise VersionedError("Invalid file name")
    # ``posixpath.normpath`` would also collapse ``a/../b`` into ``b``. Reject
    # traversal explicitly so a crafted URL can never leave the repository.
    parts = [part for part in filename.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise VersionedError("Invalid file name")
    return "/".join(parts)


def request_candidates(filename: str) -> list[str]:
    """Map a GUI URL path to the files that may satisfy it."""
    filename = (filename or "").replace("\\", "/").strip()
    is_index = filename.endswith("/")
    filename = filename.strip("/")
    if not filename:
        return ["home.html"]
    if is_index:
        return [normalize_site_path(filename + "/index.html")]
    if "." in posixpath.basename(filename):
        return [normalize_site_path(filename)]
    # Extension-less page URLs: try a directory index first, then ``.html``.
    return [
        normalize_site_path(filename + "/index.html"),
        normalize_site_path(filename + ".html"),
    ]


def _repository_root() -> str:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def _site_checkout() -> str | None:
    candidate = os.path.join(_repository_root(), STATIC_DOMAIN)
    if os.path.isdir(os.path.join(candidate, ".git")):
        return candidate
    return None


def _git(repository: str, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", repository, *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _github_get(path: str) -> tuple[object | None, dict]:
    """Return the JSON payload and pagination links for a GitHub API path."""
    url = (
        path
        if path.startswith(("http://", "https://"))
        else f"{GITHUB_API}{path}"
    )
    try:
        response = _SESSION.get(
            url,
            timeout=10,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status_code != 200:
            return None, {}
        return response.json(), response.links
    except (requests.RequestException, ValueError):
        return None, {}


def _github_json(path: str) -> object | None:
    return _github_get(path)[0]


def release_date(version: str) -> str | None:
    """Return an ISO timestamp for the g4f release ``version``."""
    version = normalize_version(version)
    now = time.time()
    cached = _RELEASE_DATE_CACHE.get(version)
    if cached and now - cached[0] < _RELEASE_DATE_TTL:
        return cached[1]

    tag = f"v{version}"
    date = _git(_repository_root(), "log", "-1", "--format=%cI", tag)

    if not date:
        release = _github_json(f"/repos/{SOURCE_REPOSITORY}/releases/tags/{quote(tag)}")
        if isinstance(release, dict):
            date = release.get("published_at") or release.get("created_at")

    if not date:
        tag_data = _github_json(f"/repos/{SOURCE_REPOSITORY}/git/ref/tags/{quote(tag)}")
        sha = None
        if isinstance(tag_data, dict):
            target = tag_data.get("object") or {}
            sha = target.get("sha")
            if target.get("type") == "tag":
                tag_object = _github_json(
                    f"/repos/{SOURCE_REPOSITORY}/git/tags/{sha}"
                )
                if isinstance(tag_object, dict):
                    sha = (tag_object.get("object") or {}).get("sha") or sha
        if sha:
            commit = _github_json(f"/repos/{SOURCE_REPOSITORY}/commits/{sha}")
            if isinstance(commit, dict):
                date = (
                    (commit.get("commit") or {}).get("committer") or {}
                ).get("date")

    if not date:
        return None

    _RELEASE_DATE_CACHE[version] = (now, date)
    return date


def _resolve_local_ref(date: str) -> str | None:
    repository = _site_checkout()
    if not repository:
        return None
    # ``--reverse`` makes the first line the earliest commit at/after the
    # release date, which is the snapshot this release shipped with.
    output = _git(
        repository,
        "log",
        "--reverse",
        "--format=%H",
        f"--since={date}",
    )
    if output:
        return output.splitlines()[0]
    # A release newer than the checkout has no later commit. Fall back to the
    # newest commit at/before the release date.
    return _git(repository, "log", "-1", "--format=%H", f"--until={date}")


def _resolve_remote_ref(date: str) -> str | None:
    per_page = 100
    path = f"/repos/{SITE_REPOSITORY}/commits?since={quote(date)}&per_page={per_page}"
    commits, links = _github_get(path)
    if isinstance(commits, list) and commits:
        # GitHub returns commits newest first. The snapshot we want is the
        # earliest one at/after the release date, so when the result spans
        # several pages we have to read the *last* page instead of the first.
        oldest = commits
        last_url = (links.get("last") or {}).get("url")
        if last_url:
            last_commits, _ = _github_get(last_url)
            if isinstance(last_commits, list) and last_commits:
                oldest = last_commits
        elif len(commits) >= per_page:
            # Some proxies strip pagination headers; walk forward instead.
            page = 2
            while (
                isinstance(commits, list)
                and len(commits) >= per_page
                and page <= 100
            ):
                commits, _ = _github_get(f"{path}&page={page}")
                if isinstance(commits, list) and commits:
                    oldest = commits
                page += 1
        return oldest[-1].get("sha")
    commits = _github_json(
        f"/repos/{SITE_REPOSITORY}/commits?until={quote(date)}&per_page=1"
    )
    if isinstance(commits, list) and commits:
        return commits[0].get("sha")
    return None


def resolve_site_ref(version: str) -> str:
    """Return the ``g4f.dev`` commit used to render ``version``."""
    version = normalize_version(version)
    now = time.time()
    cached = _REF_CACHE.get(version)
    if cached and now - cached[0] < _REF_TTL:
        return cached[1]

    date = release_date(version)
    if not date:
        raise VersionedError(f"Unknown release: {version}")

    sha = _resolve_local_ref(date) or _resolve_remote_ref(date)
    if not sha:
        raise VersionedError(f"No GUI snapshot for release: {version}")

    _REF_CACHE[version] = (now, sha)
    return sha


def _mimetype(filename: str) -> str:
    path = filename.lower()
    if path.endswith(".webmanifest"):
        return "application/manifest+json"
    if path.endswith((".js", ".mjs")):
        return "text/javascript"
    if path.endswith(".css"):
        return "text/css"
    if path.endswith(".svg"):
        return "image/svg+xml"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".wasm"):
        return "application/wasm"
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _read_local_file(sha: str, filename: str) -> bytes | None:
    repository = _site_checkout()
    if not repository:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", repository, "show", f"{sha}:{filename}"],
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _read_remote_file(sha: str, filename: str) -> bytes | None:
    encoded = quote(filename)
    urls = (
        f"{_JSDELIVR_BASE}@{sha}/{encoded}",
        f"{_RAW_BASE}/{sha}/{encoded}",
    )
    for url in urls:
        try:
            response = _SESSION.get(url, timeout=20)
            if response.status_code == 200:
                return response.content
        except requests.RequestException:
            continue
    return None


def _cache_path(sha: str, filename: str, version: str) -> str:
    # ``filename`` is already normalised and cannot contain
    return os.path.join(get_cache_dir(), "gui_versions", version, sha, secure_filename(filename))


def load_versioned_file(version: str, filename: str) -> tuple[bytes, str, str]:
    """Load a file from the snapshot for ``version``.

    Returns ``(content, mimetype, commit)``. Content is cached on disk so the
    same asset is not fetched on every request.
    """
    filename = normalize_site_path(filename)
    sha = resolve_site_ref(version)
    path = _cache_path(sha, filename, version)

    if os.path.isfile(path):
        try:
            with open(path, "rb") as file:
                return file.read(), _mimetype(filename), sha
        except OSError:
            pass

    content = _read_local_file(sha, filename)
    if content is None:
        content = _read_remote_file(sha, filename)
    if content is None:
        raise VersionedError(f"File not found: {filename}")

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as file:
            file.write(content)
        os.replace(temporary, path)
    except OSError:
        # A read-only cache directory must not make the route fail.
        pass

    return content, _mimetype(filename), sha


def _versioned_url(value: str, version: str) -> str:
    if not value.startswith("/") or value.startswith("//"):
        return value
    path = urlsplit(value).path
    if path == f"/v{version}" or path.startswith(f"/v{version}/"):
        return value
    if any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in _EXCLUDED_URL_PREFIXES
    ):
        return value
    return f"/v{version}{value}"


def rewrite_versioned_html(html: str, version: str) -> str:
    """Keep root-relative links and assets inside the versioned snapshot."""
    html = html.replace("{{ v }}", f"v{version}")
    # Older pages use ../dist/ in attributes and inline module imports. Make
    # them absolute so /v7.0.0/chat and /v7.0.0/chat/ behave alike.
    html = re.sub(
        r"([\"'])\.\./dist/",
        lambda match: match.group(1) + f"/v{version}/dist/",
        html,
    )
    # Inline ES module imports are not HTML attributes.
    html = re.sub(
        r"([\"'])/dist/",
        lambda match: match.group(1) + f"/v{version}/dist/",
        html,
    )
    html = _URL_ATTR_RE.sub(
        lambda match: (
            match.group(1)
            + match.group(2)
            + _versioned_url(match.group(3), version)
            + match.group(2)
        ),
        html,
    )
    marker = f'<meta name="g4f-version" content="v{version}">'
    if marker not in html:
        html = re.sub(
            r"(<head\b[^>]*>)",
            lambda match: match.group(1) + "\n    " + marker,
            html,
            count=1,
            flags=re.IGNORECASE,
        )
        if marker not in html:
            html = marker + html
    return html
