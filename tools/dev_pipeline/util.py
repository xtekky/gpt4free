"""Small shared helpers for the development pipeline."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def truncate(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


def one_line(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "task"


@dataclass
class CommandResult:
    command: List[str]
    exit_code: int
    output: str
    duration: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "command": " ".join(self.command),
            "exit_code": self.exit_code,
            "duration_s": round(self.duration, 2),
            "output": truncate(self.output, 2500),
        }


def run_command(
    command: Sequence[str],
    cwd: Path,
    timeout: int,
    extra_env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
) -> CommandResult:
    env = os.environ.copy()
    # Make the checkout win over an older installed copy of g4f.
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(cwd) if not python_path else str(cwd) + os.pathsep + python_path
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = completed.stdout or ""
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        output += f"\n[dev-pipeline] command timed out after {timeout}s"
        code = 124
    except FileNotFoundError as exc:
        output = f"[dev-pipeline] executable not found: {exc}"
        code = 127
    except OSError as exc:
        output = f"[dev-pipeline] could not run command: {exc}"
        code = 126
    return CommandResult(
        command=list(command),
        exit_code=code,
        output=truncate(output),
        duration=time.monotonic() - started,
    )


def run_command_streaming(
    command: Sequence[str],
    cwd: Path,
    timeout: int,
    extra_env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
    on_line: Optional[Callable[[str], None]] = None,
) -> CommandResult:
    """Run a command and forward each stdout line to ``on_line`` as it arrives.

    Unlike :func:`run_command`, stdout is consumed continuously so a chatty
    agent can report progress without waiting for the whole process to finish.
    The accumulated output is still returned for the caller.
    """
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(cwd) if not python_path else str(cwd) + os.pathsep + python_path
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra_env:
        env.update(extra_env)

    started = time.monotonic()
    output_parts: List[str] = []
    timed_out = False

    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=list(command),
            exit_code=127,
            output=f"[dev-pipeline] executable not found: {exc}",
            duration=time.monotonic() - started,
        )
    except OSError as exc:
        return CommandResult(
            command=list(command),
            exit_code=126,
            output=f"[dev-pipeline] could not run command: {exc}",
            duration=time.monotonic() - started,
        )

    def watchdog() -> None:
        nonlocal timed_out
        timed_out = True
        process.kill()

    timer = threading.Timer(timeout, watchdog)
    timer.start()
    try:
        if input_text is not None and process.stdin is not None:
            process.stdin.write(input_text)
            process.stdin.close()
        assert process.stdout is not None
        for line in process.stdout:
            output_parts.append(line)
            if on_line is not None:
                try:
                    on_line(line.rstrip("\r\n"))
                except Exception:
                    pass
        code = process.wait()
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()

    output = "".join(output_parts)
    if timed_out:
        output += f"\n[dev-pipeline] command timed out after {timeout}s"
        code = 124
    return CommandResult(
        command=list(command),
        exit_code=code,
        output=truncate(output),
        duration=time.monotonic() - started,
    )


class PipelineError(RuntimeError):
    """Base error for pipeline failures."""


class LockError(PipelineError):
    """Raised when another pipeline run owns the lock."""


@contextlib.contextmanager
def file_lock(path: Path, timeout: int = 0):
    """Advisory lock so two pipeline loops cannot edit AGENTS.md at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if timeout and time.monotonic() < deadline:
                        time.sleep(0.25)
                        continue
                    raise LockError(
                        f"another dev-pipeline run holds {path}; wait for it to finish"
                    )
        except ImportError:  # pragma: no cover - Windows fallback
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def ensure_state_dir(config) -> None:
    state_dir: Path = config.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    ignore = state_dir / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n!.gitignore\n", encoding="utf-8")
