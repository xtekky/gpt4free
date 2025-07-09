"""Git integration utilities."""

import subprocess
from pathlib import Path
from typing import Tuple, Optional
import datetime

class GitIntegration:
    @staticmethod
    def _run_git_command(command: list) -> Tuple[bool, str]:
        """Helper to run a git command and handle exceptions."""
        try:
            result = subprocess.run(["git"] + command, capture_output=True, text=True, check=True)
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, (e.stdout + e.stderr).strip()
        except FileNotFoundError:
            return False, "Git command not found. Is Git installed and in your PATH?"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def is_git_repo() -> bool:
        return Path(".git").is_dir()

    @staticmethod
    def get_status() -> str:
        ok, output = GitIntegration._run_git_command(["status", "--porcelain"])
        return output if ok else ""

    @staticmethod
    def get_staged_diff() -> str:
        ok, output = GitIntegration._run_git_command(["diff", "--cached", "--name-only"])
        return output if ok else ""

    @staticmethod
    def get_diff() -> str:
        ok, output = GitIntegration._run_git_command(["diff"])
        return output if ok else ""
        
    @staticmethod
    def add_files(files: list = None) -> Tuple[bool, str]:
        return GitIntegration._run_git_command(["add"] + (files or ["."]))

    @staticmethod
    def commit_changes(message: str, auto_add: bool = False) -> Tuple[bool, str]:
        if auto_add:
            add_ok, add_output = GitIntegration.add_files()
            if not add_ok: return False, f"Failed to add files: {add_output}"

        has_staged_changes = GitIntegration.get_staged_diff()
        if not has_staged_changes:
            return False, "No changes staged for commit. Use `git add` to stage files."
            
        return GitIntegration._run_git_command(["commit", "-m", message])
    
    @staticmethod
    def auto_commit(message: str = None) -> Tuple[bool, str]:
        if not message: message = f"Auto-commit by Agent at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return GitIntegration.commit_changes(message, auto_add=True)

    @staticmethod
    def create_backup_branch() -> Optional[str]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"g4f_backup_{timestamp}"
        ok, _ = GitIntegration._run_git_command(["branch", branch_name])
        return branch_name if ok else None
