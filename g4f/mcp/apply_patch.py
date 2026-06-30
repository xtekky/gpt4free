import re
import sys
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PatchResult:
    """Result of a patch application operation."""
    success: bool
    output: str = ""
    error: Optional[str] = None
    files_changed: list[str] = field(default_factory=list)


def _find_patch_command() -> Optional[str]:
    """
    Locate the patch command on the system.
    Returns the command path or None if not found.
    """
    # Check if patch is available in PATH
    if shutil.which('patch'):
        return 'patch'
    
    # Windows-specific locations
    if sys.platform == 'win32':
        # Common Git for Windows locations
        git_locations = [
            r'C:\Program Files\Git\usr\bin\patch.exe',
            r'C:\Program Files (x86)\Git\usr\bin\patch.exe',
            r'C:\msys64\usr\bin\patch.exe',
        ]
        for loc in git_locations:
            if Path(loc).exists():
                return loc
    
    return None


def patch_is_available() -> bool:
    """Check if the patch command is available on the system"""
    return _find_patch_command() is not None


def apply_patch(
    patch_content: str,
    target_file: str,
    backup: bool = True,
    dry_run: bool = False,
    strip: int = 0
) -> PatchResult:
    """
    Apply a unified diff patch. Falls back to Python if system patch unavailable.
    """
    # Check patch availability first
    patch_cmd = _find_patch_command()
    
    if patch_cmd is None:
        # Fallback to pure Python implementation
        return _apply_patch_python(patch_content, target_file, backup, dry_run)
    
    # Use system patch command
    return _apply_patch_system(
        patch_cmd, patch_content, target_file, backup, dry_run, strip
    )


def _apply_patch_system(
    patch_cmd: str,
    patch_content: str,
    target_file: str,
    backup: bool,
    dry_run: bool,
    strip: int
) -> PatchResult:
    """Apply patch using system patch command"""
    target_path = Path(target_file)
    
    if not target_path.exists() and not dry_run:
        return PatchResult(
            success=False,
            output="",
            error=f"Target file not found: {target_file}"
        )
    
    patch_file = None
    try:
        # Normalize line endings to Unix-style (LF) for compatibility
        normalized = patch_content.replace('\r\n', '\n').replace('\r', '\n')
        # Ensure patch ends with a newline to avoid "unexpectedly ends in middle of line"
        if normalized and not normalized.endswith('\n'):
            normalized += '\n'

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.patch', delete=False, encoding='utf-8', newline='\n'
        ) as f:
            f.write(normalized)
            patch_file = f.name

        cmd = [patch_cmd, '--binary']
        if dry_run:
            cmd.append('--dry-run')
        if backup:
            cmd.append('--backup')
        
        cmd.extend(['-p', str(strip)])
        cmd.extend(['-i', patch_file])
        
        if target_path.is_dir():
            cmd.extend(['-d', str(target_path)])
        else:
            cmd.append(str(target_path))
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(target_path.parent) if target_path.is_file() else None
        )
        
        output = process.stdout + process.stderr
        
        if process.returncode == 0:
            return PatchResult(
                success=True,
                output=output,
                files_changed=_parse_patched_files(output)
            )
        else:
            return PatchResult(
                success=False,
                output=output,
                error=_extract_error(output)
            )
            
    except subprocess.TimeoutExpired:
        return PatchResult(
            success=False, output="",
            error="Patch application timed out"
        )
    except Exception as e:
        return PatchResult(
            success=False, output="",
            error=f"System patching error: {str(e)}"
        )
    finally:
        if patch_file and os.path.exists(patch_file):
            try:
                os.unlink(patch_file)
            except OSError:
                pass


def _apply_patch_python(
    patch_content: str,
    target_file: str,
    backup: bool = False,
    dry_run: bool = False
) -> PatchResult:
    """
    Pure Python patch implementation as fallback.
    Handles basic unified diffs without external dependencies.
    """
    import difflib
    import re
    
    target_path = Path(target_file)
    
    if not target_path.exists():
        return PatchResult(
            success=False, output="",
            error=f"Target file not found: {target_file}"
        )
    
    if not target_path.is_file():
        return PatchResult(
            success=False, output="",
            error=f"Python fallback only supports single files, got: {target_file}"
        )
    
    try:
        # Read original content
        original = target_path.read_text(encoding='utf-8')
        original_lines = original.splitlines(True)
        
        # Parse the unified diff
        patched_lines = _parse_and_apply_unified_diff(original_lines, patch_content)
        
        if patched_lines is None:
            return PatchResult(
                success=False, output="",
                error="Failed to parse or apply patch"
            )
        
        # Check if any changes
        changed = original_lines != patched_lines
        
        if not changed:
            return PatchResult(
                success=True,
                output="Patch already applied",
                files_changed=[]
            )
        
        if dry_run:
            return PatchResult(
                success=True,
                output=f"Dry run successful: {len(patched_lines)} lines",
                files_changed=[str(target_path)]
            )
        
        # Apply changes
        if backup:
            backup_path = target_path.with_suffix(target_path.suffix + '.orig')
            backup_path.write_text(original, encoding='utf-8')
        
        target_path.write_text(''.join(patched_lines), encoding='utf-8')
        
        return PatchResult(
            success=True,
            output=f"Successfully patched {target_file}",
            files_changed=[str(target_path)]
        )
        
    except Exception as e:
        return PatchResult(
            success=False, output="",
            error=f"Python patching error: {str(e)}"
        )


def _parse_and_apply_unified_diff(
    original_lines: list[str],
    diff_text: str
) -> Optional[list[str]]:
    """
    Minimal unified diff parser and applier.
    Returns patched lines or None on failure.
    """
    # Parse hunks from unified diff
    hunks = re.findall(
        r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*?)(?=@@|\Z)',
        diff_text,
        re.DOTALL
    )
    
    if not hunks:
        return None
    
    result_lines = original_lines.copy()
    
    for hunk in reversed(hunks):  # Apply in reverse to maintain positions
        old_start, old_count, new_start, new_count, body = hunk
        old_start = int(old_start) - 1  # Convert to 0-based
        old_count = int(old_count) if old_count else 1
        
        # Parse hunk body
        body_lines = body.split('\n')[1:]  # Skip first empty line
        hunk_result = []
        
        old_pos = old_start
        for line in body_lines:
            if not line:
                continue
            
            if line.startswith('+'):
                hunk_result.append(line[1:] + '\n')
            elif line.startswith('-'):
                old_pos += 1
            elif line.startswith(' '):
                if old_pos < len(result_lines):
                    hunk_result.append(result_lines[old_pos])
                old_pos += 1
            else:
                # Context line
                pass
        
        # Replace the hunk region
        end_index = min(old_start + old_count, len(result_lines))
        result_lines[old_start:end_index] = hunk_result
    
    return result_lines


def _parse_patched_files(output: str) -> list[str]:
    """Extract patched file names from patch command output."""
    files = []
    for line in output.splitlines():
        if line.startswith("patching file "):
            files.append(line[len("patching file "):].strip())
    return files


def _extract_error(output: str) -> str:
    """Extract error message from patch command output."""
    for line in output.splitlines():
        if "FAILED" in line or "failed" in line or "error" in line.lower():
            return line.strip()
    return output.strip() or "Unknown patch error"


def apply_patch_with_fallback(
    patch_content: str,
    target_file: str,
    backup: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Apply patch with automatic fallback detection.
    Tries system patch first, falls back to Python implementation on failure.
    Returns structured result.
    """
    # Check system capabilities
    using_python = not patch_is_available()

    if using_python:
        # System patch not available, use Python directly
        result = apply_patch(patch_content, target_file, backup, dry_run)
    else:
        # Try system patch first
        result = apply_patch(patch_content, target_file, backup, dry_run)
        # If system patch failed, try Python fallback
        if not result.success:
            result = _apply_patch_python(patch_content, target_file, backup, dry_run)
            using_python = True

    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "files_changed": result.files_changed,
        "implementation": "python_fallback" if using_python else "system_patch"
    }