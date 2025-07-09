"""Security utilities."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List


class SecuritySandbox:
    """Security sandbox for executing commands."""
    
    # Dangerous command patterns
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/',              # Recursive force remove from root
        r'rm\s+-rf\s+~',              # Recursive force remove home
        r':\(\)\s*\{\s*:\|:&\s*\};:', # Fork bomb
        r'>\s*/dev/sda',              # Write to disk device
        r'dd\s+if=.*of=/dev/',        # dd to device
        r'mkfs\.',                    # Format filesystem
        r'fdisk\s+',                  # Disk partitioning
        r'sudo\s+',                   # Sudo commands
        r'su\s+-',                    # Switch user
        r'chmod\s+777',               # Dangerous permissions
        r'chmod\s+\+s',               # Setuid
        r'curl.*\|.*sh',              # Curl pipe to shell
        r'wget.*\|.*sh',              # Wget pipe to shell
        r'eval\s*\(',                 # Eval in various languages
        r'exec\s*\(',                 # Exec in various languages
        r'system\s*\(',               # System calls
        r'`.*`',                      # Backticks command execution
        r'\$\(.*\)',                  # Command substitution
        r'nc\s+-l',                   # Netcat listener
        r'/etc/passwd',               # Password file access
        r'/etc/shadow',               # Shadow file access
        r'iptables\s+-F',             # Flush firewall rules
    ]
    
    # File patterns that should not be executed
    DANGEROUS_FILES = [
        r'.*\.sh$',     # Shell scripts (need review)
        r'.*\.bat$',    # Batch files
        r'.*\.exe$',    # Executables
        r'.*\.dll$',    # Libraries
        r'.*\.so$',     # Shared objects
        r'.*\.dylib$',  # macOS libraries
    ]
    
    # Safe commands whitelist
    SAFE_COMMANDS = [
        'echo', 'printf', 'cat', 'ls', 'pwd', 'cd', 'mkdir', 'touch',
        'cp', 'mv', 'grep', 'sed', 'awk', 'sort', 'uniq', 'wc',
        'head', 'tail', 'less', 'more', 'find', 'which', 'whereis',
        'date', 'cal', 'df', 'du', 'free', 'ps', 'top', 'htop',
        'git', 'npm', 'yarn', 'pip', 'python', 'python3', 'node',
        'go', 'cargo', 'rustc', 'gcc', 'g++', 'make', 'cmake',
        'pytest', 'jest', 'mocha', 'rspec', 'phpunit',
    ]
    
    @staticmethod
    def is_safe_command(cmd: str) -> Tuple[bool, str]:
        """Check if command is safe to execute."""
        # Check against dangerous patterns
        for pattern in SecuritySandbox.DANGEROUS_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return False, f"Dangerous pattern detected: {pattern}"
        
        # Check if command starts with a safe command
        cmd_parts = cmd.strip().split()
        if cmd_parts:
            base_cmd = cmd_parts[0]
            if base_cmd not in SecuritySandbox.SAFE_COMMANDS:
                # Check if it's a path to a safe command
                if '/' in base_cmd:
                    base_cmd = Path(base_cmd).name
                    if base_cmd not in SecuritySandbox.SAFE_COMMANDS:
                        return False, f"Command '{base_cmd}' not in safe commands list"
        
        # Additional checks for specific commands
        if 'git' in cmd:
            # Allow most git commands except dangerous ones
            if any(dangerous in cmd for dangerous in ['push --force', 'reset --hard', 'clean -fdx']):
                return False, "Potentially dangerous git command"
        
        if 'rm' in cmd:
            # Only allow safe rm usage
            if not all(safe in cmd for safe in ['-i', '--interactive']):
                return False, "rm command must use interactive mode (-i)"
        
        return True, "Command appears safe"
    
    @staticmethod
    def is_safe_file(file_path: str) -> Tuple[bool, str]:
        """Check if file is safe to execute."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False, "File does not exist"
        
        # Check file size (limit to 10MB)
        if path.stat().st_size > 10 * 1024 * 1024:
            return False, "File too large (>10MB)"
        
        # Check against dangerous file patterns
        for pattern in SecuritySandbox.DANGEROUS_FILES:
            if re.match(pattern, str(path)):
                # Shell scripts might be okay if they're user-created
                if pattern == r'.*\.sh$' and path.stat().st_size < 1024 * 100:  # <100KB
                    try:
                        content = path.read_text()
                        # Check content for dangerous patterns
                        for dangerous in SecuritySandbox.DANGEROUS_PATTERNS:
                            if re.search(dangerous, content):
                                return False, f"Script contains dangerous pattern: {dangerous}"
                        return True, "Shell script appears safe"
                    except:
                        return False, "Cannot read file content"
                return False, f"File type not allowed: {pattern}"
        
        # Check file permissions
        mode = path.stat().st_mode
        if mode & 0o4000:  # Setuid
            return False, "File has setuid bit set"
        if mode & 0o2000:  # Setgid
            return False, "File has setgid bit set"
        
        return True, "File appears safe"
    
    @staticmethod
    def sandbox_exec(cmd: str, cwd: str = None, env: dict = None) -> Tuple[int, str, str]:
        """Execute command in sandboxed environment."""
        # First check if command is safe
        is_safe, reason = SecuritySandbox.is_safe_command(cmd)
        if not is_safe:
            return -1, "", f"Command blocked: {reason}"
        
        # Prepare safe environment
        safe_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': str(Path.home()),
            'USER': 'sandbox',
            'LANG': 'en_US.UTF-8',
        }
        
        if env:
            # Only allow specific environment variables
            allowed_env_vars = ['PYTHONPATH', 'NODE_PATH', 'GOPATH', 'CARGO_HOME']
            for key, value in env.items():
                if key in allowed_env_vars:
                    safe_env[key] = value
        
        try:
            # Use timeout and memory limits
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                # Note: Memory limits would require additional OS-specific code
            )
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out after 30 seconds"
        except Exception as e:
            return -1, "", f"Execution error: {e}"
    
    @staticmethod
    def validate_file_operation(operation: str, file_path: str) -> Tuple[bool, str]:
        """Validate file operations."""
        path = Path(file_path).resolve()
        
        # Prevent operations outside current directory tree
        try:
            path.relative_to(Path.cwd())
        except ValueError:
            return False, "File operation outside current directory not allowed"
        
        # Prevent operations on system files
        system_dirs = ['/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin', '/boot', '/sys', '/proc']
        for sys_dir in system_dirs:
            if str(path).startswith(sys_dir):
                return False, f"Operations on system directories not allowed: {sys_dir}"
        
        # Prevent operations on hidden files in home directory
        if str(path).startswith(str(Path.home())) and '/.' in str(path):
            sensitive_files = ['.ssh', '.gnupg', '.aws', '.config', '.bashrc', '.zshrc']
            for sensitive in sensitive_files:
                if sensitive in str(path):
                    return False, f"Operations on sensitive files not allowed: {sensitive}"
        
        # Check specific operations
        if operation == "delete":
            # Don't allow deleting git directory
            if '.git' in path.parts:
                return False, "Cannot delete .git directory"
            # Don't allow deleting current directory
            if path == Path.cwd():
                return False, "Cannot delete current directory"
        
        return True, "Operation appears safe"
    
    @staticmethod
    def scan_code_for_vulnerabilities(code: str, language: str) -> List[str]:
        """Basic vulnerability scanning for code."""
        vulnerabilities = []
        
        if language == "python":
            # Check for dangerous imports
            dangerous_imports = ['os', 'subprocess', 'eval', 'exec', '__import__']
            for imp in dangerous_imports:
                if re.search(rf'import\s+{imp}|from\s+{imp}\s+import', code):
                    vulnerabilities.append(f"Potentially dangerous import: {imp}")
            
            # Check for SQL injection patterns
            if re.search(r'SELECT.*\+.*%|INSERT.*\+.*%|UPDATE.*\+.*%|DELETE.*\+.*%', code):
                vulnerabilities.append("Potential SQL injection vulnerability")
            
            # Check for command injection
            if re.search(r'os\.system\(|subprocess\..*\(|eval\(|exec\(', code):
                vulnerabilities.append("Potential command injection vulnerability")
        
        elif language == "javascript":
            # Check for dangerous functions
            if re.search(r'eval\(|Function\(|setTimeout\([^,]+,', code):
                vulnerabilities.append("Potentially dangerous JavaScript evaluation")
            
            # Check for XSS patterns
            if re.search(r'innerHTML\s*=|document\.write\(', code):
                vulnerabilities.append("Potential XSS vulnerability")
        
        return vulnerabilities
    
    @staticmethod
    def create_sandbox_directory() -> Path:
        """Create a temporary sandbox directory."""
        sandbox_dir = Path(tempfile.mkdtemp(prefix="g4f_sandbox_"))
        
        # Set restrictive permissions
        sandbox_dir.chmod(0o700)
        
        return sandbox_dir
    
    @staticmethod
    def cleanup_sandbox_directory(sandbox_dir: Path) -> None:
        """Clean up sandbox directory."""
        if sandbox_dir.exists() and str(sandbox_dir).startswith("/tmp/g4f_sandbox_"):
            import shutil
            shutil.rmtree(sandbox_dir)
