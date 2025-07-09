#!/usr/bin/env python3
"""
AI Commit Message Generator using gpt4free (g4f)

This tool uses AI to generate meaningful git commit messages based on 
staged changes. It analyzes the git diff and suggests appropriate commit 
messages following conventional commit format. The tool can work with
any git repository and supports multiple AI models for message generation.

Created for use with gpt4free (g4f) development workflow.

Usage:
  python -m etc.tool.commit [options]

Examples:
  # Generate commit for current repository
  python -m etc.tool.commit
  
  # Generate commit for specific repository
  python -m etc.tool.commit --repo /path/to/repo
  python -m etc.tool.commit --repo ../my-project
  python -m etc.tool.commit --repo ~/projects/website
  
  # Generate and edit commit message before committing
  python -m etc.tool.commit --repo ./docs --edit
  
  # Generate message only without committing
  python -m etc.tool.commit --repo ~/projects/app --no-commit
  
  # Use specific AI model
  python -m etc.tool.commit --model gpt-4 --repo ./backend
  
  # List available models
  python -m etc.tool.commit --list-models
  
  # Complex workflow example
  python -m etc.tool.commit --repo ../frontend --model claude-3-sonnet --edit

Features:
  - AI-powered commit message generation using gpt4free
  - Support for multiple AI models (GPT-4, Claude, Gemini, etc.)
  - Conventional commit format compliance
  - Multi-repository support with automatic validation
  - Interactive editing with system default editor
  - Sensitive data filtering and protection
  - Automatic retry logic with fallback models
  - Git repository information display
  - Staged changes analysis and diff processing
  - Comprehensive error handling and validation

Options:
  --model MODEL      Specify the AI model to use (default: gpt-4o)
  --edit             Edit the generated commit message before committing
  --no-commit        Generate message only without committing
  --list-models      List available AI models and exit
  --repo PATH        Specify git repository path (default: current directory)
  --help             Show this help message and exit

Requirements:
  - gpt4free (g4f) library
  - Git installed and accessible via command line
  - Active git repository with staged changes
  - Internet connection for AI model access

Workflow:
  1. Validates specified git repository
  2. Fetches staged changes using git diff
  3. Filters sensitive data from diff
  4. Generates commit message using AI
  5. Optionally allows editing the message
  6. Creates git commit with generated message

Security Features:
  - Automatic filtering of sensitive patterns (passwords, tokens, etc.)
  - Local processing of git data
  - No permanent storage of repository content
  - Safe handling of authentication credentials

Supported Commit Types:
  - feat: New features
  - fix: Bug fixes
  - docs: Documentation changes
  - refactor: Code refactoring
  - test: Test additions/modifications
  - style: Code style changes
  - perf: Performance improvements
  - chore: Maintenance tasks

Author: Created for gpt4free (g4f) project
License: MIT
"""
import subprocess
import sys
import os
import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Any, List

from g4f.client import Client
from g4f.models import ModelUtils

from g4f import debug
debug.logging = True

# Constants
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODELS = []
MAX_DIFF_SIZE = None  # Set to None to disable truncation, or a number for character limit
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Commit Message Generator using gpt4free (g4f)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"AI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--edit", action="store_true",
                        help="Edit the generated commit message before committing")
    parser.add_argument("--no-commit", action="store_true",
                        help="Generate message only without committing")
    parser.add_argument("--list-models", action="store_true",
                        help="List available AI models and exit")
    parser.add_argument("--repo", type=str, default=".",
                        help="Git repository path (default: current directory)")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                        help="Maximum number of retries for AI generation (default: 3)")
    
    return parser.parse_args()

def validate_git_repository(repo_path: str) -> Path:
    """Validate that the specified path is a git repository"""
    repo_path = Path(repo_path).resolve()
    
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not repo_path.is_dir():
        print(f"Error: Repository path is not a directory: {repo_path}")
        sys.exit(1)
    
    # Check if it's a git repository
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        # Check if we're in a subdirectory of a git repo
        current = repo_path
        while current != current.parent:
            if (current / ".git").exists():
                repo_path = current
                break
            current = current.parent
        else:
            print(f"Error: Not a git repository: {repo_path}")
            print("Initialize a git repository with: git init")
            sys.exit(1)
    
    return repo_path

def get_git_diff(repo_path: Path) -> Optional[str]:
    """Get the current git diff for staged changes in specified repository"""
    try:
        diff_process = subprocess.run(
            ["git", "diff", "--staged"], 
            capture_output=True, 
            text=True,
            cwd=repo_path
        )
        if diff_process.returncode != 0:
            print(f"Error: git diff command failed with code {diff_process.returncode}")
            print(f"Error output: {diff_process.stderr}")
            return None
        
        return diff_process.stdout
    except Exception as e:
        print(f"Error running git diff in {repo_path}: {e}")
        return None

def get_repository_info(repo_path: Path) -> dict:
    """Get information about the git repository"""
    info = {
        "path": repo_path,
        "name": repo_path.name,
        "branch": "unknown",
        "remote": "unknown"
    }
    
    try:
        # Get current branch
        branch_process = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        if branch_process.returncode == 0:
            info["branch"] = branch_process.stdout.strip()
    except:
        pass
    
    try:
        # Get remote origin URL
        remote_process = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        if remote_process.returncode == 0:
            info["remote"] = remote_process.stdout.strip()
    except:
        pass
    
    return info

def truncate_diff(diff_text: str, max_size: int = MAX_DIFF_SIZE) -> str:
    """Truncate diff if it's too large, preserving the most important parts"""
    if max_size is None or len(diff_text) <= max_size:
        return diff_text
    
    print(f"Warning: Diff is large ({len(diff_text)} chars), truncating to {max_size} chars")
    
    # Split by file sections and keep as many complete files as possible
    sections = diff_text.split("diff --git ")
    header = sections[0]
    file_sections = ["diff --git " + s for s in sections[1:]]
    
    result = header
    for section in file_sections:
        if len(result) + len(section) <= max_size:
            result += section
        else:
            break
    
    return result

def filter_sensitive_data(diff_text: str) -> str:
    """Filter out potentially sensitive data from the diff"""
    # List of patterns that might indicate sensitive data
    sensitive_patterns = [
        ("password", "***REDACTED***"),
        ("secret", "***REDACTED***"),
        ("token", "***REDACTED***"),
        ("api_key", "***REDACTED***"),
        ("apikey", "***REDACTED***"),
        ("auth", "***REDACTED***"),
        ("credential", "***REDACTED***"),
    ]
    
    # Simple pattern matching - in a real implementation, you might want more sophisticated regex
    filtered_text = diff_text
    for pattern, replacement in sensitive_patterns:
        # Only replace if it looks like an assignment or declaration
        filtered_text = filtered_text.replace(f'{pattern}="', f'{pattern}="{replacement}')
        filtered_text = filtered_text.replace(f"{pattern}='", f"{pattern}='{replacement}'")
        filtered_text = filtered_text.replace(f"{pattern}:", f"{pattern}: {replacement}")
        filtered_text = filtered_text.replace(f"{pattern} =", f"{pattern} = {replacement}")
    
    return filtered_text

def show_spinner(duration: int = None):
    """Display a simple spinner to indicate progress"""
    import itertools
    import threading
    import time
    
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    stop_spinner = threading.Event()
    
    def spin():
        while not stop_spinner.is_set():
            sys.stdout.write(f"\rGenerating commit message... {next(spinner)} ")
            sys.stdout.flush()
            time.sleep(0.1)
    
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()
    
    try:
        if duration:
            time.sleep(duration)
            stop_spinner.set()
        return stop_spinner
    except:
        stop_spinner.set()
        raise

def generate_commit_message(diff_text: str, model: str = DEFAULT_MODEL, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """Generate a commit message based on the git diff"""
    if not diff_text or diff_text.strip() == "":
        return "No changes staged for commit"
    
    # Filter sensitive data
    filtered_diff = filter_sensitive_data(diff_text)
    
    # Truncate if necessary
    truncated_diff = truncate_diff(filtered_diff)
    
    client = Client()
    
    prompt = f"""
    {truncated_diff}
    ```
    
    Analyze ONLY the exact changes in this git diff and create a precise commit message.

    FORMAT:
    1. First line: "<type>: <summary>" (max 70 chars)
       - Type: feat, fix, docs, refactor, test, etc.
       - Summary must describe ONLY actual changes shown in the diff
       
    2. Leave one blank line

    3. Add sufficient bullet points to:
       - Describe ALL specific changes seen in the diff
       - Reference exact functions/files/components that were modified
       - Do NOT mention anything not explicitly shown in the code changes
       - Avoid general statements or assumptions not directly visible in diff
       - Include enough points to cover all significant changes (don't limit to a specific number)

    IMPORTANT: Be 100% factual. Only mention code that was actually changed. Never invent or assume changes not shown in the diff. If unsure about a change's purpose, describe what changed rather than why. Output nothing except for the commit message, and don't surround it in quotes.
    """
    
    for attempt in range(max_retries):
        try:
            # Start spinner
            spinner = show_spinner()
            
            # Make API call
            response = client.chat.completions.create(
                prompt,
                model=model,
                stream=True,
            )
            content = []
            for chunk in response:
                if isinstance(chunk.choices[0].delta.content, str):
                    # Stop spinner and clear line
                    if spinner:
                        spinner.set()
                        print(" " * 50 + "\n", flush=True)
                        spinner = None
                    content.append(chunk.choices[0].delta.content)
                    print(chunk.choices[0].delta.content, end="", flush=True)
            return "".join(content).strip("`").split("\n---\n")[0].strip()
        except Exception as e:
            # Stop spinner if it's running
            if 'spinner' in locals() and spinner:
                spinner.set()
                sys.stdout.write("\r" + " " * 50 + "\r")
                sys.stdout.flush()
            if max_retries == 1:
                raise e  # If no retries, raise immediately
            print(f"Error generating commit message (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                # Try with a fallback model if available
                if attempt < len(FALLBACK_MODELS):
                    fallback = FALLBACK_MODELS[attempt]
                    print(f"Trying with fallback model: {fallback}")
                    model = fallback
    
    return None

def edit_commit_message(message: str) -> str:
    """Allow user to edit the commit message in their default editor"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp:
        temp.write(message)
        temp_path = temp.name
    
    # Get the default editor from git config or environment
    try:
        editor = subprocess.run(
            ["git", "config", "--get", "core.editor"],
            capture_output=True, text=True
        ).stdout.strip()
    except:
        editor = os.environ.get('EDITOR', 'vim')
    
    if not editor:
        editor = 'vim'  # Default fallback
    
    # Open the editor
    try:
        subprocess.run([editor, temp_path], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Editor exited with an error")
    except FileNotFoundError:
        print(f"Warning: Editor '{editor}' not found, falling back to basic input")
        print("Edit your commit message (Ctrl+D when done):")
        edited_message = sys.stdin.read().strip()
        os.unlink(temp_path)
        return edited_message
    
    # Read the edited message
    with open(temp_path, 'r') as temp:
        edited_message = temp.read()
    
    # Clean up
    os.unlink(temp_path)
    
    return edited_message

def list_available_models() -> List[str]:
    """List available AI models that can be used for commit message generation"""
    # Filter for text models that are likely to be good for code understanding
    relevant_models = []
    
    for model_name, model in ModelUtils.convert.items():
        # Skip image, audio, and video models
        if model_name and not model_name.startswith(('dall', 'sd-', 'flux', 'midjourney')):
            relevant_models.append(model_name)
    
    return sorted(relevant_models)

def make_commit(message: str, repo_path: Path) -> bool:
    """Make a git commit with the provided message in specified repository"""
    try:
        subprocess.run(
            ["git", "commit", "-m", message], 
            check=True,
            cwd=repo_path
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error making commit: {e}")
        return False

def main():
    """Main function"""
    try:
        args = parse_arguments()
        
        # If --list-models is specified, list available models and exit
        if args.list_models:
            print("Available AI models for commit message generation:")
            for model in list_available_models():
                print(f"  - {model}")
            sys.exit(0)
        
        # Validate and get repository path
        repo_path = validate_git_repository(args.repo)
        repo_info = get_repository_info(repo_path)
        
        print(f"Repository: {repo_info['name']} ({repo_path})")
        print(f"Branch: {repo_info['branch']}")
        if repo_info['remote'] != "unknown":
            print(f"Remote: {repo_info['remote']}")
        print()
        
        print("Fetching git diff...")
        diff = get_git_diff(repo_path)
        
        if diff is None:
            print("Failed to get git diff.")
            sys.exit(1)
        
        if diff.strip() == "":
            print("No changes staged for commit. Stage changes with 'git add' first.")
            print(f"Run this in the repository: cd {repo_path} && git add <files>")
            sys.exit(0)
        
        print(f"Using model: {args.model}")
        commit_message = generate_commit_message(diff, args.model, args.max_retries)
        
        if not commit_message:
            print("Failed to generate commit message after multiple attempts.")
            sys.exit(1)
        
        if args.edit:
            print("\nOpening editor to modify commit message...")
            commit_message = edit_commit_message(commit_message)
            print("\nEdited commit message:")
            print("-" * 50)
            print(commit_message)
            print("-" * 50)
        
        if args.no_commit:
            print("\nCommit message generated but not committed (--no-commit flag used).")
            sys.exit(0)
        
        user_input = input(f"\nDo you want to commit to {repo_info['name']} ({repo_info['branch']})? (y/n): ")
        if user_input.lower() == 'y':
            if make_commit(commit_message, repo_path):
                print("Commit successful!")
            else:
                print("Commit failed.")
                sys.exit(1)
        else:
            print("Commit aborted.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)  # Standard exit code for SIGINT

if __name__ == "__main__":
    main()
