#!/usr/bin/env python3
"""
AI Commit Message Generator using gpt4free (g4f)

This tool uses AI to generate meaningful git commit messages based on 
staged changes. It analyzes the git diff and suggests appropriate commit 
messages following conventional commit format.

Usage:
  python -m etc.tool.commit [options]

Options:
  --model MODEL    Specify the AI model to use (default: claude-3.7-sonnet)
  --edit           Edit the generated commit message before committing
  --no-commit      Generate message only without committing
  --list-models    List available AI models and exit
  --help           Show this help message
"""
import subprocess
import sys
import os
import argparse
import tempfile
import time
from typing import Optional, Any, List

from g4f.client import Client
from g4f.models import ModelUtils

from g4f import debug
debug.logging = True

# Constants
DEFAULT_MODEL = "o1"
FALLBACK_MODELS = ["o1", "o3-mini", "gpt-4o"]
MAX_DIFF_SIZE = None  # Set to None to disable truncation, or a number for character limit
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Commit Message Generator",
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
    
    return parser.parse_args()

def get_git_diff() -> Optional[str]:
    """Get the current git diff for staged changes"""
    try:
        diff_process = subprocess.run(
            ["git", "diff", "--staged"], 
            capture_output=True, 
            text=True
        )
        if diff_process.returncode != 0:
            print(f"Error: git diff command failed with code {diff_process.returncode}")
            return None
        
        return diff_process.stdout
    except Exception as e:
        print(f"Error running git diff: {e}")
        return None

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

def generate_commit_message(diff_text: str, model: str = DEFAULT_MODEL) -> Optional[str]:
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
    
    for attempt in range(MAX_RETRIES):
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
                # Stop spinner and clear line
                if spinner:
                    spinner.set()
                    print(" " * 50 + "\n", flush=True)
                    spinner = None
                if isinstance(chunk.choices[0].delta.content, str):
                    content.append(chunk.choices[0].delta.content)
                    print(chunk.choices[0].delta.content, end="", flush=True)
            return "".join(content).strip()
        except Exception as e:
            # Stop spinner if it's running
            if 'spinner' in locals() and spinner:
                spinner.set()
                sys.stdout.write("\r" + " " * 50 + "\r")
                sys.stdout.flush()
                
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

def make_commit(message: str) -> bool:
    """Make a git commit with the provided message"""
    try:
        subprocess.run(
            ["git", "commit", "-m", message], 
            check=True
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
        
        print("Fetching git diff...")
        diff = get_git_diff()
        
        if diff is None:
            print("Failed to get git diff. Are you in a git repository?")
            sys.exit(1)
        
        if diff.strip() == "":
            print("No changes staged for commit. Stage changes with 'git add' first.")
            sys.exit(0)
        
        print(f"Using model: {args.model}")
        commit_message = generate_commit_message(diff, args.model)
        
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
        
        user_input = input("\nDo you want to use this commit message? (y/n): ")
        if user_input.lower() == 'y':
            if make_commit(commit_message):
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
