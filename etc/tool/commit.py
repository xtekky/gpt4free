#!/usr/bin/env python3
"""
AI Commit Message Generator using gpt4free (g4f)

This tool uses AI to generate meaningful git commit messages based on 
staged changes. It analyzes the git diff and suggests appropriate commit 
messages following conventional commit format.

Usage:
  python -m etc.tool.commit
"""
import subprocess
import sys
from g4f.client import Client

def get_git_diff():
    """Get the current git diff for staged changes"""
    try:
        diff_process = subprocess.run(
            ["git", "diff", "--staged"], 
            capture_output=True, 
            text=True
        )
        return diff_process.stdout
    except Exception as e:
        print(f"Error running git diff: {e}")
        return None

def generate_commit_message(diff_text):
    """Generate a commit message based on the git diff"""
    if not diff_text or diff_text.strip() == "":
        return "No changes staged for commit"
    
    client = Client()
    
    prompt = f"""
    {diff_text}
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
    
    try:
        response = client.chat.completions.create(
            model="claude-3.7-sonnet",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating commit message: {e}")
        return None

def main():
    print("Fetching git diff...")
    diff = get_git_diff()
    
    if diff is None:
        print("Failed to get git diff. Are you in a git repository?")
        sys.exit(1)
    
    if diff.strip() == "":
        print("No changes staged for commit. Stage changes with 'git add' first.")
        sys.exit(0)
    
    print("Generating commit message...")
    commit_message = generate_commit_message(diff)
    
    if commit_message:
        print("\nGenerated commit message:")
        print("-" * 50)
        print(commit_message)
        print("-" * 50)
        
        user_input = input("\nDo you want to use this commit message? (y/n): ")
        if user_input.lower() == 'y':
            try:
                subprocess.run(
                    ["git", "commit", "-m", commit_message], 
                    check=True
                )
                print("Commit successful!")
            except subprocess.CalledProcessError as e:
                print(f"Error making commit: {e}")
    else:
        print("Failed to generate commit message.")

if __name__ == "__main__":
    main()
