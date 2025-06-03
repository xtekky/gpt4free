#!/usr/bin/env python3
"""
Markdown to HTML Converter using GitHub API

This tool converts Markdown files to HTML using GitHub's Markdown API,
producing high-quality HTML with GitHub Flavored Markdown support.
It supports single files, directories, and batch processing with
comprehensive error handling and retry logic.

Created for use with gpt4free (g4f) documentation system.

Usage:
  python -m etc.tool.і [file.md] [options]

Examples:
  # Convert single file
  python -m etc.tool.md2html README.md
  
  # Convert all files in directory
  python -m etc.tool.md2html -d docs/
  
  # Convert with custom output
  python -m etc.tool.md2html file.md -o output.html
  
  # Use custom template
  python -m etc.tool.md2html -t custom.html file.md

Features:
  - GitHub Flavored Markdown conversion
  - Automatic retry logic for API failures
  - Rate limit handling
  - Template-based HTML generation
  - Recursive directory processing
  - Custom output paths
  - Comprehensive error handling

Requirements:
  - requests library
  - GITHUB_TOKEN environment variable (optional, but recommended)
  - template.html file in script directory

Author: Created for gpt4free (g4f) project
License: MIT
"""

import os
import sys
import requests
import time
import argparse
from pathlib import Path
from typing import Optional, List

def get_github_token() -> Optional[str]:
    """Get GitHub token with validation."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Warning: GITHUB_TOKEN not found. API requests may be rate-limited.")
        return None
    return token

def extract_title(content: str) -> str:
    """Extract title from markdown content with fallback."""
    if not content.strip():
        return "Untitled"
    
    lines = content.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            return line.lstrip('#').strip()
    
    return "Untitled"

def process_markdown_links(content: str) -> str:
    """Process markdown links for proper HTML conversion."""
    content = content.replace("(README.md)", "(/docs/)")
    content = content.replace("(../README.md)", "(/docs/)")
    content = content.replace(".md)", ".html)")
    return content

def convert_markdown_to_html(content: str, token: Optional[str] = None) -> str:
    """Convert markdown to HTML using GitHub API with retry logic."""
    processed_content = process_markdown_links(content)
    
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Markdown-Converter/1.0"
    }
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    payload = {
        "text": processed_content,
        "mode": "gfm",
        "context": "gpt4free/gpt4free"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.github.com/markdown",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403:
                print(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(60)
                continue
            elif response.status_code == 401:
                print("Authentication failed. Check your GITHUB_TOKEN.")
                sys.exit(1)
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"Network error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
    
    print("Failed to convert markdown after all retries")
    sys.exit(1)

def load_template(template_path: Path) -> str:
    """Load HTML template with error handling."""
    if not template_path.exists():
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading template file: {e}")
        sys.exit(1)

def process_file(file_path: Path, template: str, output_dir: Optional[Path] = None, token: Optional[str] = None) -> bool:
    """Process a single markdown file."""
    try:
        # Read markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"Warning: Empty file {file_path}")
            return False
        
        # Extract title
        title = extract_title(content)
        print(f"Processing: {file_path.name} -> Title: {title}")
        
        # Convert to HTML
        html = convert_markdown_to_html(content, token)
        
        # Generate output filename
        if file_path.name == "README.md":
            output_filename = "index.html"
        else:
            output_filename = file_path.stem + ".html"
        
        # Determine output path
        if output_dir:
            output_path = output_dir / output_filename
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = file_path.parent / output_filename
        
        # Generate final HTML
        final_html = template.replace("{{ article }}", html).replace("{{ title }}", title)
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"✓ Created: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def find_markdown_files(path: Path, recursive: bool = True) -> List[Path]:
    """Find markdown files in given path."""
    markdown_files = []
    
    if path.is_file():
        if path.suffix.lower() == '.md':
            markdown_files.append(path)
        else:
            print(f"Warning: {path} is not a markdown file")
    elif path.is_dir():
        if recursive:
            markdown_files.extend(path.rglob("*.md"))
        else:
            markdown_files.extend(path.glob("*.md"))
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)
    
    return sorted(markdown_files)

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert Markdown files to HTML using GitHub API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert single file
  python md2html.py file.md
  python md2html.py docs/README.md
  
  # Convert single file with custom output
  python md2html.py file.md -o output.html
  python md2html.py file.md --output-dir ./html/
  
  # Convert all markdown files in current directory
  python md2html.py
  
  # Convert all markdown files in specific directory (recursive)
  python md2html.py -d docs/
  python md2html.py --directory ./documentation/
  
  # Convert files in directory (non-recursive)
  python md2html.py -d docs/ --no-recursive
  
  # Use custom template
  python md2html.py -t custom_template.html file.md
  
  # Convert multiple specific files
  python md2html.py file1.md file2.md docs/file3.md

Environment Variables:
  GITHUB_TOKEN    GitHub personal access token for API authentication
                  (optional but recommended to avoid rate limits)

Template Variables:
  {{ title }}     Replaced with extracted document title
  {{ article }}   Replaced with converted HTML content

Created for gpt4free (g4f) documentation system.
'''
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='Markdown files to convert (if none specified, converts current directory)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=Path,
        help='Convert all .md files in directory'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path (only for single file conversion)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for converted files'
    )
    
    parser.add_argument(
        '-t', '--template',
        type=Path,
        default='template.html',
        help='HTML template file (default: template.html)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories when using --directory'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    return parser

def main():
    """Main conversion function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get GitHub token
    token = get_github_token()
    
    # Determine template path
    if args.template.is_absolute():
        template_path = args.template
    else:
        template_path = Path(__file__).parent / args.template
    
    # Load template
    template = load_template(template_path)
    
    # Determine what to convert
    markdown_files = []
    
    if args.files:
        # Convert specific files
        for file_str in args.files:
            file_path = Path(file_str)
            if file_path.exists():
                if file_path.is_file() and file_path.suffix.lower() == '.md':
                    markdown_files.append(file_path)
                else:
                    print(f"Warning: {file_path} is not a markdown file")
            else:
                print(f"Error: {file_path} does not exist")
                sys.exit(1)
    elif args.directory:
        # Convert directory
        recursive = not args.no_recursive
        markdown_files = find_markdown_files(args.directory, recursive)
    else:
        # Convert current directory
        current_dir = Path.cwd()
        markdown_files = find_markdown_files(current_dir, recursive=True)
    
    if not markdown_files:
        print("No markdown files found to convert.")
        return
    
    # Validate arguments
    if args.output and len(markdown_files) > 1:
        print("Error: --output can only be used with single file conversion")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(markdown_files)} markdown files to process:")
        for f in markdown_files:
            print(f"  - {f}")
        print()
    else:
        print(f"Found {len(markdown_files)} markdown files to process")
    
    # Process files
    successful = 0
    failed = 0
    
    for file_path in markdown_files:
        # Determine output location
        output_dir = None
        if args.output and len(markdown_files) == 1:
            # Single file with specific output
            output_path = args.output
            if process_single_file_with_output(file_path, template, output_path, token):
                successful += 1
            else:
                failed += 1
        else:
            # Regular processing
            if args.output_dir:
                output_dir = args.output_dir
            
            if process_file(file_path, template, output_dir, token):
                successful += 1
            else:
                failed += 1
        
        # Small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    # Summary
    print(f"\nConversion complete:")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

def process_single_file_with_output(file_path: Path, template: str, output_path: Path, token: Optional[str] = None) -> bool:
    """Process single file with specific output path."""
    try:
        # Read markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"Warning: Empty file {file_path}")
            return False
        
        # Extract title
        title = extract_title(content)
        print(f"Processing: {file_path.name} -> Title: {title}")
        
        # Convert to HTML
        html = convert_markdown_to_html(content, token)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate final HTML
        final_html = template.replace("{{ article }}", html).replace("{{ title }}", title)
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"✓ Created: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

if __name__ == "__main__":
    main()
