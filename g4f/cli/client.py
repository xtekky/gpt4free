#!/usr/bin/env python3

import os
import sys
import asyncio
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from g4f.client import AsyncClient
from g4f.providers.response import JsonConversation, is_content
from g4f.cookies import set_cookies_dir, read_cookie_files
from g4f.Provider import ProviderUtils
from g4f.image import extract_data_uri, is_accepted_format
from g4f.image.copy_images import get_media_dir
from g4f.client.helper import filter_markdown
from g4f import debug

# Platform-appropriate directories
def get_config_dir() -> Path:
    """Get platform-appropriate config directory."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    else:  # Linux and other UNIX-like
        return Path.home() / ".config"

CONFIG_DIR = get_config_dir() / "g4f-cli"
COOKIES_DIR = CONFIG_DIR / "cookies"
CONVERSATION_FILE = CONFIG_DIR / "conversation.json"

class ConversationManager:
    """Manages conversation history and state."""
    
    def __init__(self, file_path: Path, model: Optional[str] = None, provider: Optional[str] = None) -> None:
        self.file_path = file_path
        self.model: Optional[str] = model
        self.provider: Optional[str] = provider
        self.conversation = None
        self.history: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Load conversation from file."""
        if not self.file_path.is_file():
            return

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.model = data.get("model") if self.model is None else self.model
                self.provider = data.get("provider") if self.provider is None else self.provider
                if not self.provider:
                    self.provider = None
                self.data = data.get("data", {})
                if self.provider and self.data.get(self.provider):
                    self.conversation = JsonConversation(**self.data.get(self.provider))
                elif not self.provider and self.data:
                    self.conversation = JsonConversation(**self.data)
                self.history = data.get("history", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading conversation: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error loading conversation: {e}", file=sys.stderr)

    def save(self) -> None:
        """Save conversation to file."""
        if self.file_path.exists() and not self.file_path.is_file():
            return

        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                if self.conversation and self.provider:
                    self.data[self.provider] = self.conversation.get_dict()
                else:
                    self.data = self.conversation.get_dict() if self.conversation else {}
                json.dump({
                    "model": self.model,
                    "provider": self.provider,
                    "data": self.data,
                    "history": self.history
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation: {e}", file=sys.stderr)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.history.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation."""
        return self.history

async def stream_response(
    client: AsyncClient,
    input_text: str,
    conversation: ConversationManager,
    output_file: Optional[Path] = None,
    instructions: Optional[str] = None
) -> None:
    """Stream the response from the API and update conversation."""
    image = None
    if isinstance(input_text, tuple):
        image, input_text = input_text
    
    if instructions:
        # Add system instructions to conversation if provided
        conversation.add_message("system", instructions)

    # Add user message to conversation
    conversation.add_message("user", input_text)

    create_args = {
        "messages": conversation.get_messages(),
        "stream": True,
        "image": image
    }
    
    if conversation.model:
        create_args["model"] = conversation.model

    response_content = []
    last_chunk = None
    async for chunk in client.chat.completions.create(**create_args):
        last_chunk = chunk
        token = chunk.choices[0].delta.content
        if not token:
            continue
        if is_content(token):
            response_content.append(token)
        try:
            for byte in str(token).encode('utf-8'):
                sys.stdout.buffer.write(bytes([byte]))
                sys.stdout.buffer.flush()
        except (IOError, BrokenPipeError) as e:
            print(f"\nError writing to stdout: {e}", file=sys.stderr)
            break
    print("\n", end="")

    conversation.conversation = getattr(last_chunk, 'conversation', None)
    response_content = response_content[0] if len(response_content) == 1 else "".join([str(chunk) for chunk in response_content])
    if output_file:
        if save_content(response_content, output_file):
            print(f"\nResponse saved to {output_file}")

    if response_content:
        # Add assistant message to conversation
        conversation.add_message("assistant", str(response_content))
    else:
        raise RuntimeError("No response received from the API")

def save_content(content, filepath: str, allowed_types = None):
    if hasattr(content, "urls"):
        content = next(iter(content.urls), None) if isinstance(content.urls, list) else content.urls
    elif hasattr(content, "data"):
        content = content.data
    if not content:
        print("\nNo content to save.", file=sys.stderr)
        return False
    if content.startswith("/media/"):
        os.rename(content.replace("/media", get_media_dir()).split("?")[0], filepath)
        return True
    elif content.startswith("data:"):
        with open(filepath, "wb") as f:
            f.write(extract_data_uri(content))
        return True
    elif content.startswith("http://") or content.startswith("https://"):
        import requests
        response = requests.get(content)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return True
    content = filter_markdown(content, allowed_types)
    if content:
        with open(filepath, "w") as f:
            f.write(content)
            return True
    else:
        print("\nNo valid content to save.", file=sys.stderr)
        return False

def get_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="G4F CLI client with conversation history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        '-p', '--provider',
        default=None,
        help=f"Provider to use. Available: {', '.join([key for key, provider in ProviderUtils.convert.items() if provider.working])}."
    )
    parser.add_argument(
        '-m', '--model',
        help="Model to use (provider-specific)"
    )
    parser.add_argument(
        '-O', '--output',
        default=None,
        type=Path,
        metavar='FILE',
        nargs='?',
        help="Output file to save the response file."
    )
    parser.add_argument(
        '-i', '--instructions',
        default=None,
        help="Add custom system instructions."
    )
    parser.add_argument(
        '-c', '--cookies-dir',
        type=Path,
        default=COOKIES_DIR,
        help="Directory containing cookies for authenticated providers"
    )
    parser.add_argument(
        '--conversation-file',
        type=Path,
        default=CONVERSATION_FILE,
        help="File to store/load conversation state"
    )
    parser.add_argument(
        '-C', '--clear-history',
        action='store_true',
        help="Clear conversation history before starting"
    )
    parser.add_argument(
        'input',
        nargs='*',
        help="Input text (or read from stdin)"
    )
    
    return parser

async def run_args(input_text: str, args):
    try:
        # Ensure directories exist
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        args.conversation_file.parent.mkdir(parents=True, exist_ok=True)
        args.cookies_dir.mkdir(parents=True, exist_ok=True)

        if args.debug:
            debug.logging = True
        
        # Initialize conversation manager
        conversation = ConversationManager(args.conversation_file, args.model, args.provider)
        if args.clear_history:
            conversation.history = []
            conversation.conversation = None
        
        # Set cookies directory if specified
        set_cookies_dir(str(args.cookies_dir))
        read_cookie_files()
        
        # Initialize client with selected provider
        client = AsyncClient(provider=conversation.provider)
        
        # Stream response and update conversation
        await stream_response(client, input_text, conversation, args.output, args.instructions)
        
        # Save conversation state
        conversation.save()
    except:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

def run_client_args(args):
    input_text = ""
    if args.input and os.path.isfile(args.input[0]):
        try:
            with open(args.input[0], 'rb') as f:
                if is_accepted_format(f.read(12)):
                    input_text = (Path(args.input[0]), " ".join(args.input[1:]))
        except ValueError:
            # If not a valid image, read as text
            try:
                with open(args.input[0], 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
            except UnicodeDecodeError:
                print(f"Error reading file {args.input[0]} as text. Ensure it is a valid text file.", file=sys.stderr)
                sys.exit(1)
            if len(args.input) > 1:
                input_text = f"{' '.join(args.input[1:])}\n```{os.path.basename(args.input[0])}\n{file_content}\n```"
            else:
                input_text = file_content
    elif args.input:
        input_text = (" ".join(args.input)).strip()
    if not input_text:
        input_text = sys.stdin.read().strip()
    if not input_text:
        print("No input provided. Use -h for help.", file=sys.stderr)
        sys.exit(1)
    # Run the client with provided arguments
    asyncio.run(run_args(input_text, args))

if __name__ == "__main__":
    # Run the client with command line arguments
    run_client_args(get_parser().parse_args())