#!/usr/bin/env python3

import os
import sys
import asyncio
import json
import argparse
import traceback
import requests

from pathlib import Path
from typing import Optional, List, Dict
from g4f.client import AsyncClient
from g4f.providers.response import JsonConversation, MediaResponse, is_content
from g4f.cookies import set_cookies_dir, read_cookie_files
from g4f.Provider import ProviderUtils
from g4f.image import extract_data_uri, is_accepted_format
from g4f.image.copy_images import get_media_dir
from g4f.client.helper import filter_markdown
from g4f.errors import MissingRequirementsError

try:
    from g4f.integration.markitdown import MarkItDown
    has_markitdown = True
except ImportError:
    has_markitdown = False

from g4f.config import CONFIG_DIR, COOKIES_DIR
from g4f import debug

CONVERSATION_FILE = CONFIG_DIR / "conversation.json"


class ConversationManager:
    """Manages conversation history and state."""
    def __init__(
        self,
        file_path: Optional[Path] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_messages: int = 5
    ) -> None:
        self.file_path = file_path
        self.model = model
        self.provider = provider
        self.max_messages = max_messages
        self.conversation: Optional[JsonConversation] = None
        self.history: List[Dict[str, str]] = []
        self.data: Dict = {}
        self._load()

    def _load(self) -> None:
        if not self.file_path or not self.file_path.is_file():
            return
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if self.model is None:
                self.model = data.get("model")
            if self.provider is None:
                self.provider = data.get("provider")
            self.data = data.get("data", {})
            if self.provider and self.data.get(self.provider):
                self.conversation = JsonConversation(**self.data[self.provider])
            elif not self.provider and self.data:
                self.conversation = JsonConversation(**self.data)
            self.history = data.get("items", [])
        except Exception as e:
            print(f"Error loading conversation: {e}", file=sys.stderr)

    def save(self) -> None:
        if not self.file_path:
            return
        try:
            if self.conversation and self.provider:
                self.data[self.provider] = self.conversation.get_dict()
            elif self.conversation:
                self.data.update(self.conversation.get_dict())
            payload = {
                "model": self.model,
                "provider": self.provider,
                "data": self.data,
                "items": self.history
            }
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation: {e}", file=sys.stderr)

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        result = []
        for item in self.history[-self.max_messages:]:
            if item.get("role") in ["user", "system"] or result:
                result.append(item)
        return result

async def stream_response(
    client: AsyncClient,
    input_text,
    conversation: ConversationManager,
    output_file: Optional[Path] = None,
    instructions: Optional[str] = None
) -> None:
    media = None
    if isinstance(input_text, tuple):
        media, input_text = input_text

    if instructions:
        conversation.add_message("system", instructions)

    conversation.add_message("user", input_text)

    create_args = {
        "model": conversation.model,
        "messages": conversation.get_messages(),
        "stream": True,
        "media": media,
        "conversation": conversation.conversation,
    }

    response_tokens = []
    last_chunk = None
    async for chunk in client.chat.completions.create(**create_args):
        last_chunk = chunk
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if is_content(delta):
            response_tokens.append(delta)
        print(delta, end="", flush=True)
    print()

    if last_chunk and hasattr(last_chunk, "conversation"):
        conversation.conversation = last_chunk.conversation

    media_chunk = next((t for t in response_tokens if isinstance(t, MediaResponse)), None)
    text_response = ""
    if media_chunk:
        text_response = response_tokens[0] if len(response_tokens) == 1 else "".join(str(t) for t in response_tokens)
    else:
        text_response = "".join(str(t) for t in response_tokens)

    if output_file:
        if save_content(text_response, media_chunk, str(output_file)):
            print(f"\n→ Response saved to '{output_file}'")

    if text_response:
        conversation.add_message("assistant", text_response)
    else:
        raise RuntimeError("No response received")


def save_content(content, media: Optional[MediaResponse], filepath: str, allowed_types=None) -> bool:
    if media:
        for url in media.urls:
            if url.startswith(("http://", "https://")):
                try:
                    resp = requests.get(url, cookies=media.get("cookies"), headers=media.get("headers"))
                    if resp.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        return True
                except Exception as e:
                    print(f"Error fetching media '{url}': {e}", file=sys.stderr)
                return False
            else:
                content = url
                break
    if hasattr(content, "data"):
        content = content.data
    if not content:
        print("\nNo content to save.", file=sys.stderr)
        return False
    if content.startswith("data:"):
        with open(filepath, "wb") as f:
            f.write(extract_data_uri(content))
        return True
    if content.startswith("/media/"):
        src = content.replace("/media", get_media_dir()).split("?")[0]
        os.rename(src, filepath)
        return True
    filtered = filter_markdown(content, allowed_types)
    if filtered:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(filtered)
        return True
    print("\nUnable to save content.", file=sys.stderr)
    return False

def get_parser():
    parser = argparse.ArgumentParser(
        description="G4F CLI client with conversation history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--debug', action='store_true', help="Verbose debug")
    parser.add_argument('-p', '--provider', default=None,
        help=f"Provider to use: {', '.join(k for k,v in ProviderUtils.convert.items() if v.working)}")
    parser.add_argument('-m', '--model', help="Model name")
    parser.add_argument('-O', '--output', type=Path,
        help="Save assistant output to FILE (text or media)")
    parser.add_argument('-i', '--instructions', help="System instructions")
    parser.add_argument('-c', '--cookies-dir', type=Path, default=COOKIES_DIR,
        help="Cookies/HAR directory")
    parser.add_argument('--conversation-file', type=Path, default=CONVERSATION_FILE,
        help="Conversation JSON")
    parser.add_argument('-C', '--clear-history', action='store_true', help="Wipe history")
    parser.add_argument('-N', '--no-config', action='store_true', help="Skip loading history")
    # <-- updated -e/--edit to take an optional filename
    parser.add_argument(
        '-e', '--edit',
        type=Path,
        metavar='FILE',
        help="If FILE given: send its contents and overwrite it with AI's reply."
    )
    parser.add_argument('--max-messages', type=int, default=5,
        help="Max user+assistant turns in context")
    parser.add_argument('input', nargs='*',
        help="URLs, image paths or plain text")
    return parser


async def run_args(input_val, args):
    try:
        # ensure dirs
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.conversation_file:
            args.conversation_file.parent.mkdir(parents=True, exist_ok=True)
        args.cookies_dir.mkdir(parents=True, exist_ok=True)

        if args.debug:
            debug.logging = True

        conv = ConversationManager(
            None if args.no_config else args.conversation_file,
            model=args.model,
            provider=args.provider,
            max_messages=args.max_messages
        )
        if args.clear_history:
            conv.history = []
            conv.conversation = None

        set_cookies_dir(str(args.cookies_dir))
        read_cookie_files()

        client = AsyncClient(provider=conv.provider)

        if isinstance(args.edit, Path):
            file_to_edit = args.edit
            if not file_to_edit.exists():
                print(f"ERROR: file not found: {file_to_edit}", file=sys.stderr)
                sys.exit(1)
            text = file_to_edit.read_text(encoding="utf-8")
            # we will both send and overwrite this file
            input_val = f"```file: {file_to_edit}\n{text}\n```\n" + (input_val[1] if isinstance(input_val, tuple) else input_val)
            output_target = file_to_edit
        else:
            # normal, non-edit mode
            output_target = args.output

        await stream_response(client, input_val, conv, output_target, args.instructions)
        conv.save()

    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def run_client_args(args):
    input_txt = ""
    media = []
    rest = 0

    for idx, tok in enumerate(args.input):
        if tok.startswith(("http://","https://")):
            # same URL logic...
            resp = requests.head(tok, allow_redirects=True)
            if resp.ok and resp.headers.get("Content-Type","").startswith("image"):
                media.append(tok)
            else:
                if not has_markitdown:
                    raise MissingRequirementsError("Install markitdown")
                md = MarkItDown()
                txt = md.convert_url(tok).text_content
                input_txt += f"\n```source: {tok}\n{txt}\n```\n"
        elif os.path.isfile(tok):
            head = Path(tok).read_bytes()[:12]
            try:
                if is_accepted_format(head):
                    media.append(Path(tok))
                    is_img = True
                else:
                    is_img = False
            except ValueError:
                is_img = False
            if not is_img:
                txt = Path(tok).read_text(encoding="utf-8")
                input_txt += f"\n```file: {tok}\n{txt}\n```\n"
        else:
            rest = idx
            break
        rest = idx + 1

    tail = args.input[rest:]
    if tail:
        input_txt = " ".join(tail) + "\n" + input_txt

    if not sys.stdin.isatty() and not input_txt:
        input_txt = sys.stdin.read()

    if media:
        val = (media, input_txt)
    else:
        val = input_txt.strip()

    if not val:
        print("No input provided. Use -h.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_args(val, args))


if __name__ == "__main__":
    run_client_args(get_parser().parse_args())