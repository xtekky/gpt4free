from __future__ import annotations

"""
This module defines the command-line interface (CLI) entrypoint for g4f,
including:
- API server mode
- GUI mode
- Client mode
- MCP (Model Context Protocol) mode

It provides argument parsers for each mode and executes the appropriate
runtime function depending on the CLI arguments.
"""

import argparse
import os
import sys
from argparse import ArgumentParser

# Local imports (within g4f package)
from .client import get_parser, run_client_args
from ..requests import BrowserConfig
from ..gui.run import gui_parser, run_gui_args
from ..config import DEFAULT_PORT, DEFAULT_TIMEOUT, DEFAULT_STREAM_TIMEOUT
from ..Provider.needs_auth.Antigravity import cli_main as antigravity_cli_main
from ..Provider.qwen.QwenCode import cli_main as qwen_cli_main
from ..Provider.github.GithubCopilot import cli_main as github_cli_main
from g4f.Provider.needs_auth.GeminiCLI import cli_main as gemini_cli_main
from .. import Provider
from .. import cookies


# --------------------------------------------------------------
#  API PARSER
# --------------------------------------------------------------
def get_api_parser(exit_on_error: bool = True) -> ArgumentParser:
    """
    Creates and returns the argument parser used for:
        g4f api ...
    """
    api_parser = ArgumentParser(description="Run the API and GUI", exit_on_error=exit_on_error)

    api_parser.add_argument(
        "--bind",
        default=None,
        help=f"The bind address (default: 0.0.0.0:{DEFAULT_PORT})."
    )
    api_parser.add_argument(
        "--port", "-p",
        default=None,
        help=f"Port for the API server (default: {DEFAULT_PORT})."
    )
    api_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable verbose logging."
    )

    # Deprecated GUI flag but kept for compatibility
    api_parser.add_argument(
        "--gui", "-g",
        default=None,
        action="store_true",
        help="(deprecated) Use --no-gui instead."
    )

    api_parser.add_argument(
        "--no-gui", "-ng",
        default=False,
        action="store_true",
        help="Run API without the GUI."
    )

    api_parser.add_argument(
        "--model",
        default=None,
        help="Default model for chat completion (incompatible with reload/workers)."
    )

    # Providers for chat completion
    api_parser.add_argument(
        "--provider",
        choices=[p.__name__ for p in Provider.__providers__ if p.working],
        default=None,
        help="Default provider for chat completion."
    )

    # Providers for image generation
    api_parser.add_argument(
        "--media-provider",
        choices=[
            p.__name__ for p in Provider.__providers__
            if p.working and bool(getattr(p, "image_models", False))
        ],
        default=None,
        help="Default provider for image generation."
    )

    api_parser.add_argument(
        "--proxy",
        default=None,
        help="Default HTTP proxy."
    )

    api_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes."
    )

    api_parser.add_argument(
        "--disable-colors",
        action="store_true",
        help="Disable colorized output."
    )

    api_parser.add_argument(
        "--ignore-cookie-files",
        action="store_true",
        help="Do not read .har or cookie files."
    )

    api_parser.add_argument(
        "--cookies-dir",
        type=str,
        default=None,
        help="Custom directory for cookies/HAR files (overrides default)."
    )

    api_parser.add_argument(
        "--g4f-api-key",
        type=str,
        default=None,
        help="Authentication key for your API."
    )

    api_parser.add_argument(
        "--ignored-providers",
        nargs="+",
        choices=[p.__name__ for p in Provider.__providers__ if p.working],
        default=[],
        help="Providers to ignore during request processing."
    )

    api_parser.add_argument(
        "--cookie-browsers",
        nargs="+",
        choices=[browser.__name__ for browser in cookies.BROWSERS],
        default=[],
        help="Browsers to fetch cookies from."
    )

    api_parser.add_argument("--reload", action="store_true", help="Enable hot reload.")
    api_parser.add_argument("--demo", action="store_true", help="Enable demo mode.")

    api_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Default request timeout in seconds."
    )

    api_parser.add_argument(
        "--stream-timeout",
        type=int,
        default=DEFAULT_STREAM_TIMEOUT,
        help="Default streaming timeout in seconds."
    )

    api_parser.add_argument("--ssl-keyfile", type=str, default=None, help="SSL key file.")
    api_parser.add_argument("--ssl-certfile", type=str, default=None, help="SSL cert file.")
    api_parser.add_argument("--log-config", type=str, default=None, help="Path to log config.")
    api_parser.add_argument("--access-log", action="store_true", default=True, help="Enable access logging.")
    api_parser.add_argument("--no-access-log", dest="access_log", action="store_false", help="Disable access logging.")

    api_parser.add_argument(
        "--browser-port",
        type=int,
        help="Port for browser automation tool."
    )

    api_parser.add_argument(
        "--browser-host",
        type=str,
        default="127.0.0.1",
        help="Host for browser automation tool."
    )

    return api_parser


# --------------------------------------------------------------
#  API RUNNER
# --------------------------------------------------------------
def run_api_args(args):
    """
    Runs the API server using the parsed CLI arguments.
    """
    from g4f.api import AppConfig, run_api

    # Apply configuration
    AppConfig.set_config(
        ignore_cookie_files=args.ignore_cookie_files,
        ignored_providers=args.ignored_providers,
        g4f_api_key=args.g4f_api_key,
        provider=args.provider,
        media_provider=args.media_provider,
        proxy=args.proxy,
        model=args.model,
        gui=not args.no_gui,
        demo=args.demo,
        timeout=args.timeout,
        stream_timeout=args.stream_timeout
    )

    # Browser automation config
    if args.browser_port:
        BrowserConfig.port = args.browser_port
        BrowserConfig.host = args.browser_host

    # Custom cookie browsers
    if args.cookie_browsers:
        cookies.BROWSERS = [cookies[b] for b in args.cookie_browsers]

    # Allow overriding the cookies directory from CLI
    if getattr(args, "cookies_dir", None):
        # create dir if it doesn't exist and update config
        try:
            os.makedirs(args.cookies_dir, exist_ok=True)
        except Exception:
            pass
        cookies.set_cookies_dir(args.cookies_dir)

    # Launch server
    run_api(
        bind=args.bind,
        port=args.port,
        debug=args.debug,
        workers=args.workers,
        use_colors=not args.disable_colors,
        reload=args.reload,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        log_config=args.log_config,
        access_log=args.access_log,
    )


# --------------------------------------------------------------
#  MCP PARSER
# --------------------------------------------------------------
def get_mcp_parser(exit_on_error: bool = True) -> ArgumentParser:
    """
    Parser for:
        g4f mcp ...
    """
    mcp_parser = ArgumentParser(description="Run the MCP (Model Context Protocol) server", exit_on_error=exit_on_error)
    mcp_parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose logging.")
    mcp_parser.add_argument("--http", action="store_true", help="Use HTTP instead of stdio.")
    mcp_parser.add_argument("--host", default="0.0.0.0", help="HTTP server host.")
    mcp_parser.add_argument("--port", type=int, default=8765, help="HTTP server port.")
    mcp_parser.add_argument("--origin", type=str, default=None, help="CORS origin.")
    return mcp_parser


def run_mcp_args(args):
    """
    Runs the MCP server with the chosen transport method.
    """
    from ..mcp.server import main as mcp_main
    mcp_main(
        http=args.http,
        host=args.host,
        port=args.port,
        origin=args.origin
    )

def get_auth_parser(exit_on_error: bool = True) -> ArgumentParser:
    auth_parser = ArgumentParser(description="Manage authentication for providers", exit_on_error=exit_on_error)
    auth_parser.add_argument("provider", choices=["gemini-cli", "antigravity", "qwencode", "github-copilot"], help="The provider to authenticate with")
    auth_parser.add_argument("action", nargs="?", choices=["status", "login", "logout"], default="login", help="Action to perform (default: login)")
    return auth_parser

# --------------------------------------------------------------
#  MAIN ENTRYPOINT
# --------------------------------------------------------------
def main():
    """
    Main entry function exposed via CLI (e.g. g4f).
    Handles selecting: api / gui / client / mcp
    """
    parser = argparse.ArgumentParser(description="Run gpt4free", exit_on_error=False)
    parser.add_argument("--install-autocomplete", action="store_true", help="Install Bash autocompletion for g4f CLI.")
    args, remaining = parser.parse_known_args()
    if args.install_autocomplete:
        generate_autocomplete()
        return
    

    mode_parser = ArgumentParser(description="Select mode to run g4f in.", exit_on_error=False)
    mode_parser.add_argument("mode", nargs="?", choices=["api", "gui", "client", "mcp", "auth"], default="api", help="Mode to run g4f in (default: api).")
    
    args, remaining = mode_parser.parse_known_args(remaining)
    try:
        if args.mode == "auth":
            parser = get_auth_parser()
            args, remaining = parser.parse_known_args(remaining)
            print(f"Handling auth for provider: {args.provider}, action: {args.action}")
            handle_auth(args.provider, args.action, remaining)
            return
        elif args.mode == "api":
            parser = get_api_parser()
            args = parser.parse_args(remaining)
            run_api_args(args)
        elif args.mode == "gui":
            parser = gui_parser()
            args = parser.parse_args(remaining)
            run_gui_args(args)
        elif args.mode == "client":
            parser = get_parser()
            args = parser.parse_args(remaining)
            run_client_args(args)
        elif args.mode == "mcp":
            parser = get_mcp_parser()
            args = parser.parse_args(remaining)
            run_mcp_args(args)
        else:
            # No mode provided
            raise argparse.ArgumentError(
                None,
                "No valid mode specified. Use 'api', 'gui', 'client', 'mcp', or 'auth'."
            )

    except argparse.ArgumentError:
        # Fallback chain:
        # 1. Try client mode
        try:
            run_client_args(
                get_parser(exit_on_error=False).parse_args(),
                exit_on_error=False
            )
        except argparse.ArgumentError:
            # 2. Try API mode with default arguments
            run_api_args(get_api_parser().parse_args())

def generate_autocomplete():
    # Top-level commands and their subcommands/options
    commands = ["api", "gui", "client", "mcp", "auth"]
    auth_providers = ["gemini-cli", "antigravity", "qwencode", "github-copilot"]
    auth_subcommands = ["status", "login"]
    # Options for each command
    api_args = ["--bind", "--port", "--debug", "--gui", "--no-gui", "--model", "--provider", "--media-provider", "--proxy", "--workers", "--disable-colors", "--ignore-cookie-files", "--cookies-dir", "--g4f-api-key", "--ignored-providers", "--cookie-browsers", "--reload", "--demo", "--timeout", "--stream-timeout", "--ssl-keyfile", "--ssl-certfile", "--log-config", "--access-log", "--no-access-log", "--browser-port", "--browser-host"]
    gui_args = ["--debug"]
    client_args = ["--debug"]
    mcp_args = ["--debug", "--http", "--host", "--port", "--origin"]
    global_args = ["--install-autocomplete"]
    bash_completion_script = f"""
_g4f_completions() {{
    local cur prev words cword
    _get_comp_words_by_ref -n : cur prev words cword
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=($(compgen -W '{' '.join(commands + global_args)}' -- "$cur"))
    elif [[ $prev == auth && $cword -eq 2 ]]; then
        COMPREPLY=($(compgen -W '{' '.join(auth_providers)}' -- "$cur"))
    elif [[ $prev =~ ^(gemini-cli|antigravity|qwencode|github-copilot)$ && $cword -eq 3 ]]; then
        COMPREPLY=($(compgen -W '{' '.join(auth_subcommands)}' -- "$cur"))
    elif [[ $words[1] == api ]]; then
        local opts="{' '.join(api_args)}"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
    elif [[ $words[1] == gui ]]; then
        local opts="{' '.join(gui_args)}"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
    elif [[ $words[1] == client ]]; then
        local opts="{' '.join(client_args)}"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
    elif [[ $words[1] == mcp ]]; then
        local opts="{' '.join(mcp_args)}"
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
    fi
}}
complete -F _g4f_completions g4f
"""
    completion_file = os.path.expanduser("~/.g4f_bash_completion")
    with open(completion_file, "w") as f:
        f.write(bash_completion_script)
    print(f"Bash completion script written to {completion_file}. Source it in your .bashrc or .bash_profile.")


def handle_auth(provider, action, remaining):
    if provider == "gemini-cli":
        sys.exit(gemini_cli_main([action] + remaining))
    elif provider == "antigravity":
        sys.exit(antigravity_cli_main([action] + remaining))
    elif provider == "qwencode":
        sys.exit(qwen_cli_main([action] + remaining))
    elif provider == "github-copilot":
        sys.exit(github_cli_main([action] + remaining))
    else:
        print(f"Provider {provider} not supported yet.")