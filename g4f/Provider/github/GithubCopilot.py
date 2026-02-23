from __future__ import annotations

import sys
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional

from ...typing import Messages, AsyncResult
from ...errors import MissingAuthError
from ..template import OpenaiTemplate
from ...providers.asyncio import get_running_loop
from .copilotTokenProvider import CopilotTokenProvider, EDITOR_VERSION, EDITOR_PLUGIN_VERSION, USER_AGENT, API_VERSION
from .sharedTokenManager import TokenManagerError, SharedTokenManager
from .githubOAuth2 import GithubOAuth2Client
from .oauthFlow import launch_browser_for_oauth

class GithubCopilot(OpenaiTemplate):
    """
    GitHub Copilot provider with OAuth authentication.
    
    This provider uses GitHub OAuth device flow for authentication,
    allowing users to authenticate via browser without sharing credentials.
    
    Usage:
        1. Run `g4f auth github-copilot` to authenticate
        2. Use the provider normally after authentication
        
    Example:
        >>> from g4f.client import Client
        >>> from g4f.Provider.github import GithubCopilot
        >>> client = Client(provider=GithubCopilot)
        >>> response = client.chat.completions.create(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    
    label = "GitHub Copilot (OAuth) 🔐"
    url = "https://github.com/copilot"
    login_url = "https://github.com/login"
    working = True
    needs_auth = True
    active_by_default = True
    
    default_model = "gpt-4.1"
    base_url = "https://api.githubcopilot.com"
    
    fallback_models = [
        # GPT-5 Series
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
        
        # GPT-5 Codex (optimized for code)
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        
        # GPT-4 Series
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-0125-preview",
        "gpt-4-o-preview",
        
        # Claude 4 Series
        "claude-opus-4.6",
        "claude-opus-4.6-fast",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
        
        # Gemini Series
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        
        # Grok
        "grok-code-fast-1",
        
        # Legacy GPT-3.5
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        
        # Embeddings
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]
    
    _token_provider: Optional[CopilotTokenProvider] = None

    @classmethod
    def _get_token_provider(cls) -> CopilotTokenProvider:
        if cls._token_provider is None:
            cls._token_provider = CopilotTokenProvider()
        return cls._token_provider

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> AsyncResult:
        """
        Create an async generator for chat completions.
        
        If api_key is provided, it will be used directly.
        Otherwise, OAuth credentials will be used.
        """
        # If no API key provided, use OAuth token
        if not api_key:
            try:
                token_provider = cls._get_token_provider()
                creds = await token_provider.get_valid_token()
                api_key = creds.get("token")
                if not api_key:
                    raise MissingAuthError(
                        "GitHub Copilot OAuth not configured. "
                        "Please run 'g4f auth github-copilot' to authenticate."
                    )
                if not base_url:
                    base_url = creds.get("endpoint", cls.base_url)
            except TokenManagerError as e:
                if "login" in str(e).lower() or "credentials" in str(e).lower():
                    raise MissingAuthError(
                        "GitHub Copilot OAuth not configured. "
                        "Please run 'g4f auth github-copilot' to authenticate."
                    ) from e
                raise
        
        # Use parent class for actual API calls
        async for chunk in super().create_async_generator(
            model,
            messages,
            api_key=api_key,
            base_url=base_url or cls.base_url,
            **kwargs
        ):
            yield chunk

    @classmethod
    def get_models(cls, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: Optional[int] = None):
        # If no API key provided, use OAuth token
        if not api_key:
            try:
                token_provider = cls._get_token_provider()
                get_running_loop(check_nested=True)
                creds = asyncio.run(token_provider.get_valid_token())
                api_key = creds.get("token")
                if not base_url:
                    base_url = creds.get("endpoint", cls.base_url)
            except TokenManagerError as e:
                if "login" in str(e).lower() or "credentials" in str(e).lower():
                    raise MissingAuthError(
                        "GitHub Copilot OAuth not configured. "
                        "Please run 'g4f auth github-copilot' to authenticate."
                    ) from e
                raise
        return super().get_models(api_key, base_url, timeout)

    @classmethod
    def get_headers(cls, stream: bool, api_key: str | None = None, headers: dict[str, str] | None = None) -> dict[str, str]:
        headers_result = super().get_headers(stream, api_key or "", headers or {})
        # Add required Copilot headers
        copilot_headers: dict[str, str] = {
            "Editor-Version": EDITOR_VERSION,
            "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
            "Openai-Organization": "github-copilot",
            "Copilot-Integration-Id": "vscode-chat",
            "X-GitHub-Api-Version": "2024-12-15",
        }
        if headers_result:
            copilot_headers.update(headers_result)
        return copilot_headers

    @classmethod
    async def login(cls, credentials_path: Optional[Path] = None) -> SharedTokenManager:
        """
        Perform interactive OAuth login and save credentials.
        
        Args:
            credentials_path: Path to save credentials (default: g4f cache)
            
        Returns:
            SharedTokenManager with active credentials
            
        Example:
            >>> import asyncio
            >>> from g4f.Provider.github import GithubCopilot
            >>> asyncio.run(GithubCopilot.login())
        """
        print("\n" + "=" * 60)
        print("GitHub Copilot OAuth Login")
        print("=" * 60)
        
        await launch_browser_for_oauth()
        
        shared_manager = SharedTokenManager.getInstance()
        print("=" * 60 + "\n")
        
        return shared_manager

    @classmethod
    def has_credentials(cls) -> bool:
        """Check if valid credentials exist."""
        shared_manager = SharedTokenManager.getInstance()
        try:
            path = shared_manager.getCredentialFilePath()
            return path.exists()
        except Exception:
            return False

    @classmethod
    def get_credentials_path(cls) -> Optional[Path]:
        """Get path to credentials file if it exists."""
        shared_manager = SharedTokenManager.getInstance()
        try:
            path = shared_manager.getCredentialFilePath()
            if path.exists():
                return path
        except Exception:
            pass
        return None

    @classmethod
    async def get_quota(cls, api_key: Optional[str] = None) -> dict:
        """
        Fetch and summarize current GitHub Copilot usage/quota information.
        Returns a dictionary with usage details or raises an exception on failure.
        """
        client = GithubOAuth2Client()
        github_creds = await client.sharedManager.getValidCredentials(client)
        if not github_creds or not github_creds.get("access_token"):
            raise MissingAuthError(
                "GitHub Copilot OAuth not configured. "
                "Please run 'g4f auth github-copilot' to authenticate."
            )
        
        github_token = github_creds["access_token"]
        url = f"https://api.github.com/copilot_internal/user"
        headers = {
            "accept": "application/json",
            "authorization": f"token {github_token}",
            "editor-version": EDITOR_VERSION,
            "editor-plugin-version": EDITOR_PLUGIN_VERSION,
            "user-agent": USER_AGENT,
            "x-github-api-version": API_VERSION,
            "x-vscode-user-agent-library-version": "electron-fetch",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Failed to fetch Copilot usage: {resp.status} {text}")
                usage = await resp.json()
        return usage

async def main(args: Optional[List[str]] = None):
    """CLI entry point for GitHub Copilot OAuth authentication."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GitHub Copilot OAuth Authentication for gpt4free",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s login                    # Interactive device code login
  %(prog)s status                   # Check authentication status
  %(prog)s logout                   # Remove saved credentials
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Login command
    subparsers.add_parser("login", help="Authenticate with GitHub Copilot")
    
    # Status command
    subparsers.add_parser("status", help="Check authentication status")
    
    # Logout command
    subparsers.add_parser("logout", help="Remove saved credentials")
    
    args = parser.parse_args(args)
    
    if args.command == "login":
        try:
            await GithubCopilot.login()
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        print("\nGitHub Copilot OAuth Status")
        print("=" * 40)
        
        if GithubCopilot.has_credentials():
            creds_path = GithubCopilot.get_credentials_path()
            print(f"✓ Credentials found at: {creds_path}")
            
            try:
                with creds_path.open() as f:
                    creds = json.load(f)
                
                expiry = creds.get("expiry_date")
                if expiry:
                    expiry_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(expiry / 1000))
                    if expiry / 1000 > time.time():
                        print(f"  Token expires: {expiry_time}")
                    else:
                        print(f"  Token expired: {expiry_time}")
                
                if creds.get("scope"):
                    print(f"  Scope: {creds['scope']}")
            except Exception as e:
                print(f"  (Could not read credential details: {e})")
        else:
            print("✗ No credentials found")
            print(f"\nRun 'g4f auth github-copilot' to authenticate.")
        
        print()
    
    elif args.command == "logout":
        print("\nGitHub Copilot OAuth Logout")
        print("=" * 40)
        
        removed = False
        
        shared_manager = SharedTokenManager.getInstance()
        path = shared_manager.getCredentialFilePath()
        
        if path.exists():
            path.unlink()
            print(f"✓ Removed: {path}")
            removed = True
        
        # Also try the default location
        default_path = Path.home() / ".github-copilot" / "oauth_creds.json"
        if default_path.exists() and default_path != path:
            default_path.unlink()
            print(f"✓ Removed: {default_path}")
            removed = True
        
        if removed:
            print("\n✓ Credentials removed successfully.")
        else:
            print("No credentials found to remove.")
        
        print()
    
    else:
        parser.print_help()


def cli_main(args: Optional[List[str]] = None):
    """Synchronous CLI entry point for setup.py console_scripts."""
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
