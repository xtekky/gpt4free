from __future__ import annotations

import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

from ...typing import Messages, AsyncResult
from ..template import OpenaiTemplate
from .qwenContentGenerator import QwenContentGenerator
from .qwenOAuth2 import QwenOAuth2Client
from .sharedTokenManager import TokenManagerError, SharedTokenManager
from .oauthFlow import launch_browser_for_oauth

class QwenCode(OpenaiTemplate):
    label = "Qwen Code 🤖"
    url = "https://qwen.ai"
    login_url = "https://github.com/QwenLM/qwen-code"
    working = True
    needs_auth = True
    active_by_default = True
    default_model = "qwen3-coder-plus"
    models = [default_model]
    client = QwenContentGenerator(QwenOAuth2Client())

    @classmethod
    def get_models(cls, **kwargs):
        if cls.live == 0:
            cls.client.shared_manager.checkAndReloadIfNeeded()
            creds = cls.client.shared_manager.getCurrentCredentials()
            if creds:
                cls.client.shared_manager.isTokenValid(creds)
            cls.live += 1
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        base_url: str = None,
        **kwargs
    ) -> AsyncResult:
        try:
            creds = await cls.client.get_valid_token()
            last_chunk = None
            async for chunk in super().create_async_generator(
                model,
                messages,
                api_key=creds.get("token", api_key),
                base_url=creds.get("endpoint", base_url),
                **kwargs
            ):
                if isinstance(chunk, str):
                    if chunk != last_chunk:
                        yield chunk
                    last_chunk = chunk
                else:
                    yield chunk
        except TokenManagerError:
            await cls.client.shared_manager.getValidCredentials(cls.client.qwen_client, True)
            creds = await cls.client.get_valid_token()
            last_chunk = None
            async for chunk in super().create_async_generator(
                model,
                messages,
                api_key=creds.get("token"),
                base_url=creds.get("endpoint"),
                **kwargs
            ):
                if isinstance(chunk, str):
                    if chunk != last_chunk:
                        yield chunk
                    last_chunk = chunk
                else:
                    yield chunk
        except:
            raise

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
            >>> from g4f.Provider.qwen import QwenCode
            >>> asyncio.run(QwenCode.login())
        """
        print("\n" + "=" * 60)
        print("QwenCode OAuth Login")
        print("=" * 60)
        
        await launch_browser_for_oauth()
        
        shared_manager = SharedTokenManager.getInstance()
        print("=" * 60 + "\n")
        
        return shared_manager

    @classmethod
    def has_credentials(cls) -> bool:
        """Check if valid credentials exist."""
        shared_manager = SharedTokenManager.getInstance()
        path = shared_manager.getCredentialFilePath()
        return path.exists()

    @classmethod
    def get_credentials_path(cls) -> Optional[Path]:
        """Get path to credentials file if it exists."""
        shared_manager = SharedTokenManager.getInstance()
        path = shared_manager.getCredentialFilePath()
        if path.exists():
            return path
        return None


async def main():
    """CLI entry point for QwenCode authentication."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QwenCode OAuth Authentication for gpt4free",
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
    subparsers.add_parser("login", help="Authenticate with Qwen")
    
    # Status command
    subparsers.add_parser("status", help="Check authentication status")
    
    # Logout command
    subparsers.add_parser("logout", help="Remove saved credentials")
    
    args = parser.parse_args()
    
    if args.command == "login":
        try:
            await QwenCode.login()
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        print("\nQwenCode Authentication Status")
        print("=" * 40)
        
        if QwenCode.has_credentials():
            creds_path = QwenCode.get_credentials_path()
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
                        print(f"  Token expired: {expiry_time} (will auto-refresh)")
                
                if creds.get("resource_url"):
                    print(f"  Endpoint: {creds['resource_url']}")
            except Exception as e:
                print(f"  (Could not read credential details: {e})")
        else:
            print("✗ No credentials found")
            print(f"\nRun 'g4f-qwencode login' to authenticate.")
        
        print()
    
    elif args.command == "logout":
        print("\nQwenCode Logout")
        print("=" * 40)
        
        removed = False
        
        shared_manager = SharedTokenManager.getInstance()
        path = shared_manager.getCredentialFilePath()
        
        if path.exists():
            path.unlink()
            print(f"✓ Removed: {path}")
            removed = True
        
        # Also try the default location
        default_path = Path.home() / ".qwen" / "oauth_creds.json"
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


def cli_main():
    """Synchronous CLI entry point for setup.py console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()