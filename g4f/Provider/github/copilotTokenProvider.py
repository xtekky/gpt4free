"""
GitHub Copilot Token Provider

This module handles the retrieval of Copilot API tokens using GitHub OAuth credentials.
"""
from typing import Dict, Optional
import aiohttp
import time

from .githubOAuth2 import GithubOAuth2Client
from .sharedTokenManager import SharedTokenManager, TokenManagerError


# Editor/Plugin version headers required by Copilot API
EDITOR_VERSION = "vscode/1.95.0"
EDITOR_PLUGIN_VERSION = "copilot/1.250.0"
USER_AGENT = "GithubCopilot/1.250.0"
API_VERSION = "2024-12-15"

class CopilotTokenProvider:
    """Provides Copilot API tokens from GitHub OAuth credentials."""
    
    # Copilot token endpoint
    COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
    
    def __init__(self, github_client: GithubOAuth2Client = None):
        self.github_client = github_client or GithubOAuth2Client()
        self.shared_manager = SharedTokenManager.getInstance()
        self._copilot_token = None
        self._copilot_token_expires_at = 0
    
    async def get_copilot_token(self) -> Optional[str]:
        """
        Get a valid Copilot API token.
        
        This exchanges the GitHub OAuth token for a Copilot-specific token.
        
        Returns:
            The Copilot API token, or None if not available.
        """
        # Check if we have a valid cached token
        if self._copilot_token and time.time() < self._copilot_token_expires_at - 60:
            return self._copilot_token
        
        # Get GitHub OAuth token
        github_creds = await self.shared_manager.getValidCredentials(self.github_client)
        if not github_creds or not github_creds.get("access_token"):
            raise TokenManagerError("NO_TOKEN", "No GitHub OAuth token available. Please login first.")
        
        github_token = github_creds["access_token"]
        
        # Exchange for Copilot token
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.COPILOT_TOKEN_URL,
                headers={
                    "Authorization": f"token {github_token}",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                    "Editor-Version": EDITOR_VERSION,
                    "Editor-Plugin-Version": EDITOR_PLUGIN_VERSION,
                    "Openai-Organization": "github-copilot",
                    "X-GitHub-Api-Version": API_VERSION,
                }
            ) as resp:
                if resp.status == 401:
                    raise TokenManagerError(
                        "AUTH_FAILED",
                        "GitHub token is invalid or expired. Please login again."
                    )
                if resp.status != 200:
                    text = await resp.text()
                    raise TokenManagerError(
                        "TOKEN_ERROR",
                        f"Failed to get Copilot token: {resp.status} - {text}"
                    )
                
                data = await resp.json()
                self._copilot_token = data.get("token")
                
                # Parse expiration
                expires_at = data.get("expires_at")
                if expires_at:
                    try:
                        # Parse ISO format datetime
                        from datetime import datetime
                        dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        self._copilot_token_expires_at = dt.timestamp()
                    except Exception:
                        # Default to 30 minutes from now if parsing fails
                        self._copilot_token_expires_at = time.time() + 1800
                else:
                    self._copilot_token_expires_at = time.time() + 1800
                
                return self._copilot_token
    
    async def get_valid_token(self) -> Dict[str, Optional[str]]:
        """
        Get valid credentials for the Copilot API.
        
        Returns:
            Dict with 'token' and optionally 'endpoint'
        """
        token = await self.get_copilot_token()
        return {
            "token": token,
            "endpoint": "https://api.githubcopilot.com"
        }
