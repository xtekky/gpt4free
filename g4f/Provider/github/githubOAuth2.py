import time
from typing import Dict, Optional, Union

import aiohttp

from .stubs import IGithubOAuth2Client, GithubCredentials, ErrorDataDict
from .sharedTokenManager import SharedTokenManager


# GitHub OAuth endpoints
GITHUB_DEVICE_CODE_ENDPOINT = "https://github.com/login/device/code"
GITHUB_TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"

# GitHub Copilot OAuth Client ID (VS Code Extension)
GITHUB_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Scopes needed for Copilot
GITHUB_COPILOT_SCOPE = "read:user"

TOKEN_REFRESH_BUFFER_MS = 30 * 1000  # 30 seconds


def object_to_urlencoded(data: Dict[str, str]) -> str:
    return "&".join([f"{k}={v}" for k, v in data.items()])


def isDeviceAuthorizationSuccess(response: Union[Dict, ErrorDataDict]) -> bool:
    return "device_code" in response


def isDeviceTokenSuccess(response: Union[Dict, ErrorDataDict]) -> bool:
    return (
        "access_token" in response
        and response["access_token"]
        and isinstance(response["access_token"], str)
        and len(response["access_token"]) > 0
    )


def isDeviceTokenPending(response: Union[Dict, ErrorDataDict]) -> bool:
    return response.get("error") == "authorization_pending"


def isSlowDown(response: Union[Dict, ErrorDataDict]) -> bool:
    return response.get("error") == "slow_down"


def isErrorResponse(response: Union[Dict, ErrorDataDict]) -> bool:
    return "error" in response and response.get("error") not in ["authorization_pending", "slow_down"]


class GithubOAuth2Client(IGithubOAuth2Client):
    def __init__(self, client_id: str = GITHUB_COPILOT_CLIENT_ID):
        self.client_id = client_id
        self.credentials: GithubCredentials = GithubCredentials()
        self.sharedManager = SharedTokenManager.getInstance()

    def setCredentials(self, credentials: GithubCredentials):
        self.credentials = credentials

    def getCredentials(self) -> GithubCredentials:
        return self.credentials

    async def getAccessToken(self) -> Dict[str, Optional[str]]:
        try:
            credentials = await self.sharedManager.getValidCredentials(self)
            return {"token": credentials.get("access_token")}
        except Exception:
            # fallback to internal credentials if valid
            if (
                self.credentials.get("access_token")
                and self.isTokenValid(self.credentials)
            ):
                return {"token": self.credentials["access_token"]}
            return {"token": None}

    async def requestDeviceAuthorization(self, options: dict) -> Union[Dict, ErrorDataDict]:
        """
        Request device authorization from GitHub.
        
        Returns:
            dict with device_code, user_code, verification_uri, expires_in, interval
        """
        body_data = {
            "client_id": self.client_id,
            "scope": options.get("scope", GITHUB_COPILOT_SCOPE),
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GITHUB_DEVICE_CODE_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=object_to_urlencoded(body_data)
            ) as resp:
                resp_json = await resp.json()
                
                if resp.status != 200:
                    raise Exception(f"Device authorization failed {resp.status}: {resp_json}")
                    
                if not isDeviceAuthorizationSuccess(resp_json):
                    raise Exception(
                        f"Device authorization error: {resp_json.get('error')} - {resp_json.get('error_description')}"
                    )
                    
                return resp_json

    async def pollDeviceToken(self, options: dict) -> Union[Dict, ErrorDataDict]:
        """
        Poll for device token from GitHub.
        
        Args:
            options: dict with device_code
            
        Returns:
            dict with access_token, token_type, scope or status=pending
        """
        body_data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": self.client_id,
            "device_code": options["device_code"],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GITHUB_TOKEN_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=object_to_urlencoded(body_data)
            ) as resp:
                resp_json = await resp.json()
                
                # Check for OAuth RFC 8628 responses
                if "error" in resp_json:
                    if resp_json["error"] == "authorization_pending":
                        return {"status": "pending"}
                    if resp_json["error"] == "slow_down":
                        return {"status": "pending", "slowDown": True}
                    if resp_json["error"] == "expired_token":
                        raise Exception("Device code expired. Please try again.")
                    if resp_json["error"] == "access_denied":
                        raise Exception("Authorization was denied by the user.")
                    raise Exception(f"Token poll failed: {resp_json.get('error')} - {resp_json.get('error_description')}")
                
                return resp_json

    def isTokenValid(self, credentials: GithubCredentials) -> bool:
        """GitHub tokens don't expire by default, but we track expiry_date if set"""
        if not credentials.get("access_token"):
            return False
        expiry_date = credentials.get("expiry_date")
        if expiry_date is None:
            # GitHub tokens don't expire unless explicitly set
            return True
        return time.time() * 1000 < expiry_date - TOKEN_REFRESH_BUFFER_MS
