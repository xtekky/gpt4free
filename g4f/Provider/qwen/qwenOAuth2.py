import base64
import hashlib
import secrets
import uuid
import time
from typing import Dict, Optional, Union

import aiohttp

from .stubs import IQwenOAuth2Client, QwenCredentials, ErrorDataDict
from .sharedTokenManager import SharedTokenManager


QWEN_OAUTH_BASE_URL = "https://chat.qwen.ai"
QWEN_OAUTH_DEVICE_CODE_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/device/code"
QWEN_OAUTH_TOKEN_ENDPOINT = f"{QWEN_OAUTH_BASE_URL}/api/v1/oauth2/token"

QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_SCOPE = "openid profile email model.completion"
QWEN_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

QEN_DIR = ".qwen"
QWEN_CREDENTIAL_FILENAME = "oauth_creds.json"

TOKEN_REFRESH_BUFFER_MS = 30 * 1000  # 30 seconds


def generate_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")


def generate_code_challenge(code_verifier: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(code_verifier.encode())
    digest = sha256.digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def generatePKCEPair():
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    return {"code_verifier": code_verifier, "code_challenge": code_challenge}


def object_to_urlencoded(data: Dict[str, str]) -> str:
    return "&".join([f"{k}={v}" for k, v in data.items()])


def isDeviceAuthorizationSuccess(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return "device_code" in response


def isDeviceTokenSuccess(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return (
        "access_token" in response
        and response["access_token"]
        and isinstance(response["access_token"], str)
        and len(response["access_token"]) > 0
    )


def isDeviceTokenPending(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return response.get("status") == "pending"


def isErrorResponse(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return "error" in response


def isTokenRefreshResponse(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return "access_token" in response and "token_type" in response

class QwenOAuth2Client(IQwenOAuth2Client):
    def __init__(self):
        self.credentials: QwenCredentials = QwenCredentials()
        self.sharedManager = SharedTokenManager.getInstance()

    def setCredentials(self, credentials: QwenCredentials):
        self.credentials = credentials

    def getCredentials(self) -> QwenCredentials:
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
        body_data = {
            "client_id": QWEN_OAUTH_CLIENT_ID,
            "scope": options["scope"],
            "code_challenge": options["code_challenge"],
            "code_challenge_method": options["code_challenge_method"],
        }
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(QWEN_OAUTH_DEVICE_CODE_ENDPOINT, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "x-request-id": str(uuid.uuid4()),
            }, data=object_to_urlencoded(body_data)) as resp:
                resp_json = await resp.json()
                if resp.status != 200:
                    raise Exception(f"Device authorization failed {resp.status}: {resp_json}")
                if not isDeviceAuthorizationSuccess(resp_json):
                    raise Exception(
                        f"Device authorization error: {resp_json.get('error')} - {resp_json.get('error_description')}"
                    )
                return resp_json

    async def pollDeviceToken(self, options: dict) -> Union[Dict, ErrorDataDict]:
        body_data = {
            "grant_type": QWEN_OAUTH_GRANT_TYPE,
            "client_id": QWEN_OAUTH_CLIENT_ID,
            "device_code": options["device_code"],
            "code_verifier": options["code_verifier"],
        }
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(QWEN_OAUTH_TOKEN_ENDPOINT, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }, data=object_to_urlencoded(body_data)) as resp:
                resp_json = await resp.json()
                if resp.status != 200:
                    # Check for OAuth RFC 8628 responses
                    if resp.status == 400:
                        if "error" in resp_json:
                            if resp_json["error"] == "authorization_pending":
                                return {"status": "pending"}
                            if resp_json["error"] == "slow_down":
                                return {"status": "pending", "slowDown": True}
                    raise Exception(f"Token poll failed {resp.status}: {resp_json}")
                return resp_json

    async def refreshAccessToken(self) -> Union[Dict, ErrorDataDict]:
        if not self.credentials.get("refresh_token"):
            raise Exception("No refresh token")
        body_data = {
            "grant_type": "refresh_token",
            "refresh_token": self.credentials["refresh_token"],
            "client_id": QWEN_OAUTH_CLIENT_ID,
        }
        async with aiohttp.ClientSession(headers={"user-agent": ""}) as session:
            async with session.post(QWEN_OAUTH_TOKEN_ENDPOINT, headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }, data=object_to_urlencoded(body_data)) as resp:
                resp_json = await resp.json()
                if resp.status != 200:
                    if resp.status == 400:
                        # Handle token expiration
                        self.credentials = QwenCredentials()
                        raise Exception("Refresh token expired or invalid")
                    raise Exception(f"Token refresh failed {resp.status}: {resp_json}")
                return resp_json

    def isTokenValid(self, credentials: QwenCredentials) -> bool:
        if not credentials.get("expiry_date"):
            return False
        return time.time() * 1000 < credentials["expiry_date"] - TOKEN_REFRESH_BUFFER_MS
