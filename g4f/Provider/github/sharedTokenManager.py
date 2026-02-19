import os
import json
import time
import asyncio
import threading
from typing import Optional, Dict
from pathlib import Path

from ..base_provider import AuthFileMixin
from ... import debug

GITHUB_DIR = ".github-copilot"
GITHUB_CREDENTIAL_FILENAME = "oauth_creds.json"
GITHUB_LOCK_FILENAME = "oauth_creds.lock"
TOKEN_REFRESH_BUFFER_MS = 30 * 1000
CACHE_CHECK_INTERVAL_MS = 1000


class TokenError:
    REFRESH_FAILED = "REFRESH_FAILED"
    NO_REFRESH_TOKEN = "NO_REFRESH_TOKEN"
    LOCK_TIMEOUT = "LOCK_TIMEOUT"
    FILE_ACCESS_ERROR = "FILE_ACCESS_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


class TokenManagerError(Exception):
    def __init__(self, type_: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.type = type_
        self.original_error = original_error


class SharedTokenManager(AuthFileMixin):
    parent = "GithubCopilot"
    _instance: Optional["SharedTokenManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.memory_cache = {
            "credentials": None,
            "file_mod_time": 0,
            "last_check": 0,
        }
        self.refresh_promise = None

    @classmethod
    def getInstance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def getCredentialFilePath(self):
        path = Path(os.path.expanduser(f"~/{GITHUB_DIR}/{GITHUB_CREDENTIAL_FILENAME}"))
        if path.is_file():
            return path
        return SharedTokenManager.get_cache_file()

    def getLockFilePath(self):
        return Path(os.path.expanduser(f"~/{GITHUB_DIR}/{GITHUB_LOCK_FILENAME}"))

    def getCurrentCredentials(self):
        return self.memory_cache.get("credentials")

    def checkAndReloadIfNeeded(self):
        now = int(time.time() * 1000)
        if now - self.memory_cache["last_check"] < CACHE_CHECK_INTERVAL_MS:
            return
        self.memory_cache["last_check"] = now

        try:
            file_path = self.getCredentialFilePath()
            if not file_path.exists():
                self.memory_cache["file_mod_time"] = 0
                return
            stat = file_path.stat()
            file_mod_time = int(stat.st_mtime * 1000)
            if file_mod_time > self.memory_cache["file_mod_time"]:
                self.reloadCredentialsFromFile()
                self.memory_cache["file_mod_time"] = file_mod_time
        except FileNotFoundError:
            self.memory_cache["file_mod_time"] = 0
        except Exception as e:
            self.memory_cache["credentials"] = None
            raise TokenManagerError(TokenError.FILE_ACCESS_ERROR, str(e), e)

    def reloadCredentialsFromFile(self):
        file_path = self.getCredentialFilePath()
        debug.log(f"Reloading credentials from {file_path}")
        try:
            with open(file_path, "r") as fs:
                data = json.load(fs)
                credentials = self.validateCredentials(data)
                self.memory_cache["credentials"] = credentials
        except FileNotFoundError as e:
            self.memory_cache["credentials"] = None
            raise TokenManagerError(TokenError.FILE_ACCESS_ERROR, "Credentials file not found", e) from e
        except json.JSONDecodeError as e:
            self.memory_cache["credentials"] = None
            raise TokenManagerError(TokenError.FILE_ACCESS_ERROR, "Invalid JSON format", e) from e
        except Exception as e:
            self.memory_cache["credentials"] = None
            raise TokenManagerError(TokenError.FILE_ACCESS_ERROR, str(e), e) from e

    def validateCredentials(self, data):
        if not data or not isinstance(data, dict):
            raise ValueError("Invalid credentials format")
        if "access_token" not in data or not isinstance(data["access_token"], str):
            raise ValueError("Invalid credentials: missing access_token")
        if "token_type" not in data or not isinstance(data["token_type"], str):
            raise ValueError("Invalid credentials: missing token_type")
        return data

    def isTokenValid(self, credentials) -> bool:
        """GitHub tokens don't expire by default"""
        if not credentials or not credentials.get("access_token"):
            return False
        expiry_date = credentials.get("expiry_date")
        if expiry_date is None:
            return True
        return time.time() * 1000 < expiry_date - TOKEN_REFRESH_BUFFER_MS

    async def getValidCredentials(self, github_client, force_refresh: bool = False):
        try:
            self.checkAndReloadIfNeeded()

            if (
                self.memory_cache["credentials"]
                and not force_refresh
                and self.isTokenValid(self.memory_cache["credentials"])
            ):
                return self.memory_cache["credentials"]

            if self.refresh_promise:
                return await self.refresh_promise

            # Try to reload credentials from file
            try:
                self.reloadCredentialsFromFile()
                if self.memory_cache["credentials"] and self.isTokenValid(self.memory_cache["credentials"]):
                    return self.memory_cache["credentials"]
            except TokenManagerError:
                pass

            raise TokenManagerError(
                TokenError.FILE_ACCESS_ERROR,
                "No valid credentials found. Please run login first."
            )
        except Exception as e:
            if isinstance(e, TokenManagerError):
                raise
            raise TokenManagerError(TokenError.FILE_ACCESS_ERROR, str(e), e) from e

    async def saveCredentialsToFile(self, credentials: dict):
        """Save credentials to the credential file."""
        file_path = self.getCredentialFilePath()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(credentials, f, indent=2)
        
        self.memory_cache["credentials"] = credentials
        self.memory_cache["file_mod_time"] = int(time.time() * 1000)
        
        debug.log(f"Credentials saved to {file_path}")
