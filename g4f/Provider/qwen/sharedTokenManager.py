import os
import json
import time
import asyncio
import uuid
from typing import Optional, Dict, Union
from .stubs import IQwenOAuth2Client, ErrorDataDict
from pathlib import Path
import threading

from ..base_provider import AuthFileMixin
from ... import debug

QWEN_DIR = ".qwen"
QWEN_CREDENTIAL_FILENAME = "oauth_creds.json"
QWEN_LOCK_FILENAME = "oauth_creds.lock"
TOKEN_REFRESH_BUFFER_MS = 30 * 1000
LOCK_TIMEOUT_MS = 10000
CACHE_CHECK_INTERVAL_MS = 1000


def isErrorResponse(
    response: Union[Dict, ErrorDataDict]
) -> bool:
    return "error" in response


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
    parent = "QwenCode"
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
        path = Path(os.path.expanduser(f"~/{QWEN_DIR}/{QWEN_CREDENTIAL_FILENAME}"))
        if path.is_file():
            return path
        return SharedTokenManager.get_cache_file()

    def getLockFilePath(self):
        return Path(os.path.expanduser(f"~/{QWEN_DIR}/{QWEN_LOCK_FILENAME}"))

    def setLockConfig(self, config: dict):
        # Optional: allow lock config override
        pass

    def registerCleanupHandlers(self):
        import atexit

        def cleanup():
            try:
                lock_path = self.getLockFilePath()
                lock_path.unlink()
            except:
                pass

        atexit.register(cleanup)

    async def getValidCredentials(self, qwen_client: IQwenOAuth2Client, force_refresh: bool = False):
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

            self.refresh_promise = asyncio.create_task(self.performTokenRefresh(qwen_client, force_refresh))
            credentials = await self.refresh_promise
            self.refresh_promise = None
            return credentials
        except Exception as e:
            if isinstance(e, TokenManagerError):
                raise
            raise TokenManagerError(TokenError.REFRESH_FAILED, str(e), e) from e

    def checkAndReloadIfNeeded(self):
        now = int(time.time() * 1000)
        if now - self.memory_cache["last_check"] < CACHE_CHECK_INTERVAL_MS:
            return
        self.memory_cache["last_check"] = now

        try:
            file_path = self.getCredentialFilePath()
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
        for field in ["access_token", "refresh_token", "token_type"]:
            if field not in data or not isinstance(data[field], str):
                raise ValueError(f"Invalid credentials: missing {field}")
        if "expiry_date" not in data or not isinstance(data["expiry_date"], (int, float)):
            raise ValueError("Invalid credentials: missing expiry_date")
        return data

    async def performTokenRefresh(self, qwen_client: IQwenOAuth2Client, force_refresh: bool):
        lock_path = self.getLockFilePath()
        try:
            if self.memory_cache["credentials"] is None:
                self.reloadCredentialsFromFile()
            qwen_client.setCredentials(self.memory_cache["credentials"])
            current_credentials = qwen_client.getCredentials()
            if not current_credentials.get("refresh_token"):
                raise TokenManagerError(TokenError.NO_REFRESH_TOKEN, "No refresh token")
            await self.acquireLock(lock_path)

            self.checkAndReloadIfNeeded()

            if (
                not force_refresh
                and self.memory_cache["credentials"]
                and self.isTokenValid(self.memory_cache["credentials"])
            ):
                qwen_client.setCredentials(self.memory_cache["credentials"])
                return self.memory_cache["credentials"]

            response = await qwen_client.refreshAccessToken()
            if not response or isErrorResponse(response):
                raise TokenManagerError(TokenError.REFRESH_FAILED, str(response))
            token_data = response
            if "access_token" not in token_data:
                raise TokenManagerError(TokenError.REFRESH_FAILED, "No access_token returned")

            credentials = {
                "access_token": token_data["access_token"],
                "token_type": token_data["token_type"],
                "refresh_token": token_data.get("refresh_token", current_credentials.get("refresh_token")),
                "resource_url": token_data.get("resource_url"),
                "expiry_date": int(time.time() * 1000) + token_data.get("expires_in", 0) * 1000,
            }
            self.memory_cache["credentials"] = credentials
            qwen_client.setCredentials(credentials)

            await self.saveCredentialsToFile(credentials)
            return credentials
        except Exception as e:
            if isinstance(e, TokenManagerError):
                raise
            raise

        finally:
            await self.releaseLock(lock_path)

    async def acquireLock(self, lock_path: Path):
        max_attempts = 50
        attempt_interval = 200  # ms
        lock_id = str(uuid.uuid4())
        os.makedirs(lock_path.parent, exist_ok=True)

        for _ in range(max_attempts):
            try:
                with open(lock_path, "w") as f:
                    f.write(lock_id)
                return
            except:
                try:
                    stat = os.stat(str(lock_path))
                    lock_age = int(time.time() * 1000) - int(stat.st_mtime * 1000)
                    if lock_age > LOCK_TIMEOUT_MS:
                        try:
                            await os.unlink(str(lock_path))
                        except:
                            pass
                except:
                    pass
                await asyncio.sleep(attempt_interval / 1000)
        raise TokenManagerError(TokenError.LOCK_TIMEOUT, "Failed to acquire lock")

    async def releaseLock(self, lock_path: Path):
        try:
            await os.unlink(str(lock_path))
        except:
            pass

    async def saveCredentialsToFile(self, credentials: dict):
        file_path = self.getCredentialFilePath()
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps(credentials, indent=2))
        stat = os.stat(str(file_path))
        self.memory_cache["file_mod_time"] = int(stat.st_mtime * 1000)

    def isTokenValid(self, credentials: dict) -> bool:
        expiry_date = credentials.get("expiry_date")
        if not expiry_date:
            return False
        return time.time() * 1000 < expiry_date - TOKEN_REFRESH_BUFFER_MS

    def getCurrentCredentials(self):
        return self.memory_cache["credentials"]
