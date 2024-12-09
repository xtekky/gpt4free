from __future__ import annotations

import json
import aiohttp
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    BeautifulSoup = None
    
from aiohttp import ClientTimeout
from ...errors import MissingRequirementsError
from ...typing import AsyncResult, Messages
from ...cookies import get_cookies_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt

from ... import debug


class RobocodersAPI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "API Robocoders AI"
    url = "https://api.robocoders.ai/docs"
    api_endpoint = "https://api.robocoders.ai/chat"
    working = False
    supports_message_history = True
    default_model = 'GeneralCodingAgent'
    agent = [default_model, "RepoAgent", "FrontEndAgent"]
    models = [*agent]

    CACHE_DIR = Path(get_cookies_dir())
    CACHE_FILE = CACHE_DIR / "robocoders.json"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        
        timeout = ClientTimeout(total=600)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Load or create access token and session ID
            access_token, session_id = await cls._get_or_create_access_and_session(session)
            if not access_token or not session_id:
                raise Exception("Failed to initialize API interaction")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            prompt = format_prompt(messages)
            
            data = {
                "sid": session_id,
                "prompt": prompt,
                "agent": model
            }
            
            async with session.post(cls.api_endpoint, headers=headers, json=data, proxy=proxy) as response:
                if response.status == 401:  # Unauthorized, refresh token
                    cls._clear_cached_data()
                    raise Exception("Unauthorized: Invalid token, please retry.")
                elif response.status == 422:
                    raise Exception("Validation Error: Invalid input.")
                elif response.status >= 500:
                    raise Exception(f"Server Error: {response.status}")
                elif response.status != 200:
                    raise Exception(f"Unexpected Error: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            # Decode bytes into a string
                            line_str = line.decode('utf-8')
                            response_data = json.loads(line_str)
                            
                            # Get the message from the 'args.content' or 'message' field
                            message = (response_data.get('args', {}).get('content') or 
                                     response_data.get('message', ''))
                            
                            if message:
                                yield message
                                
                            # Check for reaching the resource limit
                            if (response_data.get('action') == 'message' and 
                                response_data.get('args', {}).get('wait_for_response')):
                                # Automatically continue the dialog
                                continue_data = {
                                    "sid": session_id,
                                    "prompt": "continue",
                                    "agent": model
                                }
                                async with session.post(
                                    cls.api_endpoint, 
                                    headers=headers, 
                                    json=continue_data, 
                                    proxy=proxy
                                ) as continue_response:
                                    if continue_response.status == 200:
                                        async for continue_line in continue_response.content:
                                            if continue_line:
                                                try:
                                                    continue_line_str = continue_line.decode('utf-8')
                                                    continue_data = json.loads(continue_line_str)
                                                    continue_message = (
                                                        continue_data.get('args', {}).get('content') or 
                                                        continue_data.get('message', '')
                                                    )
                                                    if continue_message:
                                                        yield continue_message
                                                except json.JSONDecodeError:
                                                    debug.log(f"Failed to decode continue JSON: {continue_line}")
                                                except Exception as e:
                                                    debug.log(f"Error processing continue response: {e}")
                                
                        except json.JSONDecodeError:
                            debug.log(f"Failed to decode JSON: {line}")
                        except Exception as e:
                            debug.log(f"Error processing response: {e}")

    @staticmethod
    async def _get_or_create_access_and_session(session: aiohttp.ClientSession):
        RobocodersAPI.CACHE_DIR.mkdir(exist_ok=True)  # Ensure cache directory exists

        # Load data from cache
        if RobocodersAPI.CACHE_FILE.exists():
            with open(RobocodersAPI.CACHE_FILE, "r") as f:
                data = json.load(f)
                access_token = data.get("access_token")
                session_id = data.get("sid")

                # Validate loaded data
                if access_token and session_id:
                    return access_token, session_id

        # If data not valid, create new access token and session ID
        access_token = await RobocodersAPI._fetch_and_cache_access_token(session)
        session_id = await RobocodersAPI._create_and_cache_session(session, access_token)
        return access_token, session_id

    @staticmethod
    async def _fetch_and_cache_access_token(session: aiohttp.ClientSession) -> str:
        if not HAS_BEAUTIFULSOUP:
            raise MissingRequirementsError('Install "beautifulsoup4" package | pip install -U beautifulsoup4')
            return token

        url_auth = 'https://api.robocoders.ai/auth'
        headers_auth = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        async with session.get(url_auth, headers=headers_auth) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                token_element = soup.find('pre', id='token')
                if token_element:
                    token = token_element.text.strip()

                    # Cache the token
                    RobocodersAPI._save_cached_data({"access_token": token})
                    return token
        return None

    @staticmethod
    async def _create_and_cache_session(session: aiohttp.ClientSession, access_token: str) -> str:
        url_create_session = 'https://api.robocoders.ai/create-session'
        headers_create_session = {
            'Authorization': f'Bearer {access_token}'
        }

        async with session.get(url_create_session, headers=headers_create_session) as response:
            if response.status == 200:
                data = await response.json()
                session_id = data.get('sid')

                # Cache session ID
                RobocodersAPI._update_cached_data({"sid": session_id})
                return session_id
            elif response.status == 401:
                RobocodersAPI._clear_cached_data()
                raise Exception("Unauthorized: Invalid token during session creation.")
            elif response.status == 422:
                raise Exception("Validation Error: Check input parameters.")
        return None

    @staticmethod
    def _save_cached_data(new_data: dict):
        """Save new data to cache file"""
        RobocodersAPI.CACHE_DIR.mkdir(exist_ok=True)
        RobocodersAPI.CACHE_FILE.touch(exist_ok=True)
        with open(RobocodersAPI.CACHE_FILE, "w") as f:
            json.dump(new_data, f)

    @staticmethod
    def _update_cached_data(updated_data: dict):
        """Update existing cache data with new values"""
        data = {}
        if RobocodersAPI.CACHE_FILE.exists():
            with open(RobocodersAPI.CACHE_FILE, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If cache file is corrupted, start with empty dict
                    data = {}
        
        data.update(updated_data)
        with open(RobocodersAPI.CACHE_FILE, "w") as f:
            json.dump(data, f)

    @staticmethod
    def _clear_cached_data():
        """Remove cache file"""
        try:
            if RobocodersAPI.CACHE_FILE.exists():
                RobocodersAPI.CACHE_FILE.unlink()
        except Exception as e:
            debug.log(f"Error clearing cache: {e}")

    @staticmethod
    def _get_cached_data() -> dict:
        """Get all cached data"""
        if RobocodersAPI.CACHE_FILE.exists():
            try:
                with open(RobocodersAPI.CACHE_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
