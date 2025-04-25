import aiohttp
import aiofiles
import json
import time
import os

from ..typing import AsyncResult, Messages # Make sure typing is imported correctly
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

# --- Async Auth Helpers ---

AUTH_FILE = 'auth_token.json'
DEBUG = False

async def read_auth_token():
    """Reads the first working auth token from the JSON file."""
    if not os.path.exists(AUTH_FILE):
        if DEBUG: print(f"Auth file {AUTH_FILE} does not exist.")
        return None
    try:
        async with aiofiles.open(AUTH_FILE, 'r') as token_file:
            content = await token_file.read()
            if not content:
                return None
            data = json.loads(content)
            # Check structure carefully
            tokens = data.get("auth", {}).get("tokens", [])
            if not tokens:
                return None
            for token_entry in tokens:
                if token_entry.get("working"):
                    return token_entry.get("token")
            return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        if DEBUG: print(f"Error reading auth token: {e}")
        return None

async def set_auth_token_working(token: str, working: bool):
    """Sets the working status of a specific token in the JSON file."""
    data = {"auth": {"tokens": []}}
    if os.path.exists(AUTH_FILE):
        try:
            async with aiofiles.open(AUTH_FILE, 'r') as token_file:
                content = await token_file.read()
                if content:
                   loaded_data = json.loads(content)
                   # Ensure the structure exists
                   if "auth" in loaded_data and "tokens" in loaded_data["auth"]:
                       data = loaded_data
                   elif "auth" not in loaded_data:
                        loaded_data["auth"] = {"tokens": []}
                        data = loaded_data
                   elif "tokens" not in loaded_data["auth"]:
                       loaded_data["auth"]["tokens"] = []
                       data = loaded_data

        except (json.JSONDecodeError, FileNotFoundError) as e:
             if DEBUG: print(f"Error reading auth file for update: {e}")
             # If file is corrupted or empty, we start fresh with the default structure

    token_found = False
    if "tokens" not in data["auth"]: # Ensure tokens list exists
        data["auth"]["tokens"] = []

    for token_entry in data["auth"]["tokens"]:
        if token_entry.get("token") == token:
            token_entry["working"] = working
            token_found = True
            break

    if not token_found:
        # Only add if it's a new token we are marking (usually as working=True)
        if working:
             data["auth"]["tokens"].append({"token": token, "working": working})
        # Don't add if we are trying to mark a non-existent token as not working

    try:
        async with aiofiles.open(AUTH_FILE, 'w') as token_file:
            await token_file.write(json.dumps(data, indent=2))
    except IOError as e:
        if DEBUG: print(f"Error writing auth token file: {e}")


async def signup_user(session: aiohttp.ClientSession, proxy: str | None = None) -> str | None:
    """Signs up a temporary user and saves the token."""
    url_signup = "https://api.puter.com/signup" 
    headers_signup = {
        "Content-Type": "application/json",
        "host": "api.puter.com",
        "connection": "keep-alive",
        "sec-ch-ua-platform": '"macOS"', 
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-mobile": "?0",
        "accept": "*/*",
        "origin": "https://puter.com",
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://puter.com/",
        "accept-encoding": "gzip",
        "accept-language": "en-US,en;q=0.9"
    }
    body_signup = {
        "is_temp": True
    }
    try:
        async with session.post(url_signup, headers=headers_signup, json=body_signup, proxy=proxy) as response:
            if DEBUG: print(f"Signup Status: {response.status}")
            if DEBUG: print(f"Signup Headers: {response.headers}")
            response_text = await response.text()
            if DEBUG: print(f"Signup Body: {response_text}")

            response.raise_for_status() 
            data = await response.json() 
            auth_token = data.get("token")

            if auth_token:
                if DEBUG: print(f"Extracted auth token: {auth_token}")
                await set_auth_token_working(auth_token, True)
                return auth_token
            else:
                if DEBUG: print("Signup successful but no token found in response.")
                return None
    except aiohttp.ClientError as e:
        if DEBUG: print(f"Error during signup: {e}")
        return None
    except json.JSONDecodeError as e:
        if DEBUG: print(f"Error decoding signup JSON response: {e}, Response Text: {response_text}")
        return None

async def get_user_app_token(session: aiohttp.ClientSession, auth_token: str, proxy: str | None = None) -> str | None:
    """Gets the short-lived user app token."""
    url_get_token = "https://api.puter.com/auth/get-user-app-token"
    headers_get_token = {
        "host": "api.puter.com",
        "connection": "keep-alive",
        "authorization": f"Bearer {auth_token}",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "accept": "*/*",
        "origin": "https://puter.com", 
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://puter.com/",
        "accept-encoding": "gzip",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json"
    }
    body_auth = {
        "origin": "https://puter.com"
    }
    try:
        async with session.post(url_get_token, headers=headers_get_token, json=body_auth, proxy=proxy) as response:
            if DEBUG: print(f"Get App Token Status: {response.status}")
            if DEBUG: print(f"Get App Token Headers: {response.headers}")
            response_text = await response.text()
            if DEBUG: print(f"Get App Token Body: {response_text}")

            response.raise_for_status()
            data = await response.json()
            new_token = data.get("token")
            if new_token:
                 if DEBUG: print(f"Extracted new token: {new_token}")
                 return new_token
            else:
                 if DEBUG: print("Get App Token successful but no token found in response.")
                 return None
    except aiohttp.ClientError as e:
        if DEBUG: print(f"Error getting user app token: {e}")
        if isinstance(e, aiohttp.ClientResponseError) and e.status == 401:
             if DEBUG: print(f"Auth token {auth_token[:5]}... likely invalid (401). Marking non-working.")
             await set_auth_token_working(auth_token, False)
        return None
    except json.JSONDecodeError as e:
        if DEBUG: print(f"Error decoding Get App Token JSON response: {e}, Response Text: {response_text}")
        return None

# --- Provider Class ---

class Puter(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Puter"
    url = "https://puter.com"
    api_endpoint = "https://api.puter.com/drivers/call"
    working = True 
    needs_auth = True
    supports_stream = True
    supports_system_message = True 
    supports_message_history = True

    default_model = 'gpt-4o-mini'
    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
        "o1-pro",
        "o3",
        "o3-mini",
        "o4-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.5-preview",
        ]

    model_aliases = {
        "gpt4o-mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4-turbo": "gpt-4.1-nano",
        
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if not model:
            return cls.default_model
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            # raise ValueError(f"Unknown model: {model}")
            print(f"Warning: Unknown model '{model}' requested. Using default '{cls.default_model}'.")
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        stream: bool = True, 
        max_retries: int = 1,
        **kwargs # 
    ) -> AsyncResult:
        model_name = cls.get_model(model)
        attempt = 0

        async with aiohttp.ClientSession() as session:
            while attempt <= max_retries:
                auth_token = await read_auth_token()
                if not auth_token:
                    if DEBUG: print("No working auth token found. Signing up...")
                    auth_token = await signup_user(session, proxy=proxy)
                    if not auth_token:
                        raise RuntimeError("Failed to sign up for a new Puter user.") 

                if DEBUG: print(f"Using auth token: {auth_token[:5]}...")
                new_token = await get_user_app_token(session, auth_token, proxy=proxy)

                if not new_token:
                    if DEBUG: print(f"Failed to get user app token for auth token {auth_token[:5]}... Trying again (attempt {attempt + 1}/{max_retries + 1}).")
                    await set_auth_token_working(auth_token, False) # 
                    attempt += 1
                    continue # 
                if DEBUG: print(f"Using app token: {new_token[:5]}...")

                headers_call = {
                    "host": "api.puter.com",
                    "connection": "keep-alive",
                    "authorization": f"Bearer {new_token}",
                    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                    "content-type": "application/json;charset=UTF-8",
                    "accept": "text/event-stream" if stream else "application/json",
                    "origin": "https://puter.com",
                    "sec-fetch-site": "same-site", 
                    "sec-fetch-mode": "cors",
                    "sec-fetch-dest": "empty",
                    "referer": "https://puter.com/",
                    "accept-encoding": "gzip",
                    "accept-language": "en-US,en;q=0.9"
                }

                api_payload_args = {
                    "messages": messages,
                    "model": model_name,
                    "stream": stream,
                    **kwargs # 
                }
                api_payload = {
                    "interface": "puter-chat-completion",
                    "driver": "openai-completion",
                    "test_mode": False,
                    "method": "complete",
                    "args": api_payload_args
                }

                try:
                    async with session.post(
                        cls.api_endpoint,
                        headers=headers_call,
                        json=api_payload,
                        proxy=proxy
                    ) as response:
                        check_data = await response.json()
                        if DEBUG: print(f"Call Driver Status: {response.status}")
                        if DEBUG: print(f"Call Driver Headers: {response.headers}")
                        if check_data.get("success") is False and (check_data.get("error", {}).get("delegate") == "usage-limited-chat" or "error_400_from_delegate" in response.content):
                            if DEBUG: print(f"Rate limit hit for auth token {auth_token[:5]}... Marking non-working.")
                            await set_auth_token_working(auth_token, False)
                            attempt += 1
                            continue # Retry the loop
                        
                        # Check for rate limit error specifically
                        if response.status == 400:
                            try:
                                error_data = await response.json()
                                if DEBUG: print(f"Call Driver Error Body: {error_data}")
                            
                                # Different 400 error
                                response.raise_for_status() # Let raise_for_status handle it
                            except (json.JSONDecodeError, aiohttp.ClientPayloadError) as json_err:
                                # Handle cases where 400 response is not valid JSON
                                if DEBUG: print(f"Could not decode 400 error response: {json_err}")
                                response.raise_for_status() # Raise based on status code

                        response.raise_for_status() 
                        if stream:
                            if DEBUG: print("Processing stream...")
                            async for line_bytes in response.content:
                                line = line_bytes.decode('utf-8').strip()
                                if DEBUG: print(f"Stream line: {line}")
                                if line:
                                    try:
                                        data = json.loads(line)
                                        if data.get("type") == "text":
                                            text_chunk = data.get("text", "")
                                            if text_chunk:
                                                yield text_chunk
                                        elif data.get("type") == "result" and data.get("result",{}).get("finish_reason"):
                                             # Stream finished (optional: log finish reason)
                                             if DEBUG: print(f"Stream finished: {data['result']['finish_reason']}")
                                             break # End generation for this response
                                        elif data.get("type") == "error":
                                             if DEBUG: print(f"Stream error: {data.get('error')}")
                                             raise RuntimeError(f"Puter stream error: {data.get('error')}")

                                    except json.JSONDecodeError:
                                        if DEBUG: print(f"Ignoring non-JSON line: {line}")
                                        continue # Ignore non-JSON lines if any
                            return 

                        else: # Not streaming
                            if DEBUG: print("Processing non-stream response...")
                            response_data = await response.json()
                            if DEBUG: print(f"Non-stream response data: {response_data}")
                            content = response_data.get("result", {}).get("message", {}).get("content")
                            if content:
                                yield content # Yield the full content as a single chunk
                            else:
                                # Handle case where response is successful but content is missing
                                if DEBUG: print("Non-stream response successful but no content found.")
                                yield "" # Yield empty string or handle as error?
                            return # End generator after yielding the single chunk

                except aiohttp.ClientResponseError as e:
                    # Handle HTTP errors not caught earlier (e.g., 5xx)
                    if DEBUG: print(f"HTTP error during driver call: {e.status} {e.message}")
                    # Retrying the whole loop might be the best general approach for now.
                    if e.status in [401, 403]:
                         if DEBUG: print("App token potentially invalid/expired. Retrying.")
                         # No need to explicitly mark auth_token here, as get_user_app_token should have worked
                         attempt += 1
                         continue
                    else:
                        # Raise for other client errors
                         raise e
                except Exception as e:
                    # Catch other unexpected errors
                    if DEBUG: print(f"Unexpected error during Puter interaction: {e}")
                    raise e # Re-raise the exception

            # If loop finishes without returning/yielding (i.e., max_retries exceeded)
            raise RuntimeError(f"Failed to get completion from Puter after {max_retries + 1} attempts.")