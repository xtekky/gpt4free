import re
import urllib.parse
from datetime import datetime, timezone, timedelta

from ...requests import StreamSession
from ...providers.response import JsonConversation
from ...cookies import get_cookies, Cookies

async def get_zerogpu_token(space: str, session: StreamSession, conversation: JsonConversation, cookies: Cookies = None):
    zerogpu_uuid = None if conversation is None else getattr(conversation, "zerogpu_uuid", None)
    zerogpu_token = "[object Object]"

    cookies = get_cookies("huggingface.co", raise_requirements_error=False) if cookies is None else cookies
    if zerogpu_uuid is None:
        async with session.get(f"https://huggingface.co/spaces/{space}", cookies=cookies) as response:
            match = re.search(r"&quot;token&quot;:&quot;([^&]+?)&quot;", await response.text())
            if match:
                zerogpu_token = match.group(1)
            match = re.search(r"&quot;sessionUuid&quot;:&quot;([^&]+?)&quot;", await response.text())
            if match:
                zerogpu_uuid = match.group(1)
    if cookies:
        # Get current UTC time + 10 minutes
        dt = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(timespec='milliseconds')
        encoded_dt = urllib.parse.quote(dt)
        async with session.get(f"https://huggingface.co/api/spaces/{space}/jwt?expiration={encoded_dt}&include_pro_status=true", cookies=cookies) as response:
            response_data = (await response.json())
            if "token" in response_data:
                zerogpu_token = response_data["token"]

    return zerogpu_uuid, zerogpu_token
