from __future__ import annotations

import json
import uuid
import random
import time
from typing import Dict, List

from aiohttp import ClientSession, BaseConnector

from ...typing import AsyncResult, Messages, Cookies
from ...requests import raise_for_status, DEFAULT_HEADERS
from ...image import ImageResponse, ImagePreview
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_connector, format_cookies

class Sources():
    def __init__(self, link_list: List[Dict[str, str]]) -> None:
        self.list = link_list

    def __str__(self) -> str:
        return "\n\n" + ("\n".join([f"[{link['title']}]({link['link']})" for link in self.list]))

class AbraGeoBlockedError(Exception):
    pass

class MetaAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Meta AI"
    url = "https://www.meta.ai"
    working = True
    default_model = ''

    def __init__(self, proxy: str = None, connector: BaseConnector = None):
        self.session = ClientSession(connector=get_connector(connector, proxy), headers=DEFAULT_HEADERS)
        self.cookies: Cookies = None
        self.access_token: str = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        async for chunk in cls(proxy).prompt(format_prompt(messages)):
            yield chunk

    async def update_access_token(self, birthday: str = "1999-01-01"):
        url = "https://www.meta.ai/api/graphql/"
        payload = {
            "lsd": self.lsd,
            "fb_api_caller_class": "RelayModern",
            "fb_api_req_friendly_name": "useAbraAcceptTOSForTempUserMutation",
            "variables": json.dumps({
                "dob": birthday,
                "icebreaker_type": "TEXT",
                "__relay_internal__pv__WebPixelRatiorelayprovider": 1,
            }),
            "doc_id": "7604648749596940",
        }
        headers = {
            "x-fb-friendly-name": "useAbraAcceptTOSForTempUserMutation",
            "x-fb-lsd": self.lsd,
            "x-asbd-id": "129477",
            "alt-used": "www.meta.ai",
            "sec-fetch-site": "same-origin"
        }
        async with self.session.post(url, headers=headers, cookies=self.cookies, data=payload) as response:
            await raise_for_status(response, "Fetch access_token failed")
            auth_json = await response.json(content_type=None)
            self.access_token = auth_json["data"]["xab_abra_accept_terms_of_service"]["new_temp_user_auth"]["access_token"]

    async def prompt(self, message: str, cookies: Cookies = None) -> AsyncResult:
        if self.cookies is None:
            await self.update_cookies(cookies)
        if cookies is not None:
            self.access_token = None
        if self.access_token is None and cookies is None:
            await self.update_access_token()
        if self.access_token is None:
            url = "https://www.meta.ai/api/graphql/"
            payload = {"lsd": self.lsd, 'fb_dtsg': self.dtsg}
            headers = {'x-fb-lsd': self.lsd}
        else:
            url = "https://graph.meta.ai/graphql?locale=user"
            payload = {"access_token": self.access_token}
            headers = {}
        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'cookie': format_cookies(self.cookies),
            'origin': 'https://www.meta.ai',
            'referer': 'https://www.meta.ai/',
            'x-asbd-id': '129477',
            'x-fb-friendly-name': 'useAbraSendMessageMutation',
            **headers
        }
        payload = {
            **payload,
            'fb_api_caller_class': 'RelayModern',
            'fb_api_req_friendly_name': 'useAbraSendMessageMutation',
            "variables": json.dumps({
                "message": {"sensitive_string_value": message},
                "externalConversationId": str(uuid.uuid4()),
                "offlineThreadingId": generate_offline_threading_id(),
                "suggestedPromptIndex": None,
                "flashVideoRecapInput": {"images": []},
                "flashPreviewInput": None,
                "promptPrefix": None,
                "entrypoint": "ABRA__CHAT__TEXT",
                "icebreaker_type": "TEXT",
                "__relay_internal__pv__AbraDebugDevOnlyrelayprovider": False,
                "__relay_internal__pv__WebPixelRatiorelayprovider": 1,
            }),
            'server_timestamps': 'true',
            'doc_id': '7783822248314888'
        }
        async with self.session.post(url, headers=headers, data=payload) as response:
            await raise_for_status(response, "Fetch response failed")
            last_snippet_len = 0
            fetch_id = None
            async for line in response.content:
                if b"<h1>Something Went Wrong</h1>" in line:
                    raise ResponseError("Response: Something Went Wrong")
                try:
                    json_line = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if json_line.get("errors"):
                    raise RuntimeError("\n".join([error.get("message") for error in json_line.get("errors")]))
                bot_response_message = json_line.get("data", {}).get("node", {}).get("bot_response_message", {})
                streaming_state = bot_response_message.get("streaming_state")
                fetch_id = bot_response_message.get("fetch_id") or fetch_id
                if streaming_state in ("STREAMING", "OVERALL_DONE"):
                    imagine_card = bot_response_message.get("imagine_card")
                    if imagine_card is not None:
                        imagine_session = imagine_card.get("session")
                        if imagine_session is not None:
                            imagine_medias = imagine_session.get("media_sets", {}).pop().get("imagine_media")
                            if imagine_medias is not None:
                                image_class = ImageResponse if streaming_state == "OVERALL_DONE" else ImagePreview
                                yield image_class([media["uri"] for media in imagine_medias], imagine_medias[0]["prompt"])
                    snippet =  bot_response_message["snippet"]
                    new_snippet_len = len(snippet)
                    if new_snippet_len > last_snippet_len:
                        yield snippet[last_snippet_len:]
                        last_snippet_len = new_snippet_len
            #if last_streamed_response is None:
            #    if attempts > 3:
            #        raise Exception("MetaAI is having issues and was not able to respond (Server Error)")
            #    access_token = await self.get_access_token()
            #    return await self.prompt(message=message, attempts=attempts + 1)
            if fetch_id is not None:
                sources = await self.fetch_sources(fetch_id)
                if sources is not None:
                    yield sources 

    async def update_cookies(self, cookies: Cookies = None):
        async with self.session.get("https://www.meta.ai/", cookies=cookies) as response:
            await raise_for_status(response, "Fetch home failed")
            text = await response.text()
            if "AbraGeoBlockedError" in text:
                raise AbraGeoBlockedError("Meta AI isn't available yet in your country")
            if cookies is None:
                cookies = {
                    "_js_datr": self.extract_value(text, "_js_datr"),
                    "abra_csrf": self.extract_value(text, "abra_csrf"),
                    "datr": self.extract_value(text, "datr"),
                }
            self.lsd = self.extract_value(text, start_str='"LSD",[],{"token":"', end_str='"}')
            self.dtsg = self.extract_value(text, start_str='"DTSGInitialData",[],{"token":"', end_str='"}')
            self.cookies = cookies

    async def fetch_sources(self, fetch_id: str) -> Sources:
        if self.access_token is None:
            url = "https://www.meta.ai/api/graphql/"
            payload = {"lsd": self.lsd, 'fb_dtsg': self.dtsg}
            headers = {'x-fb-lsd': self.lsd}
        else:
            url = "https://graph.meta.ai/graphql?locale=user"
            payload = {"access_token": self.access_token}
            headers = {}
        payload = {
            **payload,
            "fb_api_caller_class": "RelayModern",
            "fb_api_req_friendly_name": "AbraSearchPluginDialogQuery",
            "variables": json.dumps({"abraMessageFetchID": fetch_id}),
            "server_timestamps": "true",
            "doc_id": "6946734308765963",
        }
        headers = {
            "authority": "graph.meta.ai",
            "x-fb-friendly-name": "AbraSearchPluginDialogQuery",
            **headers
        }
        async with self.session.post(url, headers=headers, cookies=self.cookies, data=payload) as response:
            await raise_for_status(response, "Fetch sources failed")
            text = await response.text()
            if "<h1>Something Went Wrong</h1>" in text:
                raise ResponseError("Response: Something Went Wrong")
            try:
                response_json = json.loads(text)
                message = response_json["data"]["message"]
                if message is not None:
                    searchResults = message["searchResults"]
                    if searchResults is not None:
                        return Sources(searchResults["references"])
            except (KeyError, TypeError, json.JSONDecodeError):
                raise RuntimeError(f"Response: {text}")

    @staticmethod
    def extract_value(text: str, key: str = None, start_str = None, end_str = '",') -> str:
        if start_str is None:
            start_str = f'{key}":{{"value":"'
        start = text.find(start_str)
        if start >= 0:
            start+= len(start_str)
            end = text.find(end_str, start)
            if end >= 0:
                return text[start:end]

def generate_offline_threading_id() -> str:
    """
    Generates an offline threading ID.

    Returns:
        str: The generated offline threading ID.
    """
    # Generate a random 64-bit integer
    random_value = random.getrandbits(64)
    
    # Get the current timestamp in milliseconds
    timestamp = int(time.time() * 1000)
    
    # Combine timestamp and random value
    threading_id = (timestamp << 22) | (random_value & ((1 << 22) - 1))
    
    return str(threading_id)
