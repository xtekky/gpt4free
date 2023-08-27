from aiohttp import ClientSession

from .base_provider import AsyncProvider, format_prompt


class Yqcloud(AsyncProvider):
    url = "https://chat9.yqcloud.top/"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async(
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs,
    ) -> str:
        async with ClientSession(
            headers=_create_header()
        ) as session:
            payload = _create_payload(messages)
            async with session.post("https://api.aichatos.cloud/api/generateStream", proxy=proxy, json=payload) as response:
                response.raise_for_status()
                return await response.text()


def _create_header():
    return {
        "accept"        : "application/json, text/plain, */*",
        "content-type"  : "application/json",
        "origin"        : "https://chat9.yqcloud.top",
    }


def _create_payload(messages: list[dict[str, str]]):
    return {
        "prompt": format_prompt(messages),
        "network": True,
        "system": "",
        "withoutContext": False,
        "stream": False,
        "userId": "#/chat/1693025544336"
    }
