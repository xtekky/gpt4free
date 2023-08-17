import json
import random
import re

import browser_cookie3
import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class Bard(BaseProvider):
    url = "https://bard.google.com"
    needs_auth = True
    working = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        psid = {
            cookie.name: cookie.value
            for cookie in browser_cookie3.chrome(domain_name=".google.com")
        }["__Secure-1PSID"]

        formatted = "\n".join(
            ["%s: %s" % (message["role"], message["content"]) for message in messages]
        )
        prompt = f"{formatted}\nAssistant:"

        proxy = kwargs.get("proxy", False)
        if proxy == False:
            print(
                "warning!, you did not give a proxy, a lot of countries are banned from Google Bard, so it may not work"
            )

        snlm0e = None
        conversation_id = None
        response_id = None
        choice_id = None

        client = requests.Session()
        client.proxies = (
            {"http": f"http://{proxy}", "https": f"http://{proxy}"} if proxy else {}
        )

        client.headers = {
            "authority": "bard.google.com",
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "origin": "https://bard.google.com",
            "referer": "https://bard.google.com/",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
            "x-same-domain": "1",
            "cookie": f"__Secure-1PSID={psid}",
        }

        if snlm0e is not None:
            result = re.search(
                r"SNlM0e\":\"(.*?)\"", client.get("https://bard.google.com/").text
            )
            if result is not None:
                snlm0e = result.group(1)

        params = {
            "bl": "boq_assistant-bard-web-server_20230326.21_p0",
            "_reqid": random.randint(1111, 9999),
            "rt": "c",
        }

        data = {
            "at": snlm0e,
            "f.req": json.dumps(
                [
                    None,
                    json.dumps(
                        [[prompt], None, [conversation_id, response_id, choice_id]]
                    ),
                ]
            ),
        }

        intents = ".".join(["assistant", "lamda", "BardFrontendService"])

        response = client.post(
            f"https://bard.google.com/_/BardChatUi/data/{intents}/StreamGenerate",
            data=data,
            params=params,
        )
        response.raise_for_status()

        chat_data = json.loads(response.content.splitlines()[3])[0][2]
        if chat_data:
            json_chat_data = json.loads(chat_data)

            yield json_chat_data[0][0]

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
