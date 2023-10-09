from __future__ import annotations

import json
import random
import requests

from ...typing import Any, CreateResult, Messages
from ..base_provider import BaseProvider
from ..helper import format_prompt


class Theb(BaseProvider):
    url                     = "https://theb.ai"
    working                 = True
    supports_stream         = True
    supports_gpt_35_turbo   = True
    needs_auth              = True

    @staticmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        **kwargs
    ) -> CreateResult:
        auth = kwargs.get("auth", {
            "bearer_token":"free",
            "org_id":"theb",
        })
        
        bearer_token = auth["bearer_token"]
        org_id       = auth["org_id"]
        
        headers = {
            'authority'         : 'beta.theb.ai',
            'accept'            : 'text/event-stream',
            'accept-language'   : 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
            'authorization'     : 'Bearer '+bearer_token,
            'content-type'      : 'application/json',
            'origin'            : 'https://beta.theb.ai',
            'referer'           : 'https://beta.theb.ai/home',
            'sec-ch-ua'         : '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
            'sec-ch-ua-mobile'  : '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest'    : 'empty',
            'sec-fetch-mode'    : 'cors',
            'sec-fetch-site'    : 'same-origin',
            'user-agent'        : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
            'x-ai-model'        : 'ee8d4f29cb7047f78cbe84313ed6ace8',
        }
        
        req_rand = random.randint(100000000, 9999999999)

        json_data: dict[str, Any] = {
            "text"      : format_prompt(messages),
            "category"  : "04f58f64a4aa4191a957b47290fee864",
            "model"     : "ee8d4f29cb7047f78cbe84313ed6ace8",
            "model_params": {
                "system_prompt"     : "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: {{YYYY-MM-DD}}",
                "temperature"       : kwargs.get("temperature", 1),
                "top_p"             : kwargs.get("top_p", 1),
                "frequency_penalty" : kwargs.get("frequency_penalty", 0),
                "presence_penalty"  : kwargs.get("presence_penalty", 0),
                "long_term_memory"  : "auto"
            }
        }
        
        response = requests.post(
            f"https://beta.theb.ai/api/conversation?org_id={org_id}&req_rand={req_rand}",
            headers=headers,
            json=json_data,
            stream=True,
            proxies={"https": proxy}
        )
        
        response.raise_for_status()
        content = ""
        next_content = ""
        for chunk in response.iter_lines():
            if b"content" in chunk:
                next_content = content
                data = json.loads(chunk.decode().split("data: ")[1])
                content = data["content"]
                yield data["content"].replace(next_content, "")

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("auth", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("presence_penalty", "int"),
            ("frequency_penalty", "int"),
            ("top_p", "int")
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"