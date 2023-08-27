import json, uuid, requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class H2o(BaseProvider):
    url             = "https://gpt-gm.h2o.ai"
    working         = True
    supports_stream = True
    model           = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1"

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        conversation = ""
        for message in messages:
            conversation += "%s: %s\n" % (message["role"], message["content"])
        conversation += "assistant: "

        session = requests.Session()

        headers = {"Referer": "https://gpt-gm.h2o.ai/r/jGfKSwU"}
        data = {
            "ethicsModalAccepted"               : "true",
            "shareConversationsWithModelAuthors": "true",
            "ethicsModalAcceptedAt"             : "",
            "activeModel"                       : model,
            "searchEnabled"                     : "true",
        }
        
        session.post("https://gpt-gm.h2o.ai/settings",
                     headers=headers, data=data)

        headers = {"Referer": "https://gpt-gm.h2o.ai/"}
        data    = {"model": model}

        response = session.post("https://gpt-gm.h2o.ai/conversation",
                                headers=headers, json=data).json()
        
        if "conversationId" not in response:
            return

        data = {
            "inputs": conversation,
            "parameters": {
                "temperature"   : kwargs.get("temperature", 0.4),
                "truncate"          : kwargs.get("truncate", 2048),
                "max_new_tokens"    : kwargs.get("max_new_tokens", 1024),
                "do_sample"         : kwargs.get("do_sample", True),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
                "return_full_text"  : kwargs.get("return_full_text", False),
            },
            "stream" : True,
            "options": {
                "id"           : kwargs.get("id", str(uuid.uuid4())),
                "response_id"  : kwargs.get("response_id", str(uuid.uuid4())),
                "is_retry"     : False,
                "use_cache"    : False,
                "web_search_id": "",
            },
        }

        response = session.post(f"https://gpt-gm.h2o.ai/conversation/{response['conversationId']}",
            headers=headers, json=data)
        
        response.raise_for_status()
        response.encoding = "utf-8"
        generated_text    = response.text.replace("\n", "").split("data:")
        generated_text    = json.loads(generated_text[-1])

        yield generated_text["generated_text"]

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("truncate", "int"),
            ("max_new_tokens", "int"),
            ("do_sample", "bool"),
            ("repetition_penalty", "float"),
            ("return_full_text", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
