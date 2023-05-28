import fake_useragent
import requests
import json
from .typings import AiColorsResponse


class Completion:
    @staticmethod
    def create(
        query: str = "",
    ) -> AiColorsResponse:
        headers = {
            "authority": "jsuifmbqefnxytqwmaoy.functions.supabase.co",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.5",
            "cache-control": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": fake_useragent.UserAgent().random,
        }

        json_data = {"query": query}

        url = "https://jsuifmbqefnxytqwmaoy.functions.supabase.co/chatgpt"
        request = requests.post(url, headers=headers, json=json_data, timeout=30)
        data = request.json().get("text").get("content")
        json_data = json.loads(data.replace("\n  ", ""))

        return AiColorsResponse(**json_data)
