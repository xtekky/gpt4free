import urllib.request
import json


class Completion:
    @staticmethod
    def create(
        systemMessage: str = "You are a helpful assistant",
        prompt: str = "",
        parentMessageId: str = "",
        temperature: float = 0.8,
        top_p: float = 1,
    ):
        json_data = {
            "prompt": prompt,
            "options": {"parentMessageId": parentMessageId},
            "systemMessage": systemMessage,
            "temperature": temperature,
            "top_p": top_p,
        }

        url = "http://43.153.7.56:8080/api/chat-process"
        headers = {"Content-type": "application/json"}

        data = json.dumps(json_data).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(req)
        content = response.read().decode()

        return Completion.__load_json(content)

    @classmethod
    def __load_json(cls, content) -> dict:
        split = content.rsplit("\n", 1)[1]
        to_json = json.loads(split)
        return to_json
