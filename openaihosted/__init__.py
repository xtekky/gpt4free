import json
import re
import requests


class Completion:
    @staticmethod
    def create(messages):
        headers = {
            "authority": "openai.a2hosted.com",
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9,id;q=0.8,ja;q=0.7",
            "cache-control": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/112.0",
        }

        query_param = Completion.__create_query_param(messages)
        url = f"https://openai.a2hosted.com/chat?q={query_param}"
        request = requests.get(url, headers=headers, stream=True)
        if request.status_code != 200:
            return Completion.__get_failure_response()

        content = request.content
        response = Completion.__join_response(content)

        return {"responses": response}

    @classmethod
    def __get_failure_response(cls) -> dict:
        return dict(
            response="Unable to fetch the response, Please try again.",
            links=[],
            extra={},
        )

    @classmethod
    def __create_query_param(cls, conversation) -> str:
        encoded_conversation = json.dumps(conversation)
        return (
            encoded_conversation.replace(" ", "%20")
            .replace('"', "%22")
            .replace("'", "%27")
        )

    @classmethod
    def __convert_escape_codes(cls, text) -> str:
        rep = {'\\"': '"', "\\n": "\n", "\\'": "'"}
        for original, replacement in rep.items():
            text = text.replace(original, replacement)
        return text

    @classmethod
    def __join_response(cls, data) -> str:
        data = data.decode("utf-8")
        find_ans = re.findall(r'(?<={"msg":)[^}]*', str(data))
        ans = [Completion.__convert_escape_codes(x[1:-1]) for x in find_ans]
        response = "".join(ans)
        return response
