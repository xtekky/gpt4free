import json
import re
from fake_useragent import UserAgent

import requests

class Completion:
    @staticmethod
    def create(
        systemprompt:str,
        text:str,
        assistantprompt:str
    ):

        data = [
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": assistantprompt},
            {"role": "user", "content": text},
        ]
        url = f'https://openai.a2hosted.com/chat?q={Completion.__get_query_param(data)}'

        try:
            response = requests.get(url, headers=Completion.__get_headers(), stream=True)
        except:
            return Completion.__get_failure_response()

        sentence = ""

        for message in response.iter_content(chunk_size=1024):
            message = message.decode('utf-8')
            msg_match, num_match = re.search(r'"msg":"([^"]+)"', message), re.search(r'\[DONE\] (\d+)', message)
            if msg_match:
                # Put the captured group into a sentence
                sentence += msg_match.group(1)
        return {
            'response': sentence
        }
    
    @classmethod
    def __get_headers(cls) -> dict:
        return {
            'authority': 'openai.a2hosted.com',
            'accept': 'text/event-stream',
            'accept-language': 'en-US,en;q=0.9,id;q=0.8,ja;q=0.7',
            'cache-control': 'no-cache',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': UserAgent().random
        }
    
    @classmethod
    def __get_failure_response(cls) -> dict:
        return dict(response='Unable to fetch the response, Please try again.', links=[], extra={})
    
    @classmethod
    def __get_query_param(cls, conversation) -> str:
        encoded_conversation = json.dumps(conversation)
        return encoded_conversation.replace(" ", "%20").replace('"', '%22').replace("'", "%27")