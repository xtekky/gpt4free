from json import dumps, loads
from os import getenv
from random import randint
from re import search
from urllib.parse import urlencode

from bard.typings import BardResponse
from dotenv import load_dotenv
from requests import Session

load_dotenv()
token = getenv('1psid')
proxy = getenv('proxy')

temperatures = {
    0: "Generate text strictly following known patterns, with no creativity.",
    0.1: "Produce text adhering closely to established patterns, allowing minimal creativity.",
    0.2: "Create text with modest deviations from familiar patterns, injecting a slight creative touch.",
    0.3: "Craft text with a mild level of creativity, deviating somewhat from common patterns.",
    0.4: "Formulate text balancing creativity and recognizable patterns for coherent results.",
    0.5: "Generate text with a moderate level of creativity, allowing for a mix of familiarity and novelty.",
    0.6: "Compose text with an increased emphasis on creativity, while partially maintaining familiar patterns.",
    0.7: "Produce text favoring creativity over typical patterns for more original results.",
    0.8: "Create text heavily focused on creativity, with limited concern for familiar patterns.",
    0.9: "Craft text with a strong emphasis on unique and inventive ideas, largely ignoring established patterns.",
    1: "Generate text with maximum creativity, disregarding any constraints of known patterns or structures."
}


class Completion:
    def create(
            prompt: str = 'hello world',
            temperature: int = None,
            conversation_id: str = '',
            response_id: str = '',
            choice_id: str = '') -> BardResponse:

        if temperature:
            prompt = f'''settings: follow these settings for your response: [temperature: {temperature} - {temperatures[temperature]}] | prompt  : {prompt}'''

        client = Session()
        client.proxies = {
            'http': f'http://{proxy}',
            'https': f'http://{proxy}'} if proxy else None

        client.headers = {
            'authority': 'bard.google.com',
            'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
            'origin': 'https://bard.google.com',
            'referer': 'https://bard.google.com/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
            'x-same-domain': '1',
            'cookie': f'__Secure-1PSID={token}'
        }

        snlm0e = search(r'SNlM0e\":\"(.*?)\"',
                        client.get('https://bard.google.com/').text).group(1)

        params = urlencode({
            'bl': 'boq_assistant-bard-web-server_20230326.21_p0',
            '_reqid': randint(1111, 9999),
            'rt': 'c',
        })

        response = client.post(
            f'https://bard.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate?{params}',
            data={
                'at': snlm0e,
                'f.req': dumps([None, dumps([
                    [prompt],
                    None,
                    [conversation_id, response_id, choice_id],
                ])])
            }
            )

        chat_data = loads(response.content.splitlines()[3])[0][2]
        if not chat_data:
            print('error, retrying')
            Completion.create(prompt, temperature,
                              conversation_id, response_id, choice_id)

        json_chat_data = loads(chat_data)
        results = {
            'content': json_chat_data[0][0],
            'conversation_id': json_chat_data[1][0],
            'response_id': json_chat_data[1][1],
            'factualityQueries': json_chat_data[3],
            'textQuery': json_chat_data[2][0] if json_chat_data[2] is not None else '',
            'choices': [{'id': i[0], 'content': i[1]} for i in json_chat_data[4]],
        }

        return BardResponse(results)
