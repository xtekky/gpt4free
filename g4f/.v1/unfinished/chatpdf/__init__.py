import requests
import json

from queue import Queue, Empty
from threading import Thread
from json import loads
from re import findall


class Completion:

    def request(prompt: str):
        '''TODO: some sort of authentication + upload PDF from URL or local file
                Then you should get the atoken and chat ID
                '''

        token = "your_token_here"
        chat_id = "your_chat_id_here"

        url = "https://chat-pr4yueoqha-ue.a.run.app/"

        payload = json.dumps({
            "v": 2,
            "chatSession": {
                "type": "join",
                "chatId": chat_id
            },
            "history": [
                {
                    "id": "VNsSyJIq_0",
						"author": "p_if2GPSfyN8hjDoA7unYe",
						"msg": "<start>",
						"time": 1682672009270
                },
                {
					"id": "Zk8DRUtx_6",
						"author": "uplaceholder",
						"msg": prompt,
						"time": 1682672181339
                }
            ]
        })

        # TODO: fix headers, use random user-agent, streaming response, etc
        headers = {
            'authority': 'chat-pr4yueoqha-ue.a.run.app',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'atoken': token,
            'content-type': 'application/json',
            'origin': 'https://www.chatpdf.com',
            'referer': 'https://www.chatpdf.com/',
            'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
        }

        response = requests.request(
            "POST", url, headers=headers, data=payload).text
        Completion.stream_completed = True
        return {'response': response}

    @staticmethod
    def create(prompt: str):
        Thread(target=Completion.request, args=[prompt]).start()

        while Completion.stream_completed != True or not Completion.message_queue.empty():
            try:
                message = Completion.message_queue.get(timeout=0.01)
                for message in findall(Completion.regex, message):
                    yield loads(Completion.part1 + message + Completion.part2)['delta']

            except Empty:
                pass

    @staticmethod
    def handle_stream_response(response):
        Completion.message_queue.put(response.decode())
