from re import findall
from json import loads
from queue import Queue, Empty
from threading import Thread
from curl_cffi import requests

class Completion:
    # experimental
    part1 = '{"role":"assistant","id":"chatcmpl'
    part2 = '"},"index":0,"finish_reason":null}]}}'
    regex = rf'{part1}(.*){part2}'

    timer = None
    message_queue = Queue()
    stream_completed = False

    def request(prompt: str):
        headers = {
            'authority': 'chatbot.theb.ai',
            'content-type': 'application/json',
            'origin': 'https://chatbot.theb.ai',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
        }

        requests.post('https://chatbot.theb.ai/api/chat-process', headers=headers,
            content_callback = Completion.handle_stream_response,
            json = {
                'prompt': prompt,
                'options': {}
            }
        )

        Completion.stream_completed = True

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
