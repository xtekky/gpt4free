from json import loads
from queue import Queue, Empty
from re import findall
from threading import Thread
from typing import Generator, Optional

from curl_cffi import requests
from fake_useragent import UserAgent


class Completion:
    # experimental
    part1 = '{"role":"assistant","id":"chatcmpl'
    part2 = '"},"index":0,"finish_reason":null}]}}'
    regex = rf'{part1}(.*){part2}'

    timer = None
    message_queue = Queue()
    stream_completed = False

    @staticmethod
    def request(prompt: str, proxy: Optional[str]=None):
        headers = {
            'authority': 'chatbot.theb.ai',
            'content-type': 'application/json',
            'origin': 'https://chatbot.theb.ai',
            'user-agent': UserAgent().random,
        }

        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else None

        requests.post(
            'https://chatbot.theb.ai/api/chat-process',
            headers=headers,
            proxies=proxies,
            content_callback=Completion.handle_stream_response,
            json={'prompt': prompt, 'options': {}},
        )

        Completion.stream_completed = True

    @staticmethod
    def create(prompt: str, proxy: Optional[str]=None) -> Generator[str, None, None]:
        Thread(target=Completion.request, args=[prompt, proxy]).start()

        while not Completion.stream_completed or not Completion.message_queue.empty():
            try:
                message = Completion.message_queue.get(timeout=0.01)
                for message in findall(Completion.regex, message):
                    yield loads(Completion.part1 + message + Completion.part2)['delta']

            except Empty:
                pass

    @staticmethod
    def handle_stream_response(response):
        Completion.message_queue.put(response.decode())
