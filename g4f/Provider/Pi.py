from __future__ import annotations

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider, format_prompt

import json
from cloudscraper import CloudScraper, session, create_scraper

class Pi(BaseProvider):
    url             = "https://chat-gpt.com"
    working         = True
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        scraper: CloudScraper = None,
        conversation: dict = None,
        **kwargs
    ) -> CreateResult:
        if not scraper:
            scraper = cls.get_scraper(proxy)
        if not conversation:
            conversation = cls.start_conversation(scraper)
        answer = cls.ask(scraper, messages, conversation)
        for line in answer:
            if "text" in line:
                yield line["text"]
                        
    def get_scraper(proxy: str):
        return create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            },
            headers={
                'Accept': '*/*',
                'Accept-Encoding': 'deflate,gzip,br',
            },
            proxies={
                "https": proxy
            }
        )
        
    def start_conversation(scraper: CloudScraper):
        response = scraper.post('https://pi.ai/api/chat/start', data="{}", headers={
            'accept': 'application/json',
            'x-api-version': '3'
        })
        if 'Just a moment' in response.text:
            raise RuntimeError('Error: Cloudflare detected')
        return Conversation(
            response.json()['conversations'][0]['sid'],
            response.cookies
        )
        
    def get_chat_history(scraper: CloudScraper, conversation: Conversation):
        params = {
            'conversation': conversation.sid,
        }
        response = scraper.get('https://pi.ai/api/chat/history', params=params, cookies=conversation.cookies)
        if 'Just a moment' in response.text:
            raise RuntimeError('Error: Cloudflare detected')
        return response.json()

    def ask(scraper: CloudScraper, messages: Messages, conversation: Conversation):
        json_data = {
            'text': format_prompt(messages),
            'conversation': conversation.sid,
            'mode': 'BASE',
        }
        response = scraper.post('https://pi.ai/api/chat', json=json_data, cookies=conversation.cookies, stream=True)
        
        for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
            if 'Just a moment' in line:
                raise RuntimeError('Error: Cloudflare detected')
            if line.startswith('data: {"text":'):
               yield json.loads(line.split('data: ')[1])
            if line.startswith('data: {"title":'):
               yield json.loads(line.split('data: ')[1])
                
class Conversation():
    def __init__(self, sid: str, cookies):
        self.sid = sid
        self.cookies = cookies
        