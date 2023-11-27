from __future__ import annotations

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

import json
import cloudscraper

class PI(AsyncGeneratorProvider):
    url                   = "https://chat-gpt.com"
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        Conversation = kwargs['conversation']
        UserPrompt = messages[-1]
        if UserPrompt['role'] == 'user':
            UserPrompt = UserPrompt['content']
        else:
            UserPrompt = messages[-2]['content']
        if Conversation == None:
            Conversation = PI.Start_Conversation()
            print(Conversation)
        Answer = Ask_PI(UserPrompt,Conversation['sid'],Conversation['cookies'])

        yield Answer[0]['text']

    def Start_Conversation():
        response = scraper.post('https://pi.ai/api/chat/start', data="{}",headers={'x-api-version': '3'})
        cookies = response.cookies

        return {
            'sid': response.json()['conversations'][0]['sid'],
            'cookies': cookies
        }
        
    def GetConversationTitle(Conversation):
        response = scraper.post('https://pi.ai/api/chat/start', data="{}",headers={'x-api-version': '3'}, cookies=Conversation['cookies'])
        
        return {
            'title': response.json()['conversations'][0]['title']
        }
        
    def GetChatHistory(Conversation):
        params = {
            'conversation': Conversation['sid'],
        }
        response = scraper.get('https://pi.ai/api/chat/history', params=params, cookies=Conversation['cookies'])

        return response.json()

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def Ask_PI(message,sid,cookies):
    json_data = {
        'text': message,
        'conversation': sid,
        'mode': 'BASE',
    }
    response = scraper.post('https://pi.ai/api/chat', json=json_data, cookies=cookies)
    
    result = []
    for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
        if line.startswith('data: {"text":'):
            result.append(json.loads(line.split('data: ')[1].encode('utf-8')))
        if line.startswith('data: {"title":'):
            result.append(json.loads(line.split('data: ')[1].encode('utf-8')))
    return result
