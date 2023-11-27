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
        Answer = Ask_PI(UserPrompt,Conversation['sid'],Conversation['cookies'])

        yield Answer[0]['text']

    def Start_Conversation():
        scraper.headers = {
            'accept-type': 'application/json'
        }
        response = scraper.post('https://pi.ai/api/chat/start', data="{}",headers={'x-api-version': '3'})
        cookies = response.cookies

        if 'Just a moment' in response.text:
            return {
                'error': 'cloudflare detected',
                'sid': None,
                'cookies': None,
            }
        return {
            'sid': response.json()['conversations'][0]['sid'],
            'cookies': cookies
        }
        
    def GetConversationTitle(Conversation):
        response = scraper.post('https://pi.ai/api/chat/start', data="{}",headers={'x-api-version': '3'}, cookies=Conversation['cookies'])
        if 'Just a moment' in response.text:
            return {
                'error': 'cloudflare detected',
                'title': 'Couldnt get the title',
            }
        return {
            'title': response.json()['conversations'][0]['title']
        }
        
    def GetChatHistory(Conversation):
        params = {
            'conversation': Conversation['sid'],
        }
        response = scraper.get('https://pi.ai/api/chat/history', params=params, cookies=Conversation['cookies'])
        if 'Just a moment' in response.text:
            return {
                'error': 'cloudflare detected',
                'traceback': 'Couldnt get the chat history'
            }
        return response.json()

session = cloudscraper.session()

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    },
    sess=session
)

scraper.headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'deflate,gzip,br',
}

def Ask_PI(message,sid,cookies):
    json_data = {
        'text': message,
        'conversation': sid,
        'mode': 'BASE',
    }
    response = scraper.post('https://pi.ai/api/chat', json=json_data, cookies=cookies)
    
    if 'Just a moment' in response.text:
        return [{
            'error': 'cloudflare detected',
            'text': 'Couldnt generate the answer because we got detected by cloudflare please try again later'
        }
        ]
    result = []
    for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
        if line.startswith('data: {"text":'):
            result.append(json.loads(line.split('data: ')[1].encode('utf-8')))
        if line.startswith('data: {"title":'):
            result.append(json.loads(line.split('data: ')[1].encode('utf-8')))
            
    return result
