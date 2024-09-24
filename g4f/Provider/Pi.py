from __future__ import annotations

import json

from ..typing import CreateResult, Messages
from .base_provider import AbstractProvider, format_prompt
from ..requests import Session, get_session_from_browser, raise_for_status

class Pi(AbstractProvider):
    url             = "https://pi.ai/talk"
    working         = True
    supports_stream = True
    _session = None
    default_model = "pi"

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 180,
        conversation_id: str = None,
        webdriver: WebDriver = None,
        **kwargs
    ) -> CreateResult:
        if cls._session is None:
            cls._session = get_session_from_browser(url=cls.url, proxy=proxy, timeout=timeout)
        if not conversation_id:
            conversation_id = cls.start_conversation(cls._session)
            prompt = format_prompt(messages)
        else:
            prompt = messages[-1]["content"]
        answer = cls.ask(cls._session, prompt, conversation_id)
        for line in answer:
            if "text" in line:
                yield line["text"]
    
    @classmethod
    def start_conversation(cls, session: Session) -> str:
        response = session.post('https://pi.ai/api/chat/start', data="{}", headers={
            'accept': 'application/json',
            'x-api-version': '3'
        })
        raise_for_status(response)
        return response.json()['conversations'][0]['sid']
        
    def get_chat_history(session: Session, conversation_id: str):
        params = {
            'conversation': conversation_id,
        }
        response = session.get('https://pi.ai/api/chat/history', params=params)
        raise_for_status(response)
        return response.json()

    def ask(session: Session, prompt: str, conversation_id: str):
        json_data = {
            'text': prompt,
            'conversation': conversation_id,
            'mode': 'BASE',
        }
        response = session.post('https://pi.ai/api/chat', json=json_data, stream=True)
        raise_for_status(response)
        for line in response.iter_lines():
            if line.startswith(b'data: {"text":'):
               yield json.loads(line.split(b'data: ')[1])
            elif line.startswith(b'data: {"title":'):
               yield json.loads(line.split(b'data: ')[1])
        
