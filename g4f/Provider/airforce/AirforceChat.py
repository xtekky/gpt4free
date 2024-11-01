from __future__ import annotations
import re
from aiohttp import ClientSession
import json
from typing import List

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt

def clean_response(text: str) -> str:
    """Clean response from unwanted patterns."""
    patterns = [
        r"One message exceeds the \d+chars per message limit\..+https:\/\/discord\.com\/invite\/\S+",
        r"Rate limit \(\d+\/minute\) exceeded\. Join our discord for more: .+https:\/\/discord\.com\/invite\/\S+",
        r"Rate limit \(\d+\/hour\) exceeded\. Join our discord for more: https:\/\/discord\.com\/invite\/\S+",
        r"</s>", # zephyr-7b-beta
    ]

    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text.strip()

def split_message(message: dict, chunk_size: int = 995) -> List[dict]:
    """Split a message into chunks of specified size."""
    content = message.get('content', '')
    if len(content) <= chunk_size:
        return [message]

    chunks = []
    while content:
        chunk = content[:chunk_size]
        content = content[chunk_size:]
        chunks.append({
            'role': message['role'],
            'content': chunk
        })
    return chunks

def split_messages(messages: Messages, chunk_size: int = 995) -> Messages:
    """Split all messages that exceed chunk_size into smaller messages."""
    result = []
    for message in messages:
        result.extend(split_message(message, chunk_size))
    return result

class AirforceChat(AsyncGeneratorProvider, ProviderModelMixin):
    label = "AirForce Chat"
    api_endpoint_completions = "https://api.airforce/chat/completions"  # Замініть на реальний ендпоінт
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'llama-3-70b-chat'
    text_models = [
        # anthropic
        'claude-3-haiku-20240307', 
        'claude-3-sonnet-20240229', 
        'claude-3-5-sonnet-20240620', 
        'claude-3-5-sonnet-20241022', 
        'claude-3-opus-20240229', 
        
        # openai
        'chatgpt-4o-latest',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-2024-05-13',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o-mini',
        'gpt-4o-2024-08-06',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'gpt-4o',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-0125-preview',
        'gpt-4-1106-preview',
        
        # meta-llama
        default_model,
        'llama-3-70b-chat-turbo',
        'llama-3-8b-chat',
        'llama-3-8b-chat-turbo',
        'llama-3-70b-chat-lite',
        'llama-3-8b-chat-lite',
        'llama-2-13b-chat',
        'llama-3.1-405b-turbo',
        'llama-3.1-70b-turbo',
        'llama-3.1-8b-turbo',
        'LlamaGuard-2-8b',
        'llamaguard-7b',
        'Llama-Vision-Free',
        'Llama-Guard-7b',
        'Llama-3.2-90B-Vision-Instruct-Turbo',
        'Meta-Llama-Guard-3-8B',
        'Llama-3.2-11B-Vision-Instruct-Turbo',
        'Llama-Guard-3-11B-Vision-Turbo',
        'Llama-3.2-3B-Instruct-Turbo',
        'Llama-3.2-1B-Instruct-Turbo',
        'llama-2-7b-chat-int8',
        'llama-2-7b-chat-fp16',
        'Llama 3.1 405B Instruct',
        'Llama 3.1 70B Instruct',
        'Llama 3.1 8B Instruct',
        
        # mistral-ai
        'Mixtral-8x7B-Instruct-v0.1',
        'Mixtral-8x22B-Instruct-v0.1',
        'Mistral-7B-Instruct-v0.1',
        'Mistral-7B-Instruct-v0.2',
        'Mistral-7B-Instruct-v0.3',
        
        # Gryphe
        'MythoMax-L2-13b-Lite',
        'MythoMax-L2-13b',
        
        # openchat
        'openchat-3.5-0106',
        
        # qwen
        #'Qwen1.5-72B-Chat', Пуста відповідь
        #'Qwen1.5-110B-Chat', Пуста відповідь
        'Qwen2-72B-Instruct',
        'Qwen2.5-7B-Instruct-Turbo',
        'Qwen2.5-72B-Instruct-Turbo',
        
        # google
        'gemma-2b-it',
        'gemma-2-9b-it',
        'gemma-2-27b-it',
        
        # gemini
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        
        # databricks
        'dbrx-instruct',
        
        # deepseek-ai
        'deepseek-coder-6.7b-base',
        'deepseek-coder-6.7b-instruct',
        'deepseek-math-7b-instruct',
        
        # NousResearch
        'deepseek-math-7b-instruct',
        'Nous-Hermes-2-Mixtral-8x7B-DPO',
        'hermes-2-pro-mistral-7b',
        
        # teknium
        'openhermes-2.5-mistral-7b',
        
        # microsoft
        'WizardLM-2-8x22B',
        'phi-2',
        
        # upstage
        'SOLAR-10.7B-Instruct-v1.0',
        
        # pawan
        'cosmosrp',
        
        # liquid
        'lfm-40b-moe',
        
        # DiscoResearch
        'discolm-german-7b-v1',
        
        # tiiuae
        'falcon-7b-instruct',
        
        # defog
        'sqlcoder-7b-2',
        
        # tinyllama
        'tinyllama-1.1b-chat',
        
        # HuggingFaceH4
        'zephyr-7b-beta',
    ]
    
    models = [*text_models]
    
    model_aliases = {
		# anthropic
		"claude-3-haiku": "claude-3-haiku-20240307",
		"claude-3-sonnet": "claude-3-sonnet-20240229",
		"claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
		"claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
		"claude-3-opus": "claude-3-opus-20240229",

		# openai
		"gpt-4o": "chatgpt-4o-latest",
		#"gpt-4": "gpt-4",
		#"gpt-4-turbo": "gpt-4-turbo",
		"gpt-4o": "gpt-4o-2024-05-13",
		"gpt-4o-mini": "gpt-4o-mini-2024-07-18",
		#"gpt-4o-mini": "gpt-4o-mini",
		"gpt-4o": "gpt-4o-2024-08-06",
		"gpt-3.5-turbo": "gpt-3.5-turbo",
		"gpt-3.5-turbo": "gpt-3.5-turbo-0125",
		"gpt-3.5-turbo": "gpt-3.5-turbo-1106",
		#"gpt-4o": "gpt-4o",
		"gpt-4-turbo": "gpt-4-turbo-2024-04-09",
		"gpt-4": "gpt-4-0125-preview",
		"gpt-4": "gpt-4-1106-preview",

		# meta-llama
		"llama-3-70b": "llama-3-70b-chat",
		"llama-3-8b": "llama-3-8b-chat",
		"llama-3-8b": "llama-3-8b-chat-turbo",
		"llama-3-70b": "llama-3-70b-chat-lite",
		"llama-3-8b": "llama-3-8b-chat-lite",
		"llama-2-13b": "llama-2-13b-chat",
		"llama-3.1-405b": "llama-3.1-405b-turbo",
		"llama-3.1-70b": "llama-3.1-70b-turbo",
		"llama-3.1-8b": "llama-3.1-8b-turbo",
		"llamaguard-2-8b": "LlamaGuard-2-8b",
		"llamaguard-7b": "llamaguard-7b",
		#"llama_vision_free": "Llama-Vision-Free", # Unknown
		"llamaguard-7b": "Llama-Guard-7b",
		"llama-3.2-90b": "Llama-3.2-90B-Vision-Instruct-Turbo",
		"llamaguard-3-8b": "Meta-Llama-Guard-3-8B",
		"llama-3.2-11b": "Llama-3.2-11B-Vision-Instruct-Turbo",
		"llamaguard-3-11b": "Llama-Guard-3-11B-Vision-Turbo",
		"llama-3.2-3b": "Llama-3.2-3B-Instruct-Turbo",
		"llama-3.2-1b": "Llama-3.2-1B-Instruct-Turbo",
		"llama-2-7b": "llama-2-7b-chat-int8",
		"llama-2-7b": "llama-2-7b-chat-fp16",
		"llama-3.1-405b": "Llama 3.1 405B Instruct",
		"llama-3.1-70b": "Llama 3.1 70B Instruct",
		"llama-3.1-8b": "Llama 3.1 8B Instruct",

		# mistral-ai
		"mixtral-8x7b": "Mixtral-8x7B-Instruct-v0.1",
		"mixtral-8x22b": "Mixtral-8x22B-Instruct-v0.1",
		"mixtral-8x7b": "Mistral-7B-Instruct-v0.1",
		"mixtral-8x7b": "Mistral-7B-Instruct-v0.2",
		"mixtral-8x7b": "Mistral-7B-Instruct-v0.3",

		# Gryphe
		"mythomax-13b": "MythoMax-L2-13b-Lite",
		"mythomax-13b": "MythoMax-L2-13b",

		# openchat
		"openchat-3.5": "openchat-3.5-0106",

		# qwen
		#"qwen-1.5-72b": "Qwen1.5-72B-Chat", # Empty answer
		#"qwen-1.5-110b": "Qwen1.5-110B-Chat", # Empty answer
		"qwen-2-72b": "Qwen2-72B-Instruct",
		"qwen-2-5-7b": "Qwen2.5-7B-Instruct-Turbo",
		"qwen-2-5-72b": "Qwen2.5-72B-Instruct-Turbo",

		# google
		"gemma-2b": "gemma-2b-it",
		"gemma-2-9b": "gemma-2-9b-it",
		"gemma-2b-27b": "gemma-2-27b-it",

		# gemini
		"gemini-flash": "gemini-1.5-flash",
		"gemini-pro": "gemini-1.5-pro",

		# databricks
		"dbrx-instruct": "dbrx-instruct",

		# deepseek-ai
		#"deepseek-coder": "deepseek-coder-6.7b-base",
		"deepseek-coder": "deepseek-coder-6.7b-instruct",
		#"deepseek-math": "deepseek-math-7b-instruct",

		# NousResearch
		#"deepseek-math": "deepseek-math-7b-instruct",
		"hermes-2-dpo": "Nous-Hermes-2-Mixtral-8x7B-DPO",
		"hermes-2": "hermes-2-pro-mistral-7b",

		# teknium
		"openhermes-2.5": "openhermes-2.5-mistral-7b",

		# microsoft
		"wizardlm-2-8x22b": "WizardLM-2-8x22B",
		#"phi-2": "phi-2",

		# upstage
		"solar-10-7b": "SOLAR-10.7B-Instruct-v1.0",

		# pawan
		#"cosmosrp": "cosmosrp",

		# liquid
		"lfm-40b": "lfm-40b-moe",

		# DiscoResearch
		"german-7b": "discolm-german-7b-v1",

		# tiiuae
		#"falcon-7b": "falcon-7b-instruct",

		# defog
		#"sqlcoder-7b": "sqlcoder-7b-2",

		# tinyllama
		#"tinyllama-1b": "tinyllama-1.1b-chat",

		# HuggingFaceH4
		"zephyr-7b": "zephyr-7b-beta",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        max_tokens: str = 4096,
        temperature: str = 1,
        top_p: str = 1,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        chunked_messages = split_messages(messages)

        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'authorization': 'Bearer missing api key',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://llmplayground.net',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://llmplayground.net/',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }

        data = {
            "messages": chunked_messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint_completions, json=data, proxy=proxy) as response:
                response.raise_for_status()
                text = ""
                if stream:
                    async for line in response.content:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]
                            try:
                                chunk = json.loads(json_str)
                                if 'choices' in chunk and chunk['choices']:
                                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                                    text += content # Збираємо дельти
                            except json.JSONDecodeError as e:
                                print(f"Error decoding JSON: {json_str}, Error: {e}")
                        elif line.strip() == "[DONE]":
                            break
                    yield clean_response(text)
                else:
                    response_json = await response.json()
                    text = response_json["choices"][0]["message"]["content"]
                    yield clean_response(text)

