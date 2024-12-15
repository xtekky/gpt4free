from __future__ import annotations

import json
import base64
import random
import requests

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, format_prompt
from ...errors import ModelNotFoundError, ModelNotSupportedError, ResponseError
from ...requests import StreamSession, raise_for_status
from ...image import ImageResponse

from .HuggingChat import HuggingChat

class HuggingFace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co"
    working = True
    supports_message_history = True
    default_model = HuggingChat.default_model
    default_image_model = HuggingChat.default_image_model
    model_aliases = HuggingChat.model_aliases

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls.models:
            url = "https://huggingface.co/api/models?inference=warm&pipeline_tag=text-generation"
            cls.models = [model["id"] for model in requests.get(url).json()]
            cls.models.append("meta-llama/Llama-3.2-11B-Vision-Instruct")
            cls.models.append("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
            cls.models.sort()
        if not cls.image_models:
            url = "https://huggingface.co/api/models?pipeline_tag=text-to-image"
            cls.image_models = [model["id"] for model in requests.get(url).json() if model["trendingScore"] >= 20]
            cls.image_models.sort()
            cls.models.extend(cls.image_models)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        api_base: str = "https://api-inference.huggingface.co",
        api_key: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        try:
            model = cls.get_model(model)
        except ModelNotSupportedError:
            pass
        headers = {
            'accept': '*/*',
            'accept-language': 'en',
            'cache-control': 'no-cache',
            'origin': 'https://huggingface.co',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://huggingface.co/chat/',
            'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = None
        if model in cls.image_models:
            stream = False
            prompt = messages[-1]["content"] if prompt is None else prompt
            payload = {"inputs": prompt, "parameters": {"seed": random.randint(0, 2**32)}}
        else:
            params = {
                "return_full_text": False,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                **kwargs
            }
        async with StreamSession(
            headers=headers,
            proxy=proxy,
            timeout=600
        ) as session:
            if payload is None:
                async with session.get(f"https://huggingface.co/api/models/{model}") as response:
                    await raise_for_status(response)
                    model_data = await response.json()
                    model_type = None
                    if "config" in model_data and "model_type" in model_data["config"]:
                        model_type = model_data["config"]["model_type"]
                    if model_type in ("gpt2", "gpt_neo", "gemma", "gemma2"):
                        inputs = format_prompt(messages)
                    elif "config" in model_data and "tokenizer_config" in model_data["config"] and "eos_token" in model_data["config"]["tokenizer_config"]:
                        eos_token = model_data["config"]["tokenizer_config"]["eos_token"]
                        if eos_token in ("<|endoftext|>", "<eos>", "</s>"):
                            inputs = format_prompt_custom(messages, eos_token)
                        elif eos_token == "<|im_end|>":
                            inputs = format_prompt_qwen(messages)
                        elif eos_token == "<|eot_id|>":
                            inputs = format_prompt_llama(messages)
                        else:
                            inputs = format_prompt_default(messages)
                    else:
                        inputs = format_prompt_default(messages)
                    if model_type == "gpt2" and max_new_tokens >= 1024:
                        params["max_new_tokens"] = 512
                payload = {"inputs": inputs, "parameters": params, "stream": stream}

            async with session.post(f"{api_base.rstrip('/')}/models/{model}", json=payload) as response:
                if response.status == 404:
                    raise ModelNotFoundError(f"Model is not supported: {model}")
                await raise_for_status(response)
                if stream:
                    first = True
                    async for line in response.iter_lines():
                        if line.startswith(b"data:"):
                            data = json.loads(line[5:])
                            if "error" in data:
                                raise ResponseError(data["error"])
                            if not data["token"]["special"]:
                                chunk = data["token"]["text"]
                                if first:
                                    first = False
                                    chunk = chunk.lstrip()
                                if chunk:
                                    yield chunk
                else:
                    if response.headers["content-type"].startswith("image/"):
                        base64_data = base64.b64encode(b"".join([chunk async for chunk in response.iter_content()]))
                        url = f"data:{response.headers['content-type']};base64,{base64_data.decode()}"
                        yield ImageResponse(url, prompt)
                    else:
                        yield (await response.json())[0]["generated_text"].strip()

def format_prompt_default(messages: Messages) -> str:
    system_messages = [message["content"] for message in messages if message["role"] == "system"]
    question = " ".join([messages[-1]["content"], *system_messages])
    history = "".join([
        f"<s>[INST]{messages[idx-1]['content']} [/INST] {message['content']}</s>"
        for idx, message in enumerate(messages)
        if message["role"] == "assistant"
    ])
    return f"{history}<s>[INST] {question} [/INST]"

def format_prompt_qwen(messages: Messages) -> str:
    return "".join([
        f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>\n" for message in messages
    ]) + "<|im_start|>assistant\n"

def format_prompt_llama(messages: Messages) -> str:
    return "<|begin_of_text|>" + "".join([
        f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}\n<|eot_id|>\n" for message in messages
    ]) + "<|start_header_id|>assistant<|end_header_id|>\n\n"

def format_prompt_custom(messages: Messages, end_token: str = "</s>") -> str:
    return "".join([
        f"<|{message['role']}|>\n{message['content']}{end_token}\n" for message in messages
    ]) + "<|assistant|>\n"