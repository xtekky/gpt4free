from __future__ import annotations

import json
import base64
import random
import requests

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, format_prompt
from ...errors import ModelNotFoundError, ModelNotSupportedError, ResponseError
from ...requests import StreamSession, raise_for_status
from ...providers.response import FinishReason
from ...image import ImageResponse
from ..helper import format_image_prompt
from .models import default_model, default_image_model, model_aliases, fallback_models
from ... import debug

class HuggingFaceInference(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co"
    working = True

    default_model = default_model
    default_image_model = default_image_model
    model_aliases = model_aliases

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls.models:
            models = fallback_models.copy()
            url = "https://huggingface.co/api/models?inference=warm&pipeline_tag=text-generation"
            extra_models = [model["id"] for model in requests.get(url).json()]
            extra_models.sort()
            models.extend([model for model in extra_models if model not in models])
            if not cls.image_models:
                url = "https://huggingface.co/api/models?pipeline_tag=text-to-image"
                cls.image_models = [model["id"] for model in requests.get(url).json() if model["trendingScore"] >= 20]
                cls.image_models.sort()
                models.extend([model for model in cls.image_models if model not in models])
            cls.models = models
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
        max_tokens: int = 1024,
        temperature: float = None,
        prompt: str = None,
        action: str = None,
        extra_data: dict = {},
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
        if cls.get_models() and model in cls.image_models:
            stream = False
            prompt = format_image_prompt(messages, prompt)
            payload = {"inputs": prompt, "parameters": {"seed": random.randint(0, 2**32), **extra_data}}
        else:
            params = {
                "return_full_text": False,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **extra_data
            }
            do_continue = action == "continue"
        async with StreamSession(
            headers=headers,
            proxy=proxy,
            timeout=600
        ) as session:
            if payload is None:
                async with session.get(f"https://huggingface.co/api/models/{model}") as response:
                    if response.status == 404:
                        raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__}")
                    await raise_for_status(response)
                    model_data = await response.json()
                    model_type = None
                    if "config" in model_data and "model_type" in model_data["config"]:
                        model_type = model_data["config"]["model_type"]
                    debug.log(f"Model type: {model_type}")
                    inputs = get_inputs(messages, model_data, model_type, do_continue)
                    debug.log(f"Inputs len: {len(inputs)}")
                    if len(inputs) > 4096:
                        if len(messages) > 6:
                            messages = messages[:3] + messages[-3:]
                        else:
                            messages = [m for m in messages if m["role"] == "system"] + [messages[-1]]
                        inputs = get_inputs(messages, model_data, model_type, do_continue)
                        debug.log(f"New len: {len(inputs)}")
                    if model_type == "gpt2" and max_tokens >= 1024:
                        params["max_new_tokens"] = 512
                payload = {"inputs": inputs, "parameters": params, "stream": stream}

            async with session.post(f"{api_base.rstrip('/')}/models/{model}", json=payload) as response:
                if response.status == 404:
                    raise ModelNotFoundError(f"Model is not supported: {model}")
                await raise_for_status(response)
                if stream:
                    first = True
                    is_special = False
                    async for line in response.iter_lines():
                        if line.startswith(b"data:"):
                            data = json.loads(line[5:])
                            if "error" in data:
                                raise ResponseError(data["error"])
                            if not data["token"]["special"]:
                                chunk = data["token"]["text"]
                                if first and not do_continue:
                                    first = False
                                    chunk = chunk.lstrip()
                                if chunk:
                                    yield chunk
                            else:
                                is_special = True
                    debug.log(f"Special token: {is_special}")
                    yield FinishReason("stop" if is_special else "length")
                else:
                    if response.headers["content-type"].startswith("image/"):
                        base64_data = base64.b64encode(b"".join([chunk async for chunk in response.iter_content()]))
                        url = f"data:{response.headers['content-type']};base64,{base64_data.decode()}"
                        yield ImageResponse(url, prompt)
                    else:
                        yield (await response.json())[0]["generated_text"].strip()

def format_prompt_mistral(messages: Messages, do_continue: bool = False) -> str:
    system_messages = [message["content"] for message in messages if message["role"] == "system"]
    question = " ".join([messages[-1]["content"], *system_messages])
    history = "\n".join([
        f"<s>[INST]{messages[idx-1]['content']} [/INST] {message['content']}</s>"
        for idx, message in enumerate(messages)
        if message["role"] == "assistant"
    ])
    if do_continue:
        return history[:-len('</s>')]
    return f"{history}\n<s>[INST] {question} [/INST]"

def format_prompt_qwen(messages: Messages, do_continue: bool = False) -> str:
    prompt = "".join([
        f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>\n" for message in messages
    ]) + ("" if do_continue else "<|im_start|>assistant\n")
    if do_continue:
        return prompt[:-len("\n<|im_end|>\n")]
    return prompt

def format_prompt_qwen2(messages: Messages, do_continue: bool = False) -> str:
    prompt = "".join([
        f"\u003C｜{message['role'].capitalize()}｜\u003E{message['content']}\u003C｜end▁of▁sentence｜\u003E" for message in messages
    ]) + ("" if do_continue else "\u003C｜Assistant｜\u003E")
    if do_continue:
        return prompt[:-len("\u003C｜Assistant｜\u003E")]
    return prompt

def format_prompt_llama(messages: Messages, do_continue: bool = False) -> str:
    prompt = "<|begin_of_text|>" + "".join([
        f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}\n<|eot_id|>\n" for message in messages
    ]) + ("" if do_continue else "<|start_header_id|>assistant<|end_header_id|>\n\n")
    if do_continue:
        return prompt[:-len("\n<|eot_id|>\n")]
    return prompt

def format_prompt_custom(messages: Messages, end_token: str = "</s>", do_continue: bool = False) -> str:
    prompt = "".join([
        f"<|{message['role']}|>\n{message['content']}{end_token}\n" for message in messages
    ]) + ("" if do_continue else "<|assistant|>\n")
    if do_continue:
        return prompt[:-len(end_token + "\n")]
    return prompt

def get_inputs(messages: Messages, model_data: dict, model_type: str, do_continue: bool = False) -> str:
    if model_type in ("gpt2", "gpt_neo", "gemma", "gemma2"):
        inputs = format_prompt(messages, do_continue=do_continue)
    elif model_type == "mistral" and model_data.get("author")  == "mistralai":
        inputs = format_prompt_mistral(messages, do_continue)
    elif "config" in model_data and "tokenizer_config" in model_data["config"] and "eos_token" in model_data["config"]["tokenizer_config"]:
        eos_token = model_data["config"]["tokenizer_config"]["eos_token"]
        if eos_token in ("<|endoftext|>", "<eos>", "</s>"):
            inputs = format_prompt_custom(messages, eos_token, do_continue)
        elif eos_token == "<|im_end|>":
            inputs = format_prompt_qwen(messages, do_continue)
        elif "content" in eos_token and eos_token["content"] == "\u003C｜end▁of▁sentence｜\u003E":
            inputs = format_prompt_qwen2(messages, do_continue)
        elif eos_token == "<|eot_id|>":
            inputs = format_prompt_llama(messages, do_continue)
        else:
            inputs = format_prompt(messages, do_continue=do_continue)
    else:
        inputs = format_prompt(messages, do_continue=do_continue)
    return inputs