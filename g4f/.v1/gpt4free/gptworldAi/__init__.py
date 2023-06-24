# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/23 13:37
@Auth ： Hp_mzx
@File ：__init__.py.py
@IDE ：PyCharm
"""
import json
import uuid
import random
import binascii
import requests
import Crypto.Cipher.AES as AES
from fake_useragent import UserAgent

class ChatCompletion:
    @staticmethod
    def create(messages:[],proxy: str = None):
        url = "https://chat.getgpt.world/api/chat/stream"
        headers = {
            "Content-Type": "application/json",
            "Referer": "https://chat.getgpt.world/",
               'user-agent': UserAgent().random,
        }
        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else None
        data = json.dumps({
            "messages": messages,
            "frequency_penalty": 0,
            "max_tokens": 4000,
            "model": "gpt-3.5-turbo",
            "presence_penalty": 0,
            "temperature": 1,
            "top_p": 1,
            "stream": True,
            "uuid": str(uuid.uuid4())
        })
        signature = ChatCompletion.encrypt(data)
        res = requests.post(url, headers=headers, data=json.dumps({"signature": signature}), proxies=proxies,stream=True)
        for chunk in res.iter_content(chunk_size=None):
            res.raise_for_status()
            datas = chunk.decode('utf-8').split('data: ')
            for data in datas:
                if not data or "[DONE]" in data:
                    continue
                data_json = json.loads(data)
                content = data_json['choices'][0]['delta'].get('content')
                if content:
                    yield content


    @staticmethod
    def random_token(e):
        token = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        n = len(token)
        return "".join([token[random.randint(0, n - 1)] for i in range(e)])

    @staticmethod
    def encrypt(e):
        t = ChatCompletion.random_token(16).encode('utf-8')
        n = ChatCompletion.random_token(16).encode('utf-8')
        r = e.encode('utf-8')
        cipher = AES.new(t, AES.MODE_CBC, n)
        ciphertext = cipher.encrypt(ChatCompletion.__pad_data(r))
        return binascii.hexlify(ciphertext).decode('utf-8') + t.decode('utf-8') + n.decode('utf-8')

    @staticmethod
    def __pad_data(data: bytes) -> bytes:
        block_size = AES.block_size
        padding_size = block_size - len(data) % block_size
        padding = bytes([padding_size] * padding_size)
        return data + padding


class Completion:
    @staticmethod
    def create(prompt:str,proxy:str=None):
        return ChatCompletion.create([
            {
                "content": "You are ChatGPT, a large language model trained by OpenAI.\nCarefully heed the user's instructions. \nRespond using Markdown.",
                "role": "system"
            },
            {"role": "user", "content": prompt}
        ], proxy)


if __name__ == '__main__':
    # single completion
    text = ""
    for chunk in Completion.create("你是谁", "127.0.0.1:7890"):
        text = text + chunk
        print(chunk, end="", flush=True)
    print()


    #chat completion
    message = []
    while True:
        prompt = input("请输入问题：")
        message.append({"role": "user","content": prompt})
        text = ""
        for chunk in ChatCompletion.create(message,'127.0.0.1:7890'):
            text = text+chunk
            print(chunk, end="", flush=True)
        print()
        message.append({"role": "assistant", "content": text})