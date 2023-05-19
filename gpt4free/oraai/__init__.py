import time
import json
import uuid
import random
import requests
from fake_useragent import UserAgent


class ChatCompletion:
    def __init__(self,proxy=None,chatbotId="5111b690-edd3-403f-b02a-607332d059f9"):
        self.userId = "auto:"+str(uuid.uuid4()) 
        self.chatbotId = chatbotId
        self.proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else None
        self.conversationId = None
        self.headers = {
            "Content-Type": "application/json",
            "Origin": "https://ora.ai",
            "Referer": "https://ora.ai/",
            'user-agent': UserAgent().random,
        }

    def create(self,prompt: str):
        url = "https://ora.ai/api/conversation"
        data = {
            "chatbotId": self.chatbotId,
            "config": False,
            "includeHistory": True,
            "input": prompt,
            "provider": "OPEN_AI",
            "userId": self.userId,
        }

        if self.conversationId:
            data["conversationId"] = self.conversationId
        response = requests.post(
            url,
            data=json.dumps(data),
            proxies=self.proxies,
            headers=self.headers
        )
        if response.status_code == 200:
            response_json = response.json()
            self.conversationId = response_json["conversationId"]
            return response_json["response"]

        raise ValueError(response.text)


    def generate_image(self,prompt:str):
        url = "https://ora.ai/api/images/request"
        data = {
            "prompt":prompt,
            "seed":random.randint(0, 4294967295)
        }
        response = requests.post(
            url,
            data=json.dumps(data),
            proxies=self.proxies,
            headers=self.headers
        )
        if response.status_code == 200:
            inferenceId = response.json()["id"]
        else:
            raise ValueError(response.text)

        data = {
            "chatbotId":self.chatbotId,
            "inferenceId":inferenceId,
            "userId":self.userId,
            "userInput":"/generate " + prompt
        }
        print(data)
        if self.conversationId:
            data["conversationId"] = self.conversationId
        while True:
            response = requests.post(
                "https://ora.ai/api/images/check",
                data=json.dumps(data),
                proxies=self.proxies,
                headers=self.headers
            )
            if response.status_code == 200:
                response_json = response.json()
                if response_json.get("conversationId"):
                    self.conversationId = response_json["conversationId"]
                    return response_json["inference"]["images"][0]["uri"]
                else:
                    time.sleep(0.5)
            else:
                raise ValueError(response.text)


class Completion:
    @classmethod
    def create(self, prompt, proxy):
        return ChatCompletion(proxy).create(prompt)






