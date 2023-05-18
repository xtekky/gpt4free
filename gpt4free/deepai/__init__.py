import requests
import json
import hashlib
import random
import string
from fake_useragent import UserAgent

class ChatCompletion:
    @classmethod
    def md5(self, text):
        return hashlib.md5(text.encode()).hexdigest()[::-1]

    @classmethod
    def get_api_key(self, user_agent):
        part1 = str(random.randint(0, 10**11))
        part2 = self.md5(user_agent+self.md5(user_agent+self.md5(user_agent+part1+"x")))
        return f"tryit-{part1}-{part2}"

    @classmethod
    def create(self, messages):
        user_agent = UserAgent().random
        api_key = self.get_api_key(user_agent)
        headers = {
          "api-key": api_key,
          "user-agent": user_agent
        }
        files = {
          "chat_style": (None, "chat"),
          "chatHistory": (None, json.dumps(messages))
        }

        r = requests.post("https://api.deepai.org/chat_response", headers=headers, files=files, stream=True)

        for chunk in r.iter_content(chunk_size=None):
            r.raise_for_status()
            yield chunk.decode()

class Completion:
    @classmethod
    def create(self, prompt):
        return ChatCompletion.create([
            {
                "role": "user", 
                "content": prompt
            }
        ])