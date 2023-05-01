import requests, time, ast, json
from bs4 import BeautifulSoup
from hashlib import sha256

class Completion:
    # answer is returned with html formatting
    next_id = None
    messages = []
    answer = None

    def init(self):
        r = requests.get("https://italygpt.it")
        soup = BeautifulSoup(r.text, "html.parser")
        self.next_id = soup.find("input", {"name": "next_id"})["value"]
    
    def create(self, prompt: str, messages: list = []):
        try:
            r = requests.get("https://italygpt.it/question", params={"hash": sha256(self.next_id.encode()).hexdigest(), "prompt": prompt, "raw_messages": json.dumps(messages)}).json()
        except:
            r = requests.get("https://italygpt.it/question", params={"hash": sha256(self.next_id.encode()).hexdigest(), "prompt": prompt, "raw_messages": json.dumps(messages)}).text
            if "too many requests" in r.lower():
                # rate limit is 17 requests per 1 minute
                time.sleep(20)
                return self.create(prompt, messages)
        self.next_id = r["next_id"]
        self.messages = ast.literal_eval(r["raw_messages"])
        self.answer = r["response"]
        return self