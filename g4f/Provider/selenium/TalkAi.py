from __future__ import annotations

import time, json, time

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider
from ...webdriver import WebDriver, WebDriverSession

class TalkAi(AbstractProvider):
    url = "https://talkai.info"
    working = False
    supports_gpt_35_turbo = True
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        webdriver: WebDriver = None,
        **kwargs
    ) -> CreateResult:
        with WebDriverSession(webdriver, "", virtual_display=True, proxy=proxy) as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver.get(f"{cls.url}/chat/")
    
            # Wait for page load
            WebDriverWait(driver, 240).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body.chat-page"))
            )
            
            data = {
                "type": "chat",
                "message": messages[-1]["content"],
                "messagesHistory": [{
                    "from": "you" if message["role"] == "user" else "chatGPT",
                    "content": message["content"]
                } for message in messages],
                "model": model if model else "gpt-3.5-turbo",
                "max_tokens": 2048,
                "temperature": 1,
                "top_p": 1,
                "presence_penalty":	0,
                "frequency_penalty": 0,
                **kwargs
            }
            script = """
const response = await fetch("/chat/send2/", {
    "headers": {
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
    "body": {body},
    "method": "POST"
});
window._reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
"""
            driver.execute_script(
                script.replace("{body}", json.dumps(json.dumps(data)))
            )
            # Read response
            while True:
                chunk = driver.execute_script("""
chunk = await window._reader.read();
if (chunk.done) {
    return null;
}
content = "";
for (line of chunk.value.split("\\n")) {
    if (line.startsWith('data: ')) {
        content += line.substring('data: '.length);
    }
}
return content;
""")
                if chunk:
                    yield chunk.replace("\\n", "\n")
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)
