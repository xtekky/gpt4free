from __future__ import annotations

import time, json

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import format_prompt
from ..webdriver import WebDriver, WebDriverSession

class MyShell(BaseProvider):
    url = "https://app.myshell.ai/chat"
    working = True
    supports_gpt_35_turbo = True
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 120,
        webdriver: WebDriver = None,
        **kwargs
    ) -> CreateResult:
        with WebDriverSession(webdriver, "", proxy=proxy) as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver.get(cls.url)

            # Wait for page load and cloudflare validation
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body:not(.no-js)"))
            )
            # Send request with message
            data = {
                "botId": "4738",
                "conversation_scenario": 3,
                "message": format_prompt(messages),
                "messageType": 1
            }
            script = """
response = await fetch("https://api.myshell.ai/v1/bot/chat/send_message", {
    "headers": {
        "accept": "application/json",
        "content-type": "application/json",
        "myshell-service-name": "organics-api",
        "visitor-id": localStorage.getItem("mix_visitorId")
    },
    "body": '{body}',
    "method": "POST"
})
window._reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
"""
            driver.execute_script(script.replace("{body}", json.dumps(data)))
            script = """
chunk = await window._reader.read();
if (chunk['done']) {
    return null;
}
content = '';
chunk['value'].split('\\n').forEach((line, index) => {
    if (line.startsWith('data: ')) {
        try {
            const data = JSON.parse(line.substring('data: '.length));
            if ('content' in data) {
                content += data['content'];
            }
        } catch(e) {}
    }
});
return content;
"""
            while True:
                chunk = driver.execute_script(script)
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)