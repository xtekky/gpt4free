from __future__ import annotations

import time, json

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import WebDriver, format_prompt, get_browser

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
        browser: WebDriver = None,
        hidden_display: bool = True,
        **kwargs
    ) -> CreateResult:
        if browser:
            driver = browser
        else:
            if hidden_display:
                driver, display = get_browser("", True, proxy)
            else:
                driver = get_browser("", False, proxy)

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver.get(cls.url)
        try:
            # Wait for page load
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body:not(.no-js)"))
            )
            # Send message
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
window.reader = response.body.getReader();
"""
            data = {
                "botId": "4738",
                "conversation_scenario": 3,
                "message": format_prompt(messages),
                "messageType": 1
            }
            driver.execute_script(script.replace("{body}", json.dumps(data)))
            script = """
chunk = await window.reader.read();
if (chunk['done']) return null;
text = (new TextDecoder ()).decode(chunk['value']);
content = '';
text.split('\\n').forEach((line, index) => {
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
                chunk = driver.execute_script(script):
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
        finally:
            driver.close()
            if not browser:
                time.sleep(0.1)
                driver.quit()
            if hidden_display:
                display.stop()