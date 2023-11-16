from __future__ import annotations

import time, json

try:
    from selenium.webdriver.remote.webdriver import WebDriver
except ImportError:
    class WebDriver():
        pass

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import format_prompt, get_browser

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
        display: bool = True,
        **kwargs
    ) -> CreateResult:
        if not browser:
            if display:
                driver, display = get_browser("", True, proxy)
            else:
                display = get_browser("", False, proxy)
        else:
            driver = browser

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        driver.get(cls.url)
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body:not(.no-js)"))
            )
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
text = await (new Response(chunk['value']).text());
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
            while chunk := driver.execute_script(script):
                yield chunk
        finally:
            driver.close()
            if not browser:
                time.sleep(0.1)
                driver.quit()
            if display:
                display.stop()