from __future__ import annotations

import time

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider
from ..helper import format_prompt

models = {
    "meta-llama/Llama-2-7b-chat-hf": {"name": "Llama-2-7b"},
    "meta-llama/Llama-2-13b-chat-hf": {"name": "Llama-2-13b"},
    "meta-llama/Llama-2-70b-chat-hf": {"name": "Llama-2-70b"},
    "codellama/CodeLlama-7b-Instruct-hf": {"name": "Code-Llama-7b"},
    "codellama/CodeLlama-13b-Instruct-hf": {"name": "Code-Llama-13b"},
    "codellama/CodeLlama-34b-Instruct-hf": {"name": "Code-Llama-34b"},
    "gpt-3.5-turbo": {"name": "GPT-3.5-Turbo"},
    "gpt-3.5-turbo-instruct": {"name": "GPT-3.5-Turbo-Instruct"},
    "gpt-4": {"name": "GPT-4"},
    "palm": {"name": "Google-PaLM"},
}

class Poe(AbstractProvider):
    url = "https://poe.com"
    working = False
    needs_auth = True
    supports_stream = True
    
    models = models.keys()

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        webdriver: WebDriver = None,
        user_data_dir: str = None,
        headless: bool = True,
        **kwargs
    ) -> CreateResult:
        if not model:
            model = "gpt-3.5-turbo"
        elif model not in models:
            raise ValueError(f"Model are not supported: {model}")
        prompt = format_prompt(messages)

        session = WebDriverSession(webdriver, user_data_dir, headless, proxy=proxy)
        with session as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
    window._message = window._last_message = "";
    window._message_finished = false;
    class ProxiedWebSocket extends WebSocket {
    constructor(url, options) {
        super(url, options);
        this.addEventListener("message", (e) => {
            const data = JSON.parse(JSON.parse(e.data)["messages"][0])["payload"]["data"];
            if ("messageAdded" in data) {
                if (data["messageAdded"]["author"] != "human") {
                    window._message = data["messageAdded"]["text"];
                    if (data["messageAdded"]["state"] == "complete") {
                        window._message_finished = true;
                    }
                }
            }
        });
    }
    }
    window.WebSocket = ProxiedWebSocket;
    """
            })

            try:
                driver.get(f"{cls.url}/{models[model]['name']}")
                wait = WebDriverWait(driver, 10 if headless else 240)
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[class^='GrowingTextArea']")))
            except:
                # Reopen browser for login
                if not webdriver:
                    driver = session.reopen()
                    driver.get(f"{cls.url}/{models[model]['name']}")
                    wait = WebDriverWait(driver, 240)
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[class^='GrowingTextArea']")))
                else:
                    raise RuntimeError("Prompt textarea not found. You may not be logged in.")

            element_send_text(driver.find_element(By.CSS_SELECTOR, "footer textarea[class^='GrowingTextArea']"), prompt)
            driver.find_element(By.CSS_SELECTOR, "footer button[class*='ChatMessageSendButton']").click()

            script = """
if(window._message && window._message != window._last_message) {
    try {
        return window._message.substring(window._last_message.length);
    } finally {
        window._last_message = window._message;
    }
} else if(window._message_finished) {
    return null;
} else {
    return '';
}
"""
            while True:
                chunk = driver.execute_script(script)
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)
