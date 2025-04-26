from __future__ import annotations

import time

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider
from ..helper import format_prompt

models = {
    "theb-ai": "TheB.AI",
    "theb-ai-free": "TheB.AI Free",
    "gpt-3.5-turbo": "GPT-3.5 Turbo (New)",
    "gpt-3.5-turbo-16k": "GPT-3.5-16K",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4-32k": "GPT-4 32K",
    "claude-2": "Claude 2",
    "claude-instant-1": "Claude Instant 1.2",
    "palm-2": "PaLM 2",
    "palm-2-32k": "PaLM 2 32K",
    "palm-2-codey": "Codey",
    "palm-2-codey-32k": "Codey 32K",
    "vicuna-13b-v1.5": "Vicuna v1.5 13B",
    "llama-2-7b-chat": "Llama 2 7B",
    "llama-2-13b-chat": "Llama 2 13B",
    "llama-2-70b-chat": "Llama 2 70B",
    "code-llama-7b": "Code Llama 7B",
    "code-llama-13b": "Code Llama 13B",
    "code-llama-34b": "Code Llama 34B",
    "qwen-7b-chat": "Qwen 7B"
}

class Theb(AbstractProvider):
    label = "TheB.AI"
    url = "https://beta.theb.ai"
    working = False
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
        virtual_display: bool = True,
        **kwargs
    ) -> CreateResult:
        if model in models:
            model = models[model]
        prompt = format_prompt(messages)
        web_session = WebDriverSession(webdriver, virtual_display=virtual_display, proxy=proxy)
        with web_session as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.keys import Keys

            # Register fetch hook
            script = """
window._fetch = window.fetch;
window.fetch = async (url, options) => {
    // Call parent fetch method
    const response = await window._fetch(url, options);
    if (!url.startsWith("/api/conversation")) {
        return result;
    }
    // Copy response
    copy = response.clone();
    window._reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    return copy;
}
window._last_message = "";
"""
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": script
            })

            try:
                driver.get(f"{cls.url}/home")
                wait = WebDriverWait(driver, 5)
                wait.until(EC.visibility_of_element_located((By.ID, "textareaAutosize")))
            except:
                driver = web_session.reopen()
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": script
                })
                driver.get(f"{cls.url}/home")
                wait = WebDriverWait(driver, 240)
                wait.until(EC.visibility_of_element_located((By.ID, "textareaAutosize")))

            try:
                driver.find_element(By.CSS_SELECTOR, ".driver-overlay").click()
                driver.find_element(By.CSS_SELECTOR, ".driver-overlay").click()
            except:
                pass
            if model:
                # Load model panel
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#SelectModel svg")))
                time.sleep(0.1)
                driver.find_element(By.CSS_SELECTOR, "#SelectModel svg").click()
                try:
                    driver.find_element(By.CSS_SELECTOR, ".driver-overlay").click()
                    driver.find_element(By.CSS_SELECTOR, ".driver-overlay").click()
                except:
                    pass
                # Select model
                selector = f"div.flex-col div.items-center span[title='{model}']"
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                span = driver.find_element(By.CSS_SELECTOR, selector)
                container = span.find_element(By.XPATH, "//div/../..")
                button = container.find_element(By.CSS_SELECTOR, "button.btn-blue.btn-small.border")
                button.click()


            # Submit prompt
            wait.until(EC.visibility_of_element_located((By.ID, "textareaAutosize")))
            element_send_text(driver.find_element(By.ID, "textareaAutosize"), prompt)

            # Read response with reader
            script = """
if(window._reader) {
    chunk = await window._reader.read();
    if (chunk['done']) {
        return null;
    }
    message = '';
    chunk['value'].split('\\r\\n').forEach((line, index) => {
        if (line.startsWith('data: ')) {
            try {
                line = JSON.parse(line.substring('data: '.length));
                message = line["args"]["content"];
            } catch(e) { }
        }
    });
    if (message) {
        try {
            return message.substring(window._last_message.length);
        } finally {
            window._last_message = message;
        }
    }
}
return '';
"""
            while True:
                chunk = driver.execute_script(script)
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)
