from __future__ import annotations

import time

try:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    pass

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider
from ..helper import format_prompt
from ...webdriver import WebDriver, WebDriverSession, element_send_text

class PerplexityAi(AbstractProvider):
    url = "https://www.perplexity.ai"
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
        virtual_display: bool = True,
        copilot: bool = False,
        **kwargs
    ) -> CreateResult:
        with WebDriverSession(webdriver, "", virtual_display=virtual_display, proxy=proxy) as driver:
            prompt = format_prompt(messages)

            driver.get(f"{cls.url}/")
            wait = WebDriverWait(driver, timeout)

            # Is page loaded?
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[placeholder='Ask anything...']")))

            # Register WebSocket hook
            script = """
window._message = window._last_message = "";
window._message_finished = false;
const _socket_send = WebSocket.prototype.send;
WebSocket.prototype.send = function(...args) {
    if (!window.socket_onmessage) {
        window._socket_onmessage = this;
        this.addEventListener("message", (event) => {
            if (event.data.startsWith("42")) {
                let data = JSON.parse(event.data.substring(2));
                if (data[0] =="query_progress" || data[0] == "query_answered") {
                    let content = JSON.parse(data[1]["text"]);
                    if (data[1]["mode"] == "copilot") {
                        content = content[content.length-1]["content"]["answer"];
                        content = JSON.parse(content);
                    }
                    window._message = content["answer"];
                    if (!window._message_finished) {
                        window._message_finished = data[0] == "query_answered";
                    }
                }
            }
        });
    }
    return _socket_send.call(this, ...args);
};
"""
            driver.execute_script(script)

            if copilot:
                try:
                # Check for account
                    driver.find_element(By.CSS_SELECTOR, "img[alt='User avatar']")
                # Enable copilot
                    driver.find_element(By.CSS_SELECTOR, "button[data-testid='copilot-toggle']").click()
                except:
                    raise RuntimeError("You need a account for copilot")

            # Submit prompt
            element_send_text(driver.find_element(By.CSS_SELECTOR, "textarea[placeholder='Ask anything...']"), prompt)

            # Stream response
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