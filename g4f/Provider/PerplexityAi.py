from __future__ import annotations

import time
try:
    from selenium.webdriver.remote.webdriver import WebDriver
except ImportError:
    class WebDriver():
        pass

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import format_prompt, get_browser

class PerplexityAi(BaseProvider):
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
        browser: WebDriver = None,
        copilot: bool = False,
        display: bool = True,
        **kwargs
    ) -> CreateResult:
        if browser:
            driver = browser
        else:
            if display:
                driver, display = get_browser("", True, proxy)
            else:
                driver = get_browser("", False, proxy)

        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        prompt = format_prompt(messages)

        driver.get(f"{cls.url}/")
        wait = WebDriverWait(driver, timeout)

        # Page loaded?
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea[placeholder='Ask anything...']")))

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
                    window._message_finished = data[0] == "query_answered";
                    window._web_results = content["web_results"];
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
                driver.find_element(By.CSS_SELECTOR, "img[alt='User avatar']")
                driver.find_element(By.CSS_SELECTOR, "button[data-testid='copilot-toggle']").click()
            except:
                pass

        # Enter question
        driver.find_element(By.CSS_SELECTOR, "textarea[placeholder='Ask anything...']").send_keys(prompt)
        # Submit question
        driver.find_element(By.CSS_SELECTOR, "button.bg-super svg[data-icon='arrow-right']").click()

        try:
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
        finally:
            driver.close()
            if not browser:
                time.sleep(0.1)
                driver.quit()
            if display:
                display.stop()