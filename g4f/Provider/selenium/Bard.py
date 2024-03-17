from __future__ import annotations

import time
import os

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


class Bard(AbstractProvider):
    url = "https://bard.google.com"
    working = False
    needs_auth = True
    webdriver = True

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
        prompt = format_prompt(messages)
        session = WebDriverSession(webdriver, user_data_dir, headless, proxy=proxy)
        with session as driver:
            try:
                driver.get(f"{cls.url}/chat")
                wait = WebDriverWait(driver, 10 if headless else 240)
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ql-editor.textarea")))
            except:
                # Reopen browser for login
                if not webdriver:
                    driver = session.reopen()
                    driver.get(f"{cls.url}/chat")
                    login_url = os.environ.get("G4F_LOGIN_URL")
                    if login_url:
                        yield f"Please login: [Google Bard]({login_url})\n\n"
                    wait = WebDriverWait(driver, 240)
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ql-editor.textarea")))
                else:
                    raise RuntimeError("Prompt textarea not found. You may not be logged in.")

            # Add hook in XMLHttpRequest
            script = """
const _http_request_open = XMLHttpRequest.prototype.open;
window._message = "";
XMLHttpRequest.prototype.open = function(method, url) {
    if (url.includes("/assistant.lamda.BardFrontendService/StreamGenerate")) {
        this.addEventListener("load", (event) => {
            window._message = JSON.parse(JSON.parse(this.responseText.split("\\n")[3])[0][2])[4][0][1][0];
        });
    }
    return _http_request_open.call(this, method, url);
}
"""
            driver.execute_script(script)

            element_send_text(driver.find_element(By.CSS_SELECTOR, "div.ql-editor.textarea"), prompt)
            
            while True:
                chunk = driver.execute_script("return window._message;")
                if chunk:
                    yield chunk
                    return
                else:
                    time.sleep(0.1)