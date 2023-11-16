from __future__ import annotations

import time
from urllib.parse import quote
try:
    from selenium.webdriver.remote.webdriver import WebDriver
except ImportError:
    class WebDriver():
        pass

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import format_prompt, get_browser

class Phind(BaseProvider):
    url = "https://www.phind.com"
    working = True
    supports_gpt_4 = True
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
        creative_mode: bool = None,
        display: bool = True,
        **kwargs
    ) -> CreateResult:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        if browser:
            driver = browser
        else:
            if display:
                driver, display = get_browser("", True, proxy)
            else:
                driver = get_browser("", False, proxy)

        prompt = quote(format_prompt(messages))
        driver.get(f"{cls.url}/search?q={prompt}&source=searchbox")

        if model.startswith("gpt-4") or creative_mode:
            wait = WebDriverWait(driver, timeout)
            # Open dropdown
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.text-dark.dropdown-toggle")))
            driver.find_element(By.CSS_SELECTOR, "button.text-dark.dropdown-toggle").click()
            # Enable GPT-4
            wait.until(EC.visibility_of_element_located((By.XPATH, "//button[text()='GPT-4']")))
            if model.startswith("gpt-4"):
                driver.find_element(By.XPATH, "//button[text()='GPT-4']").click()
            # Enable creative mode
            if creative_mode or creative_mode == None:
                driver.find_element(By.ID, "Creative Mode").click()
            # Submit question
            driver.find_element(By.CSS_SELECTOR, ".search-bar-input-group button[type='submit']").click()
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".search-container")))

        try:
            script = """
window._fetch = window.fetch;
window.fetch = (url, options) => {
    const result = window._fetch(url, options);
    if (url != "/api/infer/answer") return result;
    result.then((response) => {
        if (!response.body.locked) {
            window.reader = response.body.getReader();
        }
    });
    return new Promise((resolve, reject) => {
        resolve(new Response(new ReadableStream()))
    });
}
"""
            driver.execute_script(script)
            script = """
if(window.reader) {
    chunk = await window.reader.read();
    if (chunk['done']) return null;
    text = await (new Response(chunk['value']).text());
    content = '';
    text.split('\\r\\n').forEach((line, index) => {
        if (line.startsWith('data: ')) {
            line = line.substring('data: '.length);
            if (!line.startsWith('<PHIND_METADATA>')) {
                if (line) content += line;
                else content += '\\n';
            }
        }
    });
    return content.replace('\\n\\n', '\\n');
} else {
    return ''
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