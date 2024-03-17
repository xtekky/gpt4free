from __future__ import annotations

import time
from urllib.parse import quote

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider
from ..helper import format_prompt
from ...webdriver import WebDriver, WebDriverSession

class Phind(AbstractProvider):
    url = "https://www.phind.com"
    working = False
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
        webdriver: WebDriver = None,
        creative_mode: bool = None,
        **kwargs
    ) -> CreateResult:
        with WebDriverSession(webdriver, "", proxy=proxy) as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Register fetch hook
            source = """
window._fetch = window.fetch;
window.fetch = async (url, options) => {
    const response = await window._fetch(url, options);
    if (url != "/api/infer/answer") {
        return response;
    }
    copy = response.clone();
    window._reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    return copy;
}
"""
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": source
            })

            prompt = quote(format_prompt(messages))
            driver.get(f"{cls.url}/search?q={prompt}&source=searchbox")

            # Need to change settings
            wait = WebDriverWait(driver, timeout)
            def open_dropdown():
                # Open settings dropdown
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.text-dark.dropdown-toggle")))
                driver.find_element(By.CSS_SELECTOR, "button.text-dark.dropdown-toggle").click()
                # Wait for dropdown toggle
                wait.until(EC.visibility_of_element_located((By.XPATH, "//button[text()='GPT-4']")))
            if model.startswith("gpt-4") or creative_mode:
               # Enable GPT-4
                if model.startswith("gpt-4"):
                    open_dropdown()
                    driver.find_element(By.XPATH, "//button[text()='GPT-4']").click()
                # Enable creative mode
                if creative_mode or creative_mode == None:
                    open_dropdown()
                    driver.find_element(By.ID, "Creative Mode").click()
                # Submit changes
                driver.find_element(By.CSS_SELECTOR, ".search-bar-input-group button[type='submit']").click()
                # Wait for page reload
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".search-container")))

            while True:
                chunk = driver.execute_script("""
if(window._reader) {
    chunk = await window._reader.read();
    if (chunk['done']) {
        return null;
    }
    content = '';
    chunk['value'].split('\\r\\n').forEach((line, index) => {
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
""")
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)