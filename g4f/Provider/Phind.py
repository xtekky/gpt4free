from __future__ import annotations

import time
from urllib.parse import quote

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import WebDriver, WebDriverSession, format_prompt

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
        web_driver: WebDriver = None,
        creative_mode: bool = None,
        **kwargs
    ) -> CreateResult:
        with WebDriverSession(web_driver, "", proxy=proxy) as driver:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            prompt = quote(format_prompt(messages))
            driver.get(f"{cls.url}/search?q={prompt}&source=searchbox")

            # Register fetch hook
            driver.execute_script("""
window._fetch = window.fetch;
window.fetch = (url, options) => {
    // Call parent fetch method
    const result = window._fetch(url, options);
    if (url != "/api/infer/answer") {
        return result;
    }
    // Load response reader
    result.then((response) => {
        if (!response.body.locked) {
            window._reader = response.body.getReader();
        }
    });
    // Return dummy response
    return new Promise((resolve, reject) => {
        resolve(new Response(new ReadableStream()))
    });
}
""")

            # Need to change settings
            if model.startswith("gpt-4") or creative_mode:
                wait = WebDriverWait(driver, timeout)
                # Open settings dropdown
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.text-dark.dropdown-toggle")))
                driver.find_element(By.CSS_SELECTOR, "button.text-dark.dropdown-toggle").click()
                # Wait for dropdown toggle
                wait.until(EC.visibility_of_element_located((By.XPATH, "//button[text()='GPT-4']")))
                # Enable GPT-4
                if model.startswith("gpt-4"):
                    driver.find_element(By.XPATH, "//button[text()='GPT-4']").click()
                # Enable creative mode
                if creative_mode or creative_mode == None:
                    driver.find_element(By.ID, "Creative Mode").click()
                # Submit changes
                driver.find_element(By.CSS_SELECTOR, ".search-bar-input-group button[type='submit']").click()
                # Wait for page reload
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".search-container")))

            while True:
                chunk = driver.execute_script("""
if(window._reader) {
    chunk = await window._reader.read();
    if (chunk['done']) return null;
    text = (new TextDecoder()).decode(chunk['value']);
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
""")
                if chunk:
                    yield chunk
                elif chunk != "":
                    break
                else:
                    time.sleep(0.1)