from __future__ import annotations

import time
from urllib.parse import quote

from ..typing import CreateResult, Messages
from .base_provider import BaseProvider
from .helper import WebDriver, format_prompt, get_browser

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

        prompt = quote(format_prompt(messages))
        driver.get(f"{cls.url}/search?q={prompt}&source=searchbox")

        # Need to change settinge
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

        try:
            # Add fetch hook
            script = """
window._fetch = window.fetch;
window.fetch = (url, options) => {
    // Call parent fetch method
    const result = window._fetch(url, options);
    if (url != "/api/infer/answer") return result;
    // Load response reader
    result.then((response) => {
        if (!response.body.locked) {
            window.reader = response.body.getReader();
        }
    });
    // Return dummy response
    return new Promise((resolve, reject) => {
        resolve(new Response(new ReadableStream()))
    });
}
"""
            # Read response from reader
            driver.execute_script(script)
            script = """
if(window.reader) {
    chunk = await window.reader.read();
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
            if hidden_display:
                display.stop()