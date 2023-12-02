from __future__ import annotations

import time
from platformdirs import user_config_dir
from selenium.webdriver.remote.webdriver import WebDriver 
from undetected_chromedriver import Chrome, ChromeOptions

try:
    from pyvirtualdisplay import Display
    has_pyvirtualdisplay = True
except ImportError:
    has_pyvirtualdisplay = False

def get_browser(
    user_data_dir: str = None,
    headless: bool = False,
    proxy: str = None,
    options: ChromeOptions = None
) -> WebDriver:
    if user_data_dir == None:
        user_data_dir = user_config_dir("g4f")
    if not options:
        options = ChromeOptions()
    options.add_argument("window-size=1920,1080");
    if proxy:
        options.add_argument(f'--proxy-server={proxy}')
    return Chrome(options=options, user_data_dir=user_data_dir, headless=headless)

class WebDriverSession():
    def __init__(
        self,
        webdriver: WebDriver = None,
        user_data_dir: str = None,
        headless: bool = False,
        virtual_display: bool = False,
        proxy: str = None,
        options: ChromeOptions = None
    ):
        self.webdriver = webdriver
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.virtual_display = None
        if has_pyvirtualdisplay and virtual_display:
            self.virtual_display = Display(size=(1920,1080))
        self.proxy = proxy
        self.options = options
        self.default_driver = None
    
    def reopen(
        self,
        user_data_dir: str = None,
        headless: bool = False,
        virtual_display: bool = False
    ) -> WebDriver:
        if user_data_dir == None:
            user_data_dir = self.user_data_dir
        if self.default_driver:
            self.default_driver.quit()
        if not virtual_display and self.virtual_display:
            self.virtual_display.stop()
            self.virtual_display = None
        self.default_driver = get_browser(user_data_dir, headless, self.proxy)
        return self.default_driver

    def __enter__(self) -> WebDriver:
        if self.webdriver:
            return self.webdriver
        if self.virtual_display:
            self.virtual_display.start()
        self.default_driver = get_browser(self.user_data_dir, self.headless, self.proxy, self.options)
        return self.default_driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_driver:
            try:
                self.default_driver.close()
            except:
                pass
            time.sleep(0.1)
            self.default_driver.quit()
        if self.virtual_display:
            self.virtual_display.stop()