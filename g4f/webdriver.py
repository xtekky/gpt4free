from __future__ import annotations

from platformdirs import user_config_dir
from selenium.webdriver.remote.webdriver import WebDriver 
from undetected_chromedriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from os import path
from . import debug

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
    if user_data_dir and debug.logging:
        print("Open browser with config dir:", user_data_dir)
    if not options:
        options = ChromeOptions()
    if proxy:
        options.add_argument(f'--proxy-server={proxy}')
    driver = '/usr/bin/chromedriver'
    if not path.isfile(driver):
        driver = None
    return Chrome(
        options=options,
        user_data_dir=user_data_dir,
        driver_executable_path=driver,
        headless=headless
    )

def bypass_cloudflare(driver: WebDriver, url: str, timeout: int) -> None:
    # Open website
    driver.get(url)
    # Is cloudflare protection
    if driver.find_element(By.TAG_NAME, "body").get_attribute("class") == "no-js":
        if debug.logging:
            print("Cloudflare protection detected:", url)
        try:
            # Click button in iframe
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#turnstile-wrapper iframe"))
            )
            driver.switch_to.frame(driver.find_element(By.CSS_SELECTOR, "#turnstile-wrapper iframe"))
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#challenge-stage input"))
            )
            driver.find_element(By.CSS_SELECTOR, "#challenge-stage input").click()
        except:
            pass
        finally:
            driver.switch_to.default_content()
    # No cloudflare protection
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "body:not(.no-js)"))
    )

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
            self.virtual_display = Display(size=(1920, 1080))
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
            self.default_driver.quit()
        if self.virtual_display:
            self.virtual_display.stop()