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
    """
    Creates and returns a Chrome WebDriver with the specified options.

    :param user_data_dir: Directory for user data. If None, uses default directory.
    :param headless: Boolean indicating whether to run the browser in headless mode.
    :param proxy: Proxy settings for the browser.
    :param options: ChromeOptions object with specific browser options.
    :return: An instance of WebDriver.
    """
    if user_data_dir is None:
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

def get_driver_cookies(driver: WebDriver) -> dict:
    """
    Retrieves cookies from the given WebDriver.

    :param driver: WebDriver from which to retrieve cookies.
    :return: A dictionary of cookies.
    """
    return {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}

def bypass_cloudflare(driver: WebDriver, url: str, timeout: int) -> None:
    """
    Attempts to bypass Cloudflare protection when accessing a URL using the provided WebDriver.

    :param driver: The WebDriver to use.
    :param url: URL to access.
    :param timeout: Time in seconds to wait for the page to load.
    """
    driver.get(url)
    if driver.find_element(By.TAG_NAME, "body").get_attribute("class") == "no-js":
        if debug.logging:
            print("Cloudflare protection detected:", url)
        try:
            driver.switch_to.frame(driver.find_element(By.CSS_SELECTOR, "#turnstile-wrapper iframe"))
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#challenge-stage input"))
            ).click()
        except Exception as e:
            if debug.logging:
                print(f"Error bypassing Cloudflare: {e}")
        finally:
            driver.switch_to.default_content()
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "body:not(.no-js)"))
    )

class WebDriverSession:
    """
    Manages a Selenium WebDriver session, including handling of virtual displays and proxies.
    """
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
        self.virtual_display = Display(size=(1920, 1080)) if has_pyvirtualdisplay and virtual_display else None
        self.proxy = proxy
        self.options = options
        self.default_driver = None
    
    def reopen(
        self,
        user_data_dir: str = None,
        headless: bool = False,
        virtual_display: bool = False
    ) -> WebDriver:
        """
        Reopens the WebDriver session with the specified parameters.

        :param user_data_dir: Directory for user data.
        :param headless: Boolean indicating whether to run the browser in headless mode.
        :param virtual_display: Boolean indicating whether to use a virtual display.
        :return: An instance of WebDriver.
        """
        user_data_dir = user_data_dir or self.user_data_dir
        if self.default_driver:
            self.default_driver.quit()
        if not virtual_display and self.virtual_display:
            self.virtual_display.stop()
            self.virtual_display = None
        self.default_driver = get_browser(user_data_dir, headless, self.proxy)
        return self.default_driver

    def __enter__(self) -> WebDriver:
        """
        Context management method for entering a session.
        :return: An instance of WebDriver.
        """
        if self.webdriver:
            return self.webdriver
        if self.virtual_display:
            self.virtual_display.start()
        self.default_driver = get_browser(self.user_data_dir, self.headless, self.proxy, self.options)
        return self.default_driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management method for exiting a session. Closes and quits the WebDriver.
        """
        if self.default_driver:
            try:
                self.default_driver.close()
            except Exception as e:
                if debug.logging:
                    print(f"Error closing WebDriver: {e}")
            self.default_driver.quit()
        if self.virtual_display:
            self.virtual_display.stop()