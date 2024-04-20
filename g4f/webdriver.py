from __future__ import annotations

try:
    from platformdirs import user_config_dir
    from undetected_chromedriver import Chrome, ChromeOptions, find_chrome_executable
    from selenium.webdriver.remote.webdriver import WebDriver 
    from selenium.webdriver.remote.webelement import WebElement 
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import NoSuchElementException
    has_requirements = True
except ImportError:
    from typing import Type as WebDriver
    has_requirements = False

import time
from shutil import which
from os import path
from os import access, R_OK
from .typing import Cookies
from .errors import MissingRequirementsError
from . import debug

try:
    from pyvirtualdisplay import Display
    has_pyvirtualdisplay = True
except ImportError:
    has_pyvirtualdisplay = False

try:
    from undetected_chromedriver import Chrome as _Chrome, ChromeOptions
    from seleniumwire.webdriver import InspectRequestsMixin, DriverCommonMixin

    class Chrome(InspectRequestsMixin, DriverCommonMixin, _Chrome):
        def __init__(self, *args, options=None, seleniumwire_options={}, **kwargs):
            if options is None:
                options = ChromeOptions()
            config = self._setup_backend(seleniumwire_options)
            options.add_argument(f"--proxy-server={config['proxy']['httpProxy']}")
            options.add_argument("--proxy-bypass-list=<-loopback>")
            options.add_argument("--ignore-certificate-errors")
            super().__init__(*args, options=options, **kwargs)
    has_seleniumwire = True
except:
    has_seleniumwire = False

def get_browser(
    user_data_dir: str = None,
    headless: bool = False,
    proxy: str = None,
    options: ChromeOptions = None
) -> WebDriver:
    """
    Creates and returns a Chrome WebDriver with specified options.

    Args:
        user_data_dir (str, optional): Directory for user data. If None, uses default directory.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to False.
        proxy (str, optional): Proxy settings for the browser. Defaults to None.
        options (ChromeOptions, optional): ChromeOptions object with specific browser options. Defaults to None.

    Returns:
        WebDriver: An instance of WebDriver configured with the specified options.
    """
    if not has_requirements:
        raise MissingRequirementsError('Install Webdriver packages | pip install -U g4f[webdriver]')
    browser = find_chrome_executable()
    if browser is None:
        raise MissingRequirementsError('Install "Google Chrome" browser')
    if user_data_dir is None:
        user_data_dir = user_config_dir("g4f")
    if user_data_dir and debug.logging:
        print("Open browser with config dir:", user_data_dir)
    if not options:
        options = ChromeOptions()
    if proxy:
        options.add_argument(f'--proxy-server={proxy}')
    # Check for system driver in docker
    driver = which('chromedriver') or '/usr/bin/chromedriver'
    if not path.isfile(driver) or not access(driver, R_OK):
        driver = None
    return Chrome(
        options=options,
        user_data_dir=user_data_dir,
        driver_executable_path=driver,
        browser_executable_path=browser,
        headless=headless,
        patcher_force_close=True
    )

def get_driver_cookies(driver: WebDriver) -> Cookies:
    """
    Retrieves cookies from the specified WebDriver.

    Args:
        driver (WebDriver): The WebDriver instance from which to retrieve cookies.

    Returns:
        dict: A dictionary containing cookies with their names as keys and values as cookie values.
    """
    return {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}

def bypass_cloudflare(driver: WebDriver, url: str, timeout: int) -> None:
    """
    Attempts to bypass Cloudflare protection when accessing a URL using the provided WebDriver.

    Args:
        driver (WebDriver): The WebDriver to use for accessing the URL.
        url (str): The URL to access.
        timeout (int): Time in seconds to wait for the page to load.

    Raises:
        Exception: If there is an error while bypassing Cloudflare or loading the page.
    """
    driver.get(url)
    if driver.find_element(By.TAG_NAME, "body").get_attribute("class") == "no-js":
        if debug.logging:
            print("Cloudflare protection detected:", url)

        # Open website in a new tab
        element = driver.find_element(By.ID, "challenge-body-text")
        driver.execute_script(f"""
            arguments[0].addEventListener('click', () => {{
                window.open(arguments[1]);
            }});
        """, element, url)
        element.click()
        time.sleep(5)

        # Switch to the new tab and close the old tab
        original_window = driver.current_window_handle
        for window_handle in driver.window_handles:
            if window_handle != original_window:
                driver.close()
                driver.switch_to.window(window_handle)
                break

        # Click on the challenge button in the iframe
        try:
            driver.switch_to.frame(driver.find_element(By.CSS_SELECTOR, "#turnstile-wrapper iframe"))
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#challenge-stage input"))
            ).click()
        except NoSuchElementException:
            ...
        except Exception as e:
            if debug.logging:
                print(f"Error bypassing Cloudflare: {str(e).splitlines()[0]}")
        #driver.switch_to.default_content()
        driver.switch_to.window(window_handle)
        driver.execute_script("document.href = document.href;")
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
        """
        Initializes a new instance of the WebDriverSession.

        Args:
            webdriver (WebDriver, optional): A WebDriver instance for the session. Defaults to None.
            user_data_dir (str, optional): Directory for user data. Defaults to None.
            headless (bool, optional): Whether to run the browser in headless mode. Defaults to False.
            virtual_display (bool, optional): Whether to use a virtual display. Defaults to False.
            proxy (str, optional): Proxy settings for the browser. Defaults to None.
            options (ChromeOptions, optional): ChromeOptions for the browser. Defaults to None.
        """
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
        Reopens the WebDriver session with new settings.

        Args:
            user_data_dir (str, optional): Directory for user data. Defaults to current value.
            headless (bool, optional): Whether to run the browser in headless mode. Defaults to current value.
            virtual_display (bool, optional): Whether to use a virtual display. Defaults to current value.

        Returns:
            WebDriver: The reopened WebDriver instance.
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
        Context management method for entering a session. Initializes and returns a WebDriver instance.

        Returns:
            WebDriver: An instance of WebDriver for this session.
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

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.

        Note:
            Closes the WebDriver and stops the virtual display if used.
        """
        if self.default_driver:
            try:
                self.default_driver.close()
            except Exception as e:
                if debug.logging:
                    print(f"Error closing WebDriver: {str(e).splitlines()[0]}")
            finally:
                self.default_driver.quit()
        if self.virtual_display:
            self.virtual_display.stop()  
  
def element_send_text(element: WebElement, text: str) -> None:
    script = "arguments[0].innerText = arguments[1];"
    element.parent.execute_script(script, element, text)
    element.send_keys(Keys.ENTER)