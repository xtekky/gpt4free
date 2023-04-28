# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import warnings

from selenium.webdriver.chromium.webdriver import ChromiumDriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.driver_finder import DriverFinder

from .options import Options
from .service import DEFAULT_EXECUTABLE_PATH
from .service import Service

DEFAULT_PORT = 0
DEFAULT_SERVICE_LOG_PATH = None
DEFAULT_KEEP_ALIVE = None


class WebDriver(ChromiumDriver):
    """Controls the ChromeDriver and allows you to drive the browser.

    You will need to download the ChromeDriver executable from
    http://chromedriver.storage.googleapis.com/index.html
    """

    def __init__(
        self,
        executable_path=DEFAULT_EXECUTABLE_PATH,
        port=DEFAULT_PORT,
        options: Options = None,
        service_args=None,
        desired_capabilities=None,
        service_log_path=DEFAULT_SERVICE_LOG_PATH,
        chrome_options=None,
        service: Service = None,
        keep_alive=DEFAULT_KEEP_ALIVE,
    ) -> None:
        """Creates a new instance of the chrome driver. Starts the service and
        then creates new instance of chrome driver.

        :Args:
         - executable_path - Deprecated: path to the executable. If the default is used it assumes the executable is in the $PATH
         - port - Deprecated: port you would like the service to run, if left as 0, a free port will be found.
         - options - this takes an instance of ChromeOptions
         - service - Service object for handling the browser driver if you need to pass extra details
         - service_args - Deprecated: List of args to pass to the driver service
         - desired_capabilities - Deprecated: Dictionary object with non-browser specific
           capabilities only, such as "proxy" or "loggingPref".
         - service_log_path - Deprecated: Where to log information from the driver.
         - keep_alive - Deprecated: Whether to configure ChromeRemoteConnection to use HTTP keep-alive.
        """
        if executable_path != "chromedriver":
            warnings.warn(
                "executable_path has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        if chrome_options:
            warnings.warn("use options instead of chrome_options", DeprecationWarning, stacklevel=2)
            options = chrome_options
        if keep_alive != DEFAULT_KEEP_ALIVE:
            warnings.warn(
                "keep_alive has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        else:
            keep_alive = True
        if not options:
            options = self.create_options()
        if not service:
            service = Service(executable_path, port, service_args, service_log_path)
        service.path = DriverFinder.get_path(service, options)

        super().__init__(
            DesiredCapabilities.CHROME["browserName"],
            "goog",
            port,
            options,
            service_args,
            desired_capabilities,
            service_log_path,
            service,
            keep_alive,
        )

    def create_options(self) -> Options:
        return Options()
