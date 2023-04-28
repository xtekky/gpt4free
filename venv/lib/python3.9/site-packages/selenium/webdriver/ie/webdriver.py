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

from selenium.webdriver.common import utils
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver

from .options import Options
from .service import DEFAULT_EXECUTABLE_PATH
from .service import Service

DEFAULT_TIMEOUT = 30
DEFAULT_PORT = 0
DEFAULT_HOST = None
DEFAULT_LOG_LEVEL = None
DEFAULT_SERVICE_LOG_PATH = None
DEFAULT_KEEP_ALIVE = None


class WebDriver(RemoteWebDriver):
    """Controls the IEServerDriver and allows you to drive Internet
    Explorer."""

    def __init__(
        self,
        executable_path=DEFAULT_EXECUTABLE_PATH,
        capabilities=None,
        port=DEFAULT_PORT,
        timeout=DEFAULT_TIMEOUT,
        host=DEFAULT_HOST,
        log_level=DEFAULT_LOG_LEVEL,
        service_log_path=DEFAULT_SERVICE_LOG_PATH,
        options: Options = None,
        service: Service = None,
        desired_capabilities=None,
        keep_alive=DEFAULT_KEEP_ALIVE,
    ) -> None:
        """Creates a new instance of the Ie driver.

        Starts the service and then creates new instance of Ie driver.

        :Args:
         - executable_path - Deprecated: path to the executable. If the default is used it assumes the executable is in the $PATH
         - capabilities - Deprecated: capabilities Dictionary object
         - port - Deprecated: port you would like the service to run, if left as 0, a free port will be found.
         - timeout - Deprecated: no longer used, kept for backward compatibility
         - host - Deprecated: IP address for the service
         - log_level - Deprecated: log level you would like the service to run.
         - service_log_path - Deprecated: target of logging of service, may be "stdout", "stderr" or file path.
         - options - IE Options instance, providing additional IE options
         - desired_capabilities - Deprecated: alias of capabilities; this will make the signature consistent with RemoteWebDriver.
         - keep_alive - Deprecated: Whether to configure RemoteConnection to use HTTP keep-alive.
        """
        if executable_path != "IEDriverServer.exe":
            warnings.warn(
                "executable_path has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        if capabilities:
            warnings.warn(
                "capabilities has been deprecated, please pass in an Options object." "This field will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        if port != DEFAULT_PORT:
            warnings.warn("port has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2)
        if timeout != DEFAULT_TIMEOUT:
            warnings.warn(
                "timeout has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        if host != DEFAULT_HOST:
            warnings.warn("host has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2)
        if log_level != DEFAULT_LOG_LEVEL:
            warnings.warn(
                "log_level has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        if service_log_path != DEFAULT_SERVICE_LOG_PATH:
            warnings.warn(
                "service_log_path has been deprecated, please pass in a Service object",
                DeprecationWarning,
                stacklevel=2,
            )
        if desired_capabilities:
            warnings.warn(
                "desired_capabilities has been deprecated, please pass in an Options object."
                "This field will be ignored",
                DeprecationWarning,
                stacklevel=2,
            )
        if keep_alive != DEFAULT_KEEP_ALIVE:
            warnings.warn(
                "keep_alive has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        else:
            keep_alive = True

        self.host = host
        self.port = port
        if self.port == 0:
            self.port = utils.free_port()

        if not options:
            options = self.create_options()

        if service:
            self.iedriver = service
        else:
            self.iedriver = Service(
                executable_path, port=self.port, host=self.host, log_level=log_level, log_file=service_log_path
            )

        self.iedriver.path = DriverFinder.get_path(self.iedriver, options)
        self.iedriver.start()

        super().__init__(command_executor=self.iedriver.service_url, options=options, keep_alive=keep_alive)
        self._is_remote = False

    def quit(self) -> None:
        super().quit()
        self.iedriver.stop()

    def create_options(self) -> Options:
        return Options()
