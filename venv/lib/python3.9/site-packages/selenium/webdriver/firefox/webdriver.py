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
import base64
import logging
import os
import warnings
import zipfile
from contextlib import contextmanager
from io import BytesIO
from shutil import rmtree

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver

from .firefox_binary import FirefoxBinary
from .firefox_profile import FirefoxProfile
from .options import Options
from .remote_connection import FirefoxRemoteConnection
from .service import DEFAULT_EXECUTABLE_PATH
from .service import Service

logger = logging.getLogger(__name__)

# Default for log_path variable. To be deleted when deprecations for arguments are removed.
DEFAULT_LOG_PATH = None
DEFAULT_SERVICE_LOG_PATH = "geckodriver.log"


class WebDriver(RemoteWebDriver):
    CONTEXT_CHROME = "chrome"
    CONTEXT_CONTENT = "content"

    def __init__(
        self,
        firefox_profile=None,
        firefox_binary=None,
        capabilities=None,
        proxy=None,
        executable_path=DEFAULT_EXECUTABLE_PATH,
        options=None,
        service_log_path=DEFAULT_SERVICE_LOG_PATH,
        service_args=None,
        service=None,
        desired_capabilities=None,
        log_path=DEFAULT_LOG_PATH,
        keep_alive=True,  # Todo: Why is this now unused?
    ) -> None:
        """Starts a new local session of Firefox.

        Based on the combination and specificity of the various keyword
        arguments, a capabilities dictionary will be constructed that
        is passed to the remote end.

        The keyword arguments given to this constructor are helpers to
        more easily allow Firefox WebDriver sessions to be customised
        with different options.  They are mapped on to a capabilities
        dictionary that is passed on to the remote end.

        As some of the options, such as `firefox_profile` and
        `options.profile` are mutually exclusive, precedence is
        given from how specific the setting is.  `capabilities` is the
        least specific keyword argument, followed by `options`,
        followed by `firefox_binary` and `firefox_profile`.

        In practice this means that if `firefox_profile` and
        `options.profile` are both set, the selected profile
        instance will always come from the most specific variable.
        In this case that would be `firefox_profile`.  This will result in
        `options.profile` to be ignored because it is considered
        a less specific setting than the top-level `firefox_profile`
        keyword argument.  Similarly, if you had specified a
        `capabilities["moz:firefoxOptions"]["profile"]` Base64 string,
        this would rank below `options.profile`.

        :param firefox_profile: Deprecated: Instance of ``FirefoxProfile`` object
            or a string.  If undefined, a fresh profile will be created
            in a temporary location on the system.
        :param firefox_binary: Deprecated: Instance of ``FirefoxBinary`` or full
            path to the Firefox binary.  If undefined, the system default
            Firefox installation will  be used.
        :param capabilities: Deprecated: Dictionary of desired capabilities.
        :param proxy: Deprecated: The proxy settings to use when communicating with
            Firefox via the extension connection.
        :param executable_path: Deprecated: Full path to override which geckodriver
            binary to use for Firefox 47.0.1 and greater, which
            defaults to picking up the binary from the system path.
        :param options: Instance of ``options.Options``.
        :param service: (Optional) service instance for managing the starting and stopping of the driver.
        :param service_log_path: Deprecated: Where to log information from the driver.
        :param service_args: Deprecated: List of args to pass to the driver service
        :param desired_capabilities: Deprecated: alias of capabilities. In future
            versions of this library, this will replace 'capabilities'.
            This will make the signature consistent with RemoteWebDriver.
        :param keep_alive: Whether to configure remote_connection.RemoteConnection to use
             HTTP keep-alive.
        """

        if executable_path != DEFAULT_EXECUTABLE_PATH:
            warnings.warn(
                "executable_path has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        if capabilities or desired_capabilities:
            warnings.warn(
                "capabilities and desired_capabilities have been deprecated, please pass in a Service object",
                DeprecationWarning,
                stacklevel=2,
            )
        if firefox_binary:
            warnings.warn(
                "firefox_binary has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )
        self.binary = None
        if firefox_profile:
            warnings.warn(
                "firefox_profile has been deprecated, please pass in an Options object",
                DeprecationWarning,
                stacklevel=2,
            )
        self.profile = None

        if log_path != DEFAULT_LOG_PATH:
            warnings.warn(
                "log_path has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )

        # Service Arguments being deprecated.
        if service_log_path != DEFAULT_SERVICE_LOG_PATH:
            warnings.warn(
                "service_log_path has been deprecated, please pass in a Service object",
                DeprecationWarning,
                stacklevel=2,
            )
        if service_args:
            warnings.warn(
                "service_args has been deprecated, please pass in a Service object", DeprecationWarning, stacklevel=2
            )

        self.service = service

        # If desired capabilities is set, alias it to capabilities.
        # If both are set ignore desired capabilities.
        if not capabilities and desired_capabilities:
            capabilities = desired_capabilities

        if not capabilities:
            capabilities = DesiredCapabilities.FIREFOX.copy()
        if not options:
            options = Options()

        capabilities = dict(capabilities)

        if capabilities.get("binary"):
            options.binary = capabilities["binary"]

        # options overrides capabilities
        if options:
            if options.binary:
                self.binary = options.binary
            if options.profile:
                self.profile = options.profile

        # firefox_binary and firefox_profile
        # override options
        if firefox_binary:
            if isinstance(firefox_binary, str):
                firefox_binary = FirefoxBinary(firefox_binary)
            self.binary = firefox_binary
            options.binary = firefox_binary
        if firefox_profile:
            if isinstance(firefox_profile, str):
                firefox_profile = FirefoxProfile(firefox_profile)
            self.profile = firefox_profile
            options.profile = firefox_profile

        if not capabilities.get("acceptInsecureCerts") or not options.accept_insecure_certs:
            options.accept_insecure_certs = False

        if not self.service:
            self.service = Service(executable_path, service_args=service_args, log_path=service_log_path)
        self.service.path = DriverFinder.get_path(self.service, options)
        self.service.start()

        executor = FirefoxRemoteConnection(
            remote_server_addr=self.service.service_url, ignore_proxy=options._ignore_local_proxy
        )
        super().__init__(command_executor=executor, options=options, keep_alive=True)

        self._is_remote = False

    def quit(self) -> None:
        """Quits the driver and close every associated window."""
        try:
            super().quit()
        except Exception:
            # We don't care about the message because something probably has gone wrong
            pass

        self.service.stop()

        if self.profile:
            try:
                rmtree(self.profile.path)
                if self.profile.tempfolder:
                    rmtree(self.profile.tempfolder)
            except Exception:
                logger.exception("Unable to remove profile specific paths.")

        self._close_binary_file_handle()

    def _close_binary_file_handle(self) -> None:
        """Attempts to close the underlying file handles for `FirefoxBinary`
        instances if they are used and open.

        To keep inline with other cleanup raising here is swallowed and
        will not cause a runtime error.
        """
        try:
            if isinstance(self.binary, FirefoxBinary):
                if hasattr(self.binary._log_file, "close"):
                    self.binary._log_file.close()
        except Exception:
            logger.exception("Unable to close open file handle for firefox binary log file.")

    @property
    def firefox_profile(self):
        return self.profile

    # Extension commands:

    def set_context(self, context) -> None:
        self.execute("SET_CONTEXT", {"context": context})

    @contextmanager
    def context(self, context):
        """Sets the context that Selenium commands are running in using a
        `with` statement. The state of the context on the server is saved
        before entering the block, and restored upon exiting it.

        :param context: Context, may be one of the class properties
            `CONTEXT_CHROME` or `CONTEXT_CONTENT`.

        Usage example::

            with selenium.context(selenium.CONTEXT_CHROME):
                # chrome scope
                ... do stuff ...
        """
        initial_context = self.execute("GET_CONTEXT").pop("value")
        self.set_context(context)
        try:
            yield
        finally:
            self.set_context(initial_context)

    def install_addon(self, path, temporary=False) -> str:
        """Installs Firefox addon.

        Returns identifier of installed addon. This identifier can later
        be used to uninstall addon.

        :param path: Absolute path to the addon that will be installed.

        :Usage:
            ::

                driver.install_addon('/path/to/firebug.xpi')
        """

        if os.path.isdir(path):
            fp = BytesIO()
            path_root = len(path) + 1  # account for trailing slash
            with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zipped:
                for base, dirs, files in os.walk(path):
                    for fyle in files:
                        filename = os.path.join(base, fyle)
                        zipped.write(filename, filename[path_root:])
            addon = base64.b64encode(fp.getvalue()).decode("UTF-8")
        else:
            with open(path, "rb") as file:
                addon = base64.b64encode(file.read()).decode("UTF-8")

        payload = {"addon": addon, "temporary": temporary}
        return self.execute("INSTALL_ADDON", payload)["value"]

    def uninstall_addon(self, identifier) -> None:
        """Uninstalls Firefox addon using its identifier.

        :Usage:
            ::

                driver.uninstall_addon('addon@foo.com')
        """
        self.execute("UNINSTALL_ADDON", {"id": identifier})

    def get_full_page_screenshot_as_file(self, filename) -> bool:
        """Saves a full document screenshot of the current window to a PNG
        image file. Returns False if there is any IOError, else returns True.
        Use full paths in your filename.

        :Args:
         - filename: The full path you wish to save your screenshot to. This
           should end with a `.png` extension.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_file('/Screenshots/foo.png')
        """
        if not filename.lower().endswith(".png"):
            warnings.warn(
                "name used for saved screenshot does not match file " "type. It should end with a `.png` extension",
                UserWarning,
            )
        png = self.get_full_page_screenshot_as_png()
        try:
            with open(filename, "wb") as f:
                f.write(png)
        except OSError:
            return False
        finally:
            del png
        return True

    def save_full_page_screenshot(self, filename) -> bool:
        """Saves a full document screenshot of the current window to a PNG
        image file. Returns False if there is any IOError, else returns True.
        Use full paths in your filename.

        :Args:
         - filename: The full path you wish to save your screenshot to. This
           should end with a `.png` extension.

        :Usage:
            ::

                driver.save_full_page_screenshot('/Screenshots/foo.png')
        """
        return self.get_full_page_screenshot_as_file(filename)

    def get_full_page_screenshot_as_png(self) -> bytes:
        """Gets the full document screenshot of the current window as a binary
        data.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_png()
        """
        return base64.b64decode(self.get_full_page_screenshot_as_base64().encode("ascii"))

    def get_full_page_screenshot_as_base64(self) -> str:
        """Gets the full document screenshot of the current window as a base64
        encoded string which is useful in embedded images in HTML.

        :Usage:
            ::

                driver.get_full_page_screenshot_as_base64()
        """
        return self.execute("FULL_PAGE_SCREENSHOT")["value"]
