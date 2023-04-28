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
from enum import Enum

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions


class ElementScrollBehavior(Enum):
    TOP = 0
    BOTTOM = 1


class Options(ArgOptions):
    KEY = "se:ieOptions"
    SWITCHES = "ie.browserCommandLineSwitches"

    BROWSER_ATTACH_TIMEOUT = "browserAttachTimeout"
    ELEMENT_SCROLL_BEHAVIOR = "elementScrollBehavior"
    ENSURE_CLEAN_SESSION = "ie.ensureCleanSession"
    FILE_UPLOAD_DIALOG_TIMEOUT = "ie.fileUploadDialogTimeout"
    FORCE_CREATE_PROCESS_API = "ie.forceCreateProcessApi"
    FORCE_SHELL_WINDOWS_API = "ie.forceShellWindowsApi"
    FULL_PAGE_SCREENSHOT = "ie.enableFullPageScreenshot"
    IGNORE_PROTECTED_MODE_SETTINGS = "ignoreProtectedModeSettings"
    IGNORE_ZOOM_LEVEL = "ignoreZoomSetting"
    INITIAL_BROWSER_URL = "initialBrowserUrl"
    NATIVE_EVENTS = "nativeEvents"
    PERSISTENT_HOVER = "enablePersistentHover"
    REQUIRE_WINDOW_FOCUS = "requireWindowFocus"
    USE_PER_PROCESS_PROXY = "ie.usePerProcessProxy"
    USE_LEGACY_FILE_UPLOAD_DIALOG_HANDLING = "ie.useLegacyFileUploadDialogHandling"
    ATTACH_TO_EDGE_CHROME = "ie.edgechromium"
    EDGE_EXECUTABLE_PATH = "ie.edgepath"

    def __init__(self) -> None:
        super().__init__()
        self._options = {}
        self._additional = {}

    @property
    def options(self) -> dict:
        """:Returns: A dictionary of browser options"""
        return self._options

    @property
    def browser_attach_timeout(self) -> int:
        """
        :Returns: The options Browser Attach Timeout in milliseconds
        """
        return self._options.get(self.BROWSER_ATTACH_TIMEOUT)

    @browser_attach_timeout.setter
    def browser_attach_timeout(self, value: int) -> None:
        """Sets the options Browser Attach Timeout.

        :Args:
         - value: Timeout in milliseconds
        """
        if not isinstance(value, int):
            raise ValueError("Browser Attach Timeout must be an integer.")
        self._options[self.BROWSER_ATTACH_TIMEOUT] = value

    @property
    def element_scroll_behavior(self) -> ElementScrollBehavior:
        """:Returns: The options Element Scroll Behavior value"""
        return self._options.get(self.ELEMENT_SCROLL_BEHAVIOR)

    @element_scroll_behavior.setter
    def element_scroll_behavior(self, value: ElementScrollBehavior) -> None:
        """Sets the options Element Scroll Behavior.

        :Args:
         - value: 0 - Top, 1 - Bottom
        """
        if value not in [ElementScrollBehavior.TOP, ElementScrollBehavior.BOTTOM]:
            raise ValueError("Element Scroll Behavior out of range.")
        self._options[self.ELEMENT_SCROLL_BEHAVIOR] = value

    @property
    def ensure_clean_session(self) -> bool:
        """:Returns: The options Ensure Clean Session value"""
        return self._options.get(self.ENSURE_CLEAN_SESSION)

    @ensure_clean_session.setter
    def ensure_clean_session(self, value: bool) -> None:
        """Sets the options Ensure Clean Session value.

        :Args:
         - value: boolean value
        """
        self._options[self.ENSURE_CLEAN_SESSION] = value

    @property
    def file_upload_dialog_timeout(self) -> int:
        """:Returns: The options File Upload Dialog Timeout in milliseconds"""
        return self._options.get(self.FILE_UPLOAD_DIALOG_TIMEOUT)

    @file_upload_dialog_timeout.setter
    def file_upload_dialog_timeout(self, value: int) -> None:
        """Sets the options File Upload Dialog Timeout value.

        :Args:
         - value: Timeout in milliseconds
        """
        if not isinstance(value, int):
            raise ValueError("File Upload Dialog Timeout must be an integer.")
        self._options[self.FILE_UPLOAD_DIALOG_TIMEOUT] = value

    @property
    def force_create_process_api(self) -> bool:
        """:Returns: The options Force Create Process Api value"""
        return self._options.get(self.FORCE_CREATE_PROCESS_API)

    @force_create_process_api.setter
    def force_create_process_api(self, value: bool) -> None:
        """Sets the options Force Create Process Api value.

        :Args:
         - value: boolean value
        """
        self._options[self.FORCE_CREATE_PROCESS_API] = value

    @property
    def force_shell_windows_api(self) -> bool:
        """:Returns: The options Force Shell Windows Api value"""
        return self._options.get(self.FORCE_SHELL_WINDOWS_API)

    @force_shell_windows_api.setter
    def force_shell_windows_api(self, value: bool) -> None:
        """Sets the options Force Shell Windows Api value.

        :Args:
         - value: boolean value
        """
        self._options[self.FORCE_SHELL_WINDOWS_API] = value

    @property
    def full_page_screenshot(self) -> bool:
        """:Returns: The options Full Page Screenshot value"""
        return self._options.get(self.FULL_PAGE_SCREENSHOT)

    @full_page_screenshot.setter
    def full_page_screenshot(self, value: bool) -> None:
        """Sets the options Full Page Screenshot value.

        :Args:
         - value: boolean value
        """
        self._options[self.FULL_PAGE_SCREENSHOT] = value

    @property
    def ignore_protected_mode_settings(self) -> bool:
        """:Returns: The options Ignore Protected Mode Settings value"""
        return self._options.get(self.IGNORE_PROTECTED_MODE_SETTINGS)

    @ignore_protected_mode_settings.setter
    def ignore_protected_mode_settings(self, value: bool) -> None:
        """Sets the options Ignore Protected Mode Settings value.

        :Args:
         - value: boolean value
        """
        self._options[self.IGNORE_PROTECTED_MODE_SETTINGS] = value

    @property
    def ignore_zoom_level(self) -> bool:
        """:Returns: The options Ignore Zoom Level value"""
        return self._options.get(self.IGNORE_ZOOM_LEVEL)

    @ignore_zoom_level.setter
    def ignore_zoom_level(self, value: bool) -> None:
        """Sets the options Ignore Zoom Level value.

        :Args:
         - value: boolean value
        """
        self._options[self.IGNORE_ZOOM_LEVEL] = value

    @property
    def initial_browser_url(self) -> str:
        """:Returns: The options Initial Browser Url value"""
        return self._options.get(self.INITIAL_BROWSER_URL)

    @initial_browser_url.setter
    def initial_browser_url(self, value: str) -> None:
        """Sets the options Initial Browser Url value.

        :Args:
         - value: URL string
        """
        self._options[self.INITIAL_BROWSER_URL] = value

    @property
    def native_events(self) -> bool:
        """:Returns: The options Native Events value"""
        return self._options.get(self.NATIVE_EVENTS)

    @native_events.setter
    def native_events(self, value: bool) -> None:
        """Sets the options Native Events value.

        :Args:
         - value: boolean value
        """
        self._options[self.NATIVE_EVENTS] = value

    @property
    def persistent_hover(self) -> bool:
        """:Returns: The options Persistent Hover value"""
        return self._options.get(self.PERSISTENT_HOVER)

    @persistent_hover.setter
    def persistent_hover(self, value: bool) -> None:
        """Sets the options Persistent Hover value.

        :Args:
         - value: boolean value
        """
        self._options[self.PERSISTENT_HOVER] = value

    @property
    def require_window_focus(self: bool):
        """:Returns: The options Require Window Focus value"""
        return self._options.get(self.REQUIRE_WINDOW_FOCUS)

    @require_window_focus.setter
    def require_window_focus(self, value: bool) -> None:
        """Sets the options Require Window Focus value.

        :Args:
         - value: boolean value
        """
        self._options[self.REQUIRE_WINDOW_FOCUS] = value

    @property
    def use_per_process_proxy(self) -> bool:
        """:Returns: The options User Per Process Proxy value"""
        return self._options.get(self.USE_PER_PROCESS_PROXY)

    @use_per_process_proxy.setter
    def use_per_process_proxy(self, value: bool) -> None:
        """Sets the options User Per Process Proxy value.

        :Args:
         - value: boolean value
        """
        self._options[self.USE_PER_PROCESS_PROXY] = value

    @property
    def use_legacy_file_upload_dialog_handling(self) -> bool:
        """:Returns: The options Use Legacy File Upload Dialog Handling value"""
        return self._options.get(self.USE_LEGACY_FILE_UPLOAD_DIALOG_HANDLING)

    @use_legacy_file_upload_dialog_handling.setter
    def use_legacy_file_upload_dialog_handling(self, value: bool) -> None:
        """Sets the options Use Legacy File Upload Dialog Handling value.

        :Args:
         - value: boolean value
        """
        self._options[self.USE_LEGACY_FILE_UPLOAD_DIALOG_HANDLING] = value

    @property
    def attach_to_edge_chrome(self) -> bool:
        """:Returns: The options Attach to Edge Chrome value"""
        return self._options.get(self.ATTACH_TO_EDGE_CHROME)

    @attach_to_edge_chrome.setter
    def attach_to_edge_chrome(self, value: bool) -> None:
        """Sets the options Attach to Edge Chrome value.

        :Args:
         - value: boolean value
        """
        self._options[self.ATTACH_TO_EDGE_CHROME] = value

    @property
    def edge_executable_path(self) -> str:
        """:Returns: The options Edge Executable Path value"""
        return self._options.get(self.EDGE_EXECUTABLE_PATH)

    @edge_executable_path.setter
    def edge_executable_path(self, value: str) -> None:
        """Sets the options Initial Browser Url value.

        :Args:
         - value: Path string
        """
        self._options[self.EDGE_EXECUTABLE_PATH] = value

    @property
    def additional_options(self) -> dict:
        """:Returns: The additional options"""
        return self._additional

    def add_additional_option(self, name: str, value):
        """Adds an additional option not yet added as a safe option for IE.

        :Args:
         - name: name of the option to add
         - value: value of the option to add
        """
        self._additional[name] = value

    def to_capabilities(self) -> dict:
        """Marshals the IE options to the correct object."""
        caps = self._caps

        opts = self._options.copy()
        if len(self._arguments) > 0:
            opts[self.SWITCHES] = " ".join(self._arguments)

        if len(self._additional) > 0:
            opts.update(self._additional)

        if len(opts) > 0:
            caps[Options.KEY] = opts
        return caps

    @property
    def default_capabilities(self) -> dict:
        return DesiredCapabilities.INTERNETEXPLORER.copy()
