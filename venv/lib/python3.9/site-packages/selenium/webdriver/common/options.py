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
import typing
from abc import ABCMeta
from abc import abstractmethod

from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy


class BaseOptions(metaclass=ABCMeta):
    """Base class for individual browser options."""

    def __init__(self) -> None:
        super().__init__()
        self._caps = self.default_capabilities
        self.set_capability("pageLoadStrategy", "normal")
        self.mobile_options = None

    @property
    def capabilities(self):
        return self._caps

    def set_capability(self, name, value) -> None:
        """Sets a capability."""
        self._caps[name] = value

    @property
    def browser_version(self) -> str:
        """
        :returns: the version of the browser if set, otherwise None.
        """
        return self._caps.get("browserVersion")

    @browser_version.setter
    def browser_version(self, version: str) -> None:
        """Requires the major version of the browser to match provided value:
        https://w3c.github.io/webdriver/#dfn-browser-version.

        :param version: The required version of the browser
        """
        self.set_capability("browserVersion", version)

    @property
    def platform_name(self) -> str:
        """
        :returns: The name of the platform
        """
        return self._caps["platformName"]

    @platform_name.setter
    def platform_name(self, platform: str) -> None:
        """Requires the platform to match the provided value:
        https://w3c.github.io/webdriver/#dfn-platform-name.

        :param platform: the required name of the platform
        """
        self.set_capability("platformName", platform)

    @property
    def page_load_strategy(self) -> str:
        """
        :returns: page load strategy if set, the default is "normal"
        """
        return self._caps["pageLoadStrategy"]

    @page_load_strategy.setter
    def page_load_strategy(self, strategy: str) -> None:
        """Determines the point at which a navigation command is returned:
        https://w3c.github.io/webdriver/#dfn-table-of-page-load-strategies.

        :param strategy: the strategy corresponding to a document readiness state
        """
        if strategy in ["normal", "eager", "none"]:
            self.set_capability("pageLoadStrategy", strategy)
        else:
            raise ValueError("Strategy can only be one of the following: normal, eager, none")

    @property
    def unhandled_prompt_behavior(self) -> str:
        """
        :returns: unhandled prompt behavior if set, the default is "dismiss and notify"
        """
        return self._caps["unhandledPromptBehavior"]

    @unhandled_prompt_behavior.setter
    def unhandled_prompt_behavior(self, behavior: str) -> None:
        """How the driver should respond when an alert is present and the
        command sent is not handling the alert:
        https://w3c.github.io/webdriver/#dfn-table-of-page-load-strategies.

        :param behavior: behavior to use when an alert is encountered
        """
        if behavior in ["dismiss", "accept", "dismiss and notify", "accept and notify", "ignore"]:
            self.set_capability("unhandledPromptBehavior", behavior)
        else:
            raise ValueError(
                "Behavior can only be one of the following: dismiss, accept, dismiss and notify, "
                "accept and notify, ignore"
            )

    @property
    def timeouts(self) -> dict:
        """
        :returns: Values for implicit timeout, pageLoad timeout and script timeout if set (in milliseconds)
        """
        return self._caps["timeouts"]

    @timeouts.setter
    def timeouts(self, timeouts: dict) -> None:
        """How long the driver should wait for actions to complete before
        returning an error https://w3c.github.io/webdriver/#timeouts.

        :param timeouts: values in milliseconds for implicit wait, page load and script timeout
        """
        if all(x in ("implicit", "pageLoad", "script") for x in timeouts.keys()):
            self.set_capability("timeouts", timeouts)
        else:
            raise ValueError("Timeout keys can only be one of the following: implicit, pageLoad, script")

    def enable_mobile(
        self,
        android_package: typing.Optional[str] = None,
        android_activity: typing.Optional[str] = None,
        device_serial: typing.Optional[str] = None,
    ) -> None:
        """Enables mobile browser use for browsers that support it.

        :Args:
            android_activity: The name of the android package to start
        """
        if not android_package:
            raise AttributeError("android_package must be passed in")
        self.mobile_options = {"androidPackage": android_package}
        if android_activity:
            self.mobile_options["androidActivity"] = android_activity
        if device_serial:
            self.mobile_options["androidDeviceSerial"] = device_serial

    @property
    def accept_insecure_certs(self) -> bool:
        """
        :returns: whether the session accepts insecure certificates
        """
        return self._caps.get("acceptInsecureCerts", False)

    @accept_insecure_certs.setter
    def accept_insecure_certs(self, value: bool) -> None:
        """Whether untrusted and self-signed TLS certificates are implicitly
        trusted: https://w3c.github.io/webdriver/#dfn-insecure-tls-
        certificates.

        :param value: whether to accept insecure certificates
        """
        self._caps["acceptInsecureCerts"] = value

    @property
    def strict_file_interactability(self) -> bool:
        """
        :returns: whether session is strict about file interactability
        """
        return self._caps.get("strictFileInteractability", False)

    @strict_file_interactability.setter
    def strict_file_interactability(self, value: bool) -> None:
        """Whether interactability checks will be applied to file type input
        elements. The default is false.

        :param value: whether file interactability is strict
        """
        self._caps["strictFileInteractability"] = value

    @property
    def set_window_rect(self) -> bool:
        """
        :returns: whether the remote end supports setting window size and position
        """
        return self._caps.get("setWindowRect", False)

    @set_window_rect.setter
    def set_window_rect(self, value: bool) -> None:
        """Whether the remote end supports all of the resizing and positioning
        commands. The default is false. https://w3c.github.io/webdriver/#dfn-
        strict-file-interactability.

        :param value: whether remote end must support setting window resizing and repositioning
        """
        self._caps["setWindowRect"] = value

    @property
    def proxy(self) -> Proxy:
        """
        :Returns: Proxy if set, otherwise None.
        """
        return self._proxy

    @proxy.setter
    def proxy(self, value: Proxy) -> None:
        if not isinstance(value, Proxy):
            raise InvalidArgumentException("Only Proxy objects can be passed in.")
        self._proxy = value

    @abstractmethod
    def to_capabilities(self):
        """Convert options into capabilities dictionary."""

    @property
    @abstractmethod
    def default_capabilities(self):
        """Return minimal capabilities necessary as a dictionary."""


class ArgOptions(BaseOptions):
    def __init__(self) -> None:
        super().__init__()
        self._arguments = []
        self._ignore_local_proxy = False

    @property
    def arguments(self):
        """
        :Returns: A list of arguments needed for the browser
        """
        return self._arguments

    def add_argument(self, argument):
        """Adds an argument to the list.

        :Args:
         - Sets the arguments
        """
        if argument:
            self._arguments.append(argument)
        else:
            raise ValueError("argument can not be null")

    def ignore_local_proxy_environment_variables(self) -> None:
        """By calling this you will ignore HTTP_PROXY and HTTPS_PROXY from
        being picked up and used."""
        self._ignore_local_proxy = True

    def to_capabilities(self):
        return self._caps

    @property
    def default_capabilities(self):
        return {}
