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

"""The Proxy implementation."""


class ProxyTypeFactory:
    """Factory for proxy types."""

    @staticmethod
    def make(ff_value, string):
        return {"ff_value": ff_value, "string": string}


class ProxyType:
    """Set of possible types of proxy.

    Each proxy type has 2 properties:    'ff_value' is value of Firefox
    profile preference,    'string' is id of proxy type.
    """

    DIRECT = ProxyTypeFactory.make(0, "DIRECT")  # Direct connection, no proxy (default on Windows).
    MANUAL = ProxyTypeFactory.make(1, "MANUAL")  # Manual proxy settings (e.g., for httpProxy).
    PAC = ProxyTypeFactory.make(2, "PAC")  # Proxy autoconfiguration from URL.
    RESERVED_1 = ProxyTypeFactory.make(3, "RESERVED1")  # Never used.
    AUTODETECT = ProxyTypeFactory.make(4, "AUTODETECT")  # Proxy autodetection (presumably with WPAD).
    SYSTEM = ProxyTypeFactory.make(5, "SYSTEM")  # Use system settings (default on Linux).
    UNSPECIFIED = ProxyTypeFactory.make(6, "UNSPECIFIED")  # Not initialized (for internal use).

    @classmethod
    def load(cls, value):
        if isinstance(value, dict) and "string" in value:
            value = value["string"]
        value = str(value).upper()
        for attr in dir(cls):
            attr_value = getattr(cls, attr)
            if isinstance(attr_value, dict) and "string" in attr_value and attr_value["string"] == value:
                return attr_value
        raise Exception(f"No proxy type is found for {value}")


class Proxy:
    """Proxy contains information about proxy type and necessary proxy
    settings."""

    proxyType = ProxyType.UNSPECIFIED
    autodetect = False
    ftpProxy = ""
    httpProxy = ""
    noProxy = ""
    proxyAutoconfigUrl = ""
    sslProxy = ""
    socksProxy = ""
    socksUsername = ""
    socksPassword = ""
    socksVersion = None

    def __init__(self, raw=None):
        """Creates a new Proxy.

        :Args:
         - raw: raw proxy data. If None, default class values are used.
        """
        if raw:
            if "proxyType" in raw and raw["proxyType"]:
                self.proxy_type = ProxyType.load(raw["proxyType"])
            if "ftpProxy" in raw and raw["ftpProxy"]:
                self.ftp_proxy = raw["ftpProxy"]
            if "httpProxy" in raw and raw["httpProxy"]:
                self.http_proxy = raw["httpProxy"]
            if "noProxy" in raw and raw["noProxy"]:
                self.no_proxy = raw["noProxy"]
            if "proxyAutoconfigUrl" in raw and raw["proxyAutoconfigUrl"]:
                self.proxy_autoconfig_url = raw["proxyAutoconfigUrl"]
            if "sslProxy" in raw and raw["sslProxy"]:
                self.sslProxy = raw["sslProxy"]
            if "autodetect" in raw and raw["autodetect"]:
                self.auto_detect = raw["autodetect"]
            if "socksProxy" in raw and raw["socksProxy"]:
                self.socks_proxy = raw["socksProxy"]
            if "socksUsername" in raw and raw["socksUsername"]:
                self.socks_username = raw["socksUsername"]
            if "socksPassword" in raw and raw["socksPassword"]:
                self.socks_password = raw["socksPassword"]
            if "socksVersion" in raw and raw["socksVersion"]:
                self.socks_version = raw["socksVersion"]

    @property
    def proxy_type(self):
        """Returns proxy type as `ProxyType`."""
        return self.proxyType

    @proxy_type.setter
    def proxy_type(self, value) -> None:
        """Sets proxy type.

        :Args:
         - value: The proxy type.
        """
        self._verify_proxy_type_compatibility(value)
        self.proxyType = value

    @property
    def auto_detect(self):
        """Returns autodetect setting."""
        return self.autodetect

    @auto_detect.setter
    def auto_detect(self, value) -> None:
        """Sets autodetect setting.

        :Args:
         - value: The autodetect value.
        """
        if isinstance(value, bool):
            if self.autodetect is not value:
                self._verify_proxy_type_compatibility(ProxyType.AUTODETECT)
                self.proxyType = ProxyType.AUTODETECT
                self.autodetect = value
        else:
            raise ValueError("Autodetect proxy value needs to be a boolean")

    @property
    def ftp_proxy(self):
        """Returns ftp proxy setting."""
        return self.ftpProxy

    @ftp_proxy.setter
    def ftp_proxy(self, value) -> None:
        """Sets ftp proxy setting.

        :Args:
         - value: The ftp proxy value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.ftpProxy = value

    @property
    def http_proxy(self):
        """Returns http proxy setting."""
        return self.httpProxy

    @http_proxy.setter
    def http_proxy(self, value) -> None:
        """Sets http proxy setting.

        :Args:
         - value: The http proxy value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.httpProxy = value

    @property
    def no_proxy(self):
        """Returns noproxy setting."""
        return self.noProxy

    @no_proxy.setter
    def no_proxy(self, value) -> None:
        """Sets noproxy setting.

        :Args:
         - value: The noproxy value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.noProxy = value

    @property
    def proxy_autoconfig_url(self):
        """Returns proxy autoconfig url setting."""
        return self.proxyAutoconfigUrl

    @proxy_autoconfig_url.setter
    def proxy_autoconfig_url(self, value) -> None:
        """Sets proxy autoconfig url setting.

        :Args:
         - value: The proxy autoconfig url value.
        """
        self._verify_proxy_type_compatibility(ProxyType.PAC)
        self.proxyType = ProxyType.PAC
        self.proxyAutoconfigUrl = value

    @property
    def ssl_proxy(self):
        """Returns https proxy setting."""
        return self.sslProxy

    @ssl_proxy.setter
    def ssl_proxy(self, value) -> None:
        """Sets https proxy setting.

        :Args:
         - value: The https proxy value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.sslProxy = value

    @property
    def socks_proxy(self):
        """Returns socks proxy setting."""
        return self.socksProxy

    @socks_proxy.setter
    def socks_proxy(self, value) -> None:
        """Sets socks proxy setting.

        :Args:
         - value: The socks proxy value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.socksProxy = value

    @property
    def socks_username(self):
        """Returns socks proxy username setting."""
        return self.socksUsername

    @socks_username.setter
    def socks_username(self, value) -> None:
        """Sets socks proxy username setting.

        :Args:
         - value: The socks proxy username value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.socksUsername = value

    @property
    def socks_password(self):
        """Returns socks proxy password setting."""
        return self.socksPassword

    @socks_password.setter
    def socks_password(self, value) -> None:
        """Sets socks proxy password setting.

        :Args:
         - value: The socks proxy password value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.socksPassword = value

    @property
    def socks_version(self):
        """Returns socks proxy version setting."""
        return self.socksVersion

    @socks_version.setter
    def socks_version(self, value) -> None:
        """Sets socks proxy version setting.

        :Args:
         - value: The socks proxy version value.
        """
        self._verify_proxy_type_compatibility(ProxyType.MANUAL)
        self.proxyType = ProxyType.MANUAL
        self.socksVersion = value

    def _verify_proxy_type_compatibility(self, compatible_proxy):
        if self.proxyType not in (ProxyType.UNSPECIFIED, compatible_proxy):
            raise Exception(
                f"Specified proxy type ({compatible_proxy}) not compatible with current setting ({self.proxyType})"
            )

    def add_to_capabilities(self, capabilities):
        """Adds proxy information as capability in specified capabilities.

        :Args:
         - capabilities: The capabilities to which proxy will be added.
        """
        proxy_caps = {}
        proxy_caps["proxyType"] = self.proxyType["string"]
        if self.autodetect:
            proxy_caps["autodetect"] = self.autodetect
        if self.ftpProxy:
            proxy_caps["ftpProxy"] = self.ftpProxy
        if self.httpProxy:
            proxy_caps["httpProxy"] = self.httpProxy
        if self.proxyAutoconfigUrl:
            proxy_caps["proxyAutoconfigUrl"] = self.proxyAutoconfigUrl
        if self.sslProxy:
            proxy_caps["sslProxy"] = self.sslProxy
        if self.noProxy:
            proxy_caps["noProxy"] = self.noProxy
        if self.socksProxy:
            proxy_caps["socksProxy"] = self.socksProxy
        if self.socksUsername:
            proxy_caps["socksUsername"] = self.socksUsername
        if self.socksPassword:
            proxy_caps["socksPassword"] = self.socksPassword
        if self.socksVersion:
            proxy_caps["socksVersion"] = self.socksVersion
        capabilities["proxy"] = proxy_caps
