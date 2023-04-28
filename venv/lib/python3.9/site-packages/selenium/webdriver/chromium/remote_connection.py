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

from selenium.webdriver.remote.remote_connection import RemoteConnection


class ChromiumRemoteConnection(RemoteConnection):
    def __init__(
        self,
        remote_server_addr: str,
        vendor_prefix: str,
        browser_name: str,
        keep_alive: bool = True,
        ignore_proxy: typing.Optional[bool] = False,
    ) -> None:
        super().__init__(remote_server_addr, keep_alive, ignore_proxy=ignore_proxy)
        self.browser_name = browser_name
        self._commands["launchApp"] = ("POST", "/session/$sessionId/chromium/launch_app")
        self._commands["setPermissions"] = ("POST", "/session/$sessionId/permissions")
        self._commands["setNetworkConditions"] = ("POST", "/session/$sessionId/chromium/network_conditions")
        self._commands["getNetworkConditions"] = ("GET", "/session/$sessionId/chromium/network_conditions")
        self._commands["deleteNetworkConditions"] = ("DELETE", "/session/$sessionId/chromium/network_conditions")
        self._commands["executeCdpCommand"] = ("POST", f"/session/$sessionId/{vendor_prefix}/cdp/execute")
        self._commands["getSinks"] = ("GET", f"/session/$sessionId/{vendor_prefix}/cast/get_sinks")
        self._commands["getIssueMessage"] = ("GET", f"/session/$sessionId/{vendor_prefix}/cast/get_issue_message")
        self._commands["setSinkToUse"] = ("POST", f"/session/$sessionId/{vendor_prefix}/cast/set_sink_to_use")
        self._commands["startDesktopMirroring"] = (
            "POST",
            f"/session/$sessionId/{vendor_prefix}/cast/start_desktop_mirroring",
        )
        self._commands["startTabMirroring"] = ("POST", f"/session/$sessionId/{vendor_prefix}/cast/start_tab_mirroring")
        self._commands["stopCasting"] = ("POST", f"/session/$sessionId/{vendor_prefix}/cast/stop_casting")
