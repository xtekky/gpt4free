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

from selenium.webdriver.common import service


class ChromiumService(service.Service):
    """A Service class that is responsible for the starting and stopping the
    WebDriver instance of the ChromiumDriver.

    :param executable_path: install path of the executable.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param service_args: (Optional) List of args to be passed to the subprocess when launching the executable.
    :param log_path: (Optional) String to be passed to the executable as `--log-path`.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    :param start_error_message: (Optional) Error message that forms part of the error when problems occur
    launching the subprocess.
    """

    def __init__(
        self,
        executable_path: str,
        port: int = 0,
        service_args: typing.Optional[typing.List[str]] = None,
        log_path: typing.Optional[str] = None,
        env: typing.Optional[typing.Mapping[str, str]] = None,
        start_error_message: typing.Optional[str] = None,
        **kwargs,
    ) -> None:
        self.service_args = service_args or []
        if log_path:
            self.service_args.append(f"--log-path={log_path}")

        super().__init__(
            executable=executable_path,
            port=port,
            env=env,
            start_error_message=start_error_message,
            **kwargs,
        )

    def command_line_args(self) -> typing.List[str]:
        return [f"--port={self.port}"] + self.service_args
