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
import errno
import logging
import os
import subprocess
import typing
from abc import ABC
from abc import abstractmethod
from platform import system
from subprocess import DEVNULL
from subprocess import PIPE
from time import sleep
from urllib import request
from urllib.error import URLError

from selenium.common.exceptions import WebDriverException
from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import utils

logger = logging.getLogger(__name__)


_HAS_NATIVE_DEVNULL = True


class Service(ABC):
    """The abstract base class for all service objects.  Services typically
    launch a child program in a new process as an interim process to
    communicate with a browser.

    :param executable: install path of the executable.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param log_file: (Optional) file descriptor (pos int) or file object with a valid file descriptor.
        subprocess.PIPE & subprocess.DEVNULL are also valid values.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    """

    def __init__(
        self,
        executable: str,
        port: int = 0,
        log_file: SubprocessStdAlias = DEVNULL,
        env: typing.Optional[typing.Mapping[typing.Any, typing.Any]] = None,
        start_error_message: typing.Optional[str] = None,
        **kwargs,
    ) -> None:
        self._path = executable
        self.port = port or utils.free_port()
        self.log_file = open(os.devnull, "wb") if not _HAS_NATIVE_DEVNULL and log_file == DEVNULL else log_file
        self.start_error_message = start_error_message or ""
        # Default value for every python subprocess: subprocess.Popen(..., creationflags=0)
        self.popen_kw = kwargs.pop("popen_kw", {})
        self.creation_flags = self.popen_kw.pop("creation_flags", 0)
        self.env = env or os.environ

    @property
    def service_url(self) -> str:
        """Gets the url of the Service."""
        return f"http://{utils.join_host_port('localhost', self.port)}"

    @abstractmethod
    def command_line_args(self) -> typing.List[str]:
        """A List of program arguments (excluding the executable)."""
        raise NotImplementedError("This method needs to be implemented in a sub class")

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str) -> None:
        self._path = str(value)

    def start(self) -> None:
        """Starts the Service.

        :Exceptions:
         - WebDriverException : Raised either when it can't start the service
           or when it can't connect to the service
        """
        self._start_process(self._path)

        count = 0
        while True:
            self.assert_process_still_running()
            if self.is_connectable():
                break

            count += 1
            sleep(0.5)
            if count == 60:
                raise WebDriverException(f"Can not connect to the Service {self._path}")

    def assert_process_still_running(self) -> None:
        """Check if the underlying process is still running."""
        return_code = self.process.poll()
        if return_code:
            raise WebDriverException(f"Service {self._path} unexpectedly exited. Status code was: {return_code}")

    def is_connectable(self) -> bool:
        """Establishes a socket connection to determine if the service running
        on the port is accessible."""
        return utils.is_connectable(self.port)

    def send_remote_shutdown_command(self) -> None:
        """Dispatch an HTTP request to the shutdown endpoint for the service in
        an attempt to stop it."""
        try:
            request.urlopen(f"{self.service_url}/shutdown")
        except URLError:
            return

        for _ in range(30):
            if not self.is_connectable():
                break
            sleep(1)

    def stop(self) -> None:
        """Stops the service."""
        if self.log_file != PIPE and not (self.log_file == DEVNULL and _HAS_NATIVE_DEVNULL):
            try:
                # Todo: Be explicit in what we are catching here.
                if hasattr(self.log_file, "close"):
                    self.log_file.close()  # type: ignore
            except Exception:
                pass

        if self.process is not None:
            try:
                self.send_remote_shutdown_command()
            except TypeError:
                pass
            self._terminate_process()

    def _terminate_process(self) -> None:
        """Terminate the child process.

        On POSIX this attempts a graceful SIGTERM followed by a SIGKILL,
        on a Windows OS kill is an alias to terminate.  Terminating does
        not raise itself if something has gone wrong but (currently)
        silently ignores errors here.
        """
        try:
            stdin, stdout, stderr = self.process.stdin, self.process.stdout, self.process.stderr
            for stream in stdin, stdout, stderr:
                try:
                    stream.close()  # type: ignore
                except AttributeError:
                    pass
            self.process.terminate()
            try:
                self.process.wait(60)
            except subprocess.TimeoutError:
                logger.error(
                    "Service process refused to terminate gracefully with SIGTERM, escalating to SIGKILL.",
                    exc_info=True,
                )
                self.process.kill()
        except OSError:
            logger.error("Error terminating service process.", exc_info=True)

    def __del__(self) -> None:
        # `subprocess.Popen` doesn't send signal on `__del__`;
        # so we attempt to close the launched process when `__del__`
        # is triggered.
        # do not use globals here; interpreter shutdown may have already cleaned them up
        # and they would be `None`. This goes for anything this method is referencing internally.
        try:
            self.stop()
        except Exception:
            pass

    def _start_process(self, path: str) -> None:
        """Creates a subprocess by executing the command provided.

        :param cmd: full command to execute
        """
        cmd = [path]
        cmd.extend(self.command_line_args())
        close_file_descriptors = self.popen_kw.pop("close_fds", system() != "Windows")
        try:
            self.process = subprocess.Popen(
                cmd,
                env=self.env,
                close_fds=close_file_descriptors,
                stdout=self.log_file,
                stderr=self.log_file,
                stdin=PIPE,
                creationflags=self.creation_flags,
                **self.popen_kw,
            )
            logger.debug(f"Started executable: `{self._path}` in a child process with pid: {self.process.pid}")
        except TypeError:
            raise
        except OSError as err:
            if err.errno == errno.ENOENT:
                raise WebDriverException(
                    f"'{os.path.basename(self._path)}' executable needs to be in PATH. {self.start_error_message}"
                )
            if err.errno == errno.EACCES:
                raise WebDriverException(
                    f"'{os.path.basename(self._path)}' executable may have wrong permissions. {self.start_error_message}"
                )
            raise
        except Exception as e:
            raise WebDriverException(
                f"The executable {os.path.basename(self._path)} needs to be available in the path. {self.start_error_message}\n{str(e)}"
            )
