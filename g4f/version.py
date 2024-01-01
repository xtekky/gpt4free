from os import environ
import requests
from functools import cached_property
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE
from .errors import VersionNotFoundError


class VersionUtils():
    @cached_property
    def current_version(self) -> str:
        # Read from package manager
        try:
            return get_package_version("g4f")
        except PackageNotFoundError:
            pass
        # Read from docker environment
        version = environ.get("G4F_VERSION")
        if version:
            return version
        # Read from git repository
        try:
            command = ["git", "describe", "--tags", "--abbrev=0"]
            return check_output(command, text=True, stderr=PIPE).strip()
        except CalledProcessError:
            pass
        raise VersionNotFoundError("Version not found")
    
    @cached_property
    def latest_version(self) -> str:
        try:
            get_package_version("g4f")
            response = requests.get("https://pypi.org/pypi/g4f/json").json()
            return response["info"]["version"]
        except PackageNotFoundError:
            url = "https://api.github.com/repos/xtekky/gpt4free/releases/latest"
            response = requests.get(url).json()
            return response["tag_name"]

    def check_pypi_version(self) -> None:
        try:
            if self.current_version != self.latest_version:
                print(f'New pypi version: {self.latest_version} (current: {self.version}) | pip install -U g4f')
        except Exception as e:
            print(f'Failed to check g4f pypi version: {e}')
         
utils = VersionUtils()