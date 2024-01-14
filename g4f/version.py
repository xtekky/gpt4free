from os import environ
import requests
from functools import cached_property
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE
from .errors import VersionNotFoundError

def get_pypi_version(package_name: str) -> str:
    """
    Get the latest version of a package from PyPI.

    :param package_name: The name of the package.
    :return: The latest version of the package as a string.
    """
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json").json()
        return response["info"]["version"]
    except requests.RequestException as e:
        raise VersionNotFoundError(f"Failed to get PyPI version: {e}")

def get_github_version(repo: str) -> str:
    """
    Get the latest release version from a GitHub repository.

    :param repo: The name of the GitHub repository.
    :return: The latest release version as a string.
    """
    try:
        response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
        return response["tag_name"]
    except requests.RequestException as e:
        raise VersionNotFoundError(f"Failed to get GitHub release version: {e}")

def get_latest_version():
    """
    Get the latest release version from PyPI or the GitHub repository.

    :return: The latest release version as a string.
    """
    try:
        # Is installed via package manager?
        get_package_version("g4f")
        return get_pypi_version("g4f")
    except PackageNotFoundError:
        # Else use Github version:
        return get_github_version("xtekky/gpt4free")

class VersionUtils:
    """
    Utility class for managing and comparing package versions.
    """
    @cached_property
    def current_version(self) -> str:
        """
        Get the current version of the g4f package.

        :return: The current version as a string.
        """
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
        """
        Get the latest version of the g4f package.

        :return: The latest version as a string.
        """
        return get_latest_version()

    def check_version(self) -> None:
        """
        Check if the current version is up to date with the latest version.
        """
        try:
            if self.current_version != self.latest_version:
                print(f'New g4f version: {self.latest_version} (current: {self.current_version}) | pip install -U g4f')
        except Exception as e:
            print(f'Failed to check g4f version: {e}')

utils = VersionUtils()