from __future__ import annotations

from os import environ
import requests
from functools import cached_property
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE
from .errors import VersionNotFoundError
from . import debug

PACKAGE_NAME = "g4f"
GITHUB_REPOSITORY = "xtekky/gpt4free"

def get_pypi_version(package_name: str) -> str:
    """
    Retrieves the latest version of a package from PyPI.

    Args:
        package_name (str): The name of the package for which to retrieve the version.

    Returns:
        str: The latest version of the specified package from PyPI.

    Raises:
        VersionNotFoundError: If there is an error in fetching the version from PyPI.
    """
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json").json()
        return response["info"]["version"]
    except requests.RequestException as e:
        raise VersionNotFoundError(f"Failed to get PyPI version: {e}")

def get_github_version(repo: str) -> str:
    """
    Retrieves the latest release version from a GitHub repository.

    Args:
        repo (str): The name of the GitHub repository.

    Returns:
        str: The latest release version from the specified GitHub repository.

    Raises:
        VersionNotFoundError: If there is an error in fetching the version from GitHub.
    """
    try:
        response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()
        return response["tag_name"]
    except requests.RequestException as e:
        raise VersionNotFoundError(f"Failed to get GitHub release version: {e}")

class VersionUtils:
    """
    Utility class for managing and comparing package versions of 'g4f'.
    """
    @cached_property
    def current_version(self) -> str:
        """
        Retrieves the current version of the 'g4f' package.

        Returns:
            str: The current version of 'g4f'.

        Raises:
            VersionNotFoundError: If the version cannot be determined from the package manager, 
                                  Docker environment, or git repository.
        """
        if debug.version:
            return debug.version

        # Read from package manager
        try:
            return get_package_version(PACKAGE_NAME)
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

        return None

    @property
    def latest_version(self) -> str:
        """
        Retrieves the latest version of the 'g4f' package.

        Returns:
            str: The latest version of 'g4f'.
        """
        # Is installed via package manager?
        try:
            get_package_version(PACKAGE_NAME)
        except PackageNotFoundError:
            return get_github_version(GITHUB_REPOSITORY)
        return get_pypi_version(PACKAGE_NAME)

    def check_version(self) -> None:
        """
        Checks if the current version of 'g4f' is up to date with the latest version.

        Note:
            If a newer version is available, it prints a message with the new version and update instructions.
        """
        try:
            if self.current_version != self.latest_version:
                print(f'New g4f version: {self.latest_version} (current: {self.current_version}) | pip install -U g4f')
        except Exception as e:
            print(f'Failed to check g4f version: {e}')

utils = VersionUtils()
