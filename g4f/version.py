from __future__ import annotations

import requests
from os import environ
from functools import cached_property, lru_cache
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE

from .errors import VersionNotFoundError
from .config import PACKAGE_NAME, GITHUB_REPOSITORY
from . import debug

# Default request timeout (seconds)
REQUEST_TIMEOUT = 5


@lru_cache(maxsize=1)
def get_pypi_version(package_name: str) -> str:
    """
    Retrieves the latest version of a package from PyPI.

    Raises:
        VersionNotFoundError: If there is a network or parsing error.
    """
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/json",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["info"]["version"]
    except requests.RequestException as e:
        raise VersionNotFoundError(
            f"Failed to get PyPI version for '{package_name}'"
        ) from e


@lru_cache(maxsize=1)
def get_github_version(repo: str) -> str:
    """
    Retrieves the latest release version from a GitHub repository.

    Raises:
        VersionNotFoundError: If there is a network or parsing error.
    """
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}/releases/latest",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        if "tag_name" not in data:
            raise VersionNotFoundError(f"No tag_name found in latest GitHub release for '{repo}'")
        return data["tag_name"]
    except requests.RequestException as e:
        raise VersionNotFoundError(
            f"Failed to get GitHub release version for '{repo}'"
        ) from e


def get_git_version() -> str | None:
    """Return latest Git tag if available, else None."""
    try:
        return check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            text=True,
            stderr=PIPE
        ).strip()
    except CalledProcessError:
        return None


class VersionUtils:
    """
    Utility class for managing and comparing package versions of 'g4f'.
    """

    @cached_property
    def current_version(self) -> str:
        """
        Returns the current installed version of g4f from:
        - debug override
        - package metadata
        - environment variable (Docker)
        - git tags
        """
        if debug.version:
            return debug.version

        try:
            return get_package_version(PACKAGE_NAME)
        except PackageNotFoundError:
            pass

        version_env = environ.get("G4F_VERSION")
        if version_env:
            return version_env

        git_version = get_git_version()
        if git_version:
            return git_version

        return None

    @property
    def latest_version(self) -> str:
        """
        Returns the latest available version of g4f.
        If not installed via PyPI, falls back to GitHub releases.
        """
        try:
            get_package_version(PACKAGE_NAME)
        except PackageNotFoundError:
            return get_github_version(GITHUB_REPOSITORY)
        return get_pypi_version(PACKAGE_NAME)

    @cached_property
    def latest_version_cached(self) -> str:
        return self.latest_version

    def check_version(self, silent: bool = False) -> bool:
        """
        Checks if the current version is up-to-date.
        Returns:
            bool: True if current version is the latest, False otherwise.
        """
        try:
            current = self.current_version
            latest = self.latest_version
            up_to_date = current == latest
            if not silent:
                if up_to_date:
                    print(f"g4f is up-to-date (version {current}).")
                else:
                    print(
                        f"New g4f version available: {latest} "
                        f"(current: {current}) | pip install -U g4f"
                    )
            return up_to_date
        except Exception as e:
            if not silent:
                print(f"Failed to check g4f version: {e}")
            return True  # Assume up-to-date if check fails


# Singleton instance
utils = VersionUtils()