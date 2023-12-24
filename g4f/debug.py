from os import environ
import requests
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE
from .errors import VersionNotFoundError

logging = False
version_check = True

def get_version() -> str:
    # Read from package manager
    try:
        return get_package_version("g4f")
    except PackageNotFoundError:
        pass
    # Read from docker environment
    current_version = environ.get("G4F_VERSION")
    if current_version:
        return current_version
    # Read from git repository
    try:
        command = ["git", "describe", "--tags", "--abbrev=0"]
        return check_output(command, text=True, stderr=PIPE).strip()
    except CalledProcessError:
        pass
    raise VersionNotFoundError("Version not found")
    
def get_latest_version() -> str:
    if environ.get("G4F_VERSION"):
        url = "https://registry.hub.docker.com/v2/repositories/"
        url += "hlohaus789/g4f"
        url += "/tags?page_size=2&ordering=last_updated"
        response = requests.get(url).json()
        return response["results"][1]["name"]
    response = requests.get("https://pypi.org/pypi/g4f/json").json()
    return response["info"]["version"]

def check_pypi_version() -> None:
    try:
        version = get_version()
        latest_version = get_latest_version()
        if version != latest_version:
            print(f'New pypi version: {latest_version} (current: {version}) | pip install -U g4f')
    except Exception as e:
        print(f'Failed to check g4f pypi version: {e}')