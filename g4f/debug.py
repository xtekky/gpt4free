from os import environ
from requests import get
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
    
def get_lastet_version() -> str:
    response = get("https://pypi.org/pypi/g4f/json").json()
    return response["info"]["version"]

def check_pypi_version() -> None:
    try:
        version = get_version()
        latest_version = get_lastet_version()
        if version != latest_version:
            print(f'New pypi version: {latest_version} (current: {version}) | pip install -U g4f')
    except Exception as e:
        print(f'Failed to check g4f pypi version: {e}')