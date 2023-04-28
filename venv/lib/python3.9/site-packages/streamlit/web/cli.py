# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script which is run when the Streamlit package is executed."""

import os
import sys
from typing import Any, Dict, List, Optional

import click

import streamlit.runtime.caching as caching
import streamlit.runtime.legacy_caching as legacy_caching
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config
from streamlit.case_converters import to_snake_case
from streamlit.config_option import ConfigOption
from streamlit.runtime.credentials import Credentials, check_credentials
from streamlit.web.cache_storage_manager_config import (
    create_default_cache_storage_manager,
)

ACCEPTED_FILE_EXTENSIONS = ("py", "py3")

LOG_LEVELS = ("error", "warning", "info", "debug")


def _convert_config_option_to_click_option(
    config_option: ConfigOption,
) -> Dict[str, Any]:
    """Composes given config option options as options for click lib."""
    option = f"--{config_option.key}"
    param = config_option.key.replace(".", "_")
    description = config_option.description
    if config_option.deprecated:
        if description is None:
            description = ""
        description += (
            f"\n {config_option.deprecation_text} - {config_option.expiration_date}"
        )
    envvar = f"STREAMLIT_{to_snake_case(param).upper()}"

    return {
        "param": param,
        "description": description,
        "type": config_option.type,
        "option": option,
        "envvar": envvar,
    }


def configurator_options(func):
    """Decorator that adds config param keys to click dynamically."""
    for _, value in reversed(_config._config_options_template.items()):
        parsed_parameter = _convert_config_option_to_click_option(value)
        config_option = click.option(
            parsed_parameter["option"],
            parsed_parameter["param"],
            help=parsed_parameter["description"],
            type=parsed_parameter["type"],
            show_envvar=True,
            envvar=parsed_parameter["envvar"],
        )
        func = config_option(func)
    return func


def _download_remote(main_script_path: str, url_path: str) -> None:
    """Fetch remote file at url_path to main_script_path"""
    import requests

    with open(main_script_path, "wb") as fp:
        try:
            resp = requests.get(url_path)
            resp.raise_for_status()
            fp.write(resp.content)
        except requests.exceptions.RequestException as e:
            raise click.BadParameter(f"Unable to fetch {url_path}.\n{e}")


@click.group(context_settings={"auto_envvar_prefix": "STREAMLIT"})
@click.option("--log_level", show_default=True, type=click.Choice(LOG_LEVELS))
@click.version_option(prog_name="Streamlit")
def main(log_level="info"):
    """Try out a demo with:

        $ streamlit hello

    Or use the line below to run your own script:

        $ streamlit run your_script.py
    """

    if log_level:
        from streamlit.logger import get_logger

        LOGGER = get_logger(__name__)
        LOGGER.warning(
            "Setting the log level using the --log_level flag is unsupported."
            "\nUse the --logger.level flag (after your streamlit command) instead."
        )


@main.command("help")
def help():
    """Print this help message."""
    # We use _get_command_line_as_string to run some error checks but don't do
    # anything with its return value.
    _get_command_line_as_string()

    assert len(sys.argv) == 2  # This is always true, but let's assert anyway.

    # Pretend user typed 'streamlit --help' instead of 'streamlit help'.
    sys.argv[1] = "--help"
    main(prog_name="streamlit")


@main.command("version")
def main_version():
    """Print Streamlit's version number."""
    # Pretend user typed 'streamlit --version' instead of 'streamlit version'
    import sys

    # We use _get_command_line_as_string to run some error checks but don't do
    # anything with its return value.
    _get_command_line_as_string()

    assert len(sys.argv) == 2  # This is always true, but let's assert anyway.
    sys.argv[1] = "--version"
    main()


@main.command("docs")
def main_docs():
    """Show help in browser."""
    print("Showing help page in browser...")
    from streamlit import util

    util.open_browser("https://docs.streamlit.io")


@main.command("hello")
@configurator_options
def main_hello(**kwargs):
    """Runs the Hello World script."""
    from streamlit.hello import Hello

    bootstrap.load_config_options(flag_options=kwargs)
    filename = Hello.__file__
    _main_run(filename, flag_options=kwargs)


@main.command("run")
@configurator_options
@click.argument("target", required=True, envvar="STREAMLIT_RUN_TARGET")
@click.argument("args", nargs=-1)
def main_run(target: str, args=None, **kwargs):
    """Run a Python script, piping stderr to Streamlit.

    The script can be local or it can be an url. In the latter case, Streamlit
    will download the script to a temporary file and runs this file.

    """
    from validators import url

    bootstrap.load_config_options(flag_options=kwargs)

    _, extension = os.path.splitext(target)
    if extension[1:] not in ACCEPTED_FILE_EXTENSIONS:
        if extension[1:] == "":
            raise click.BadArgumentUsage(
                "Streamlit requires raw Python (.py) files, but the provided file has no extension.\nFor more information, please see https://docs.streamlit.io"
            )
        else:
            raise click.BadArgumentUsage(
                f"Streamlit requires raw Python (.py) files, not {extension}.\nFor more information, please see https://docs.streamlit.io"
            )

    if url(target):
        from streamlit.temporary_directory import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            from urllib.parse import urlparse

            from streamlit import url_util

            path = urlparse(target).path
            main_script_path = os.path.join(
                temp_dir, path.strip("/").rsplit("/", 1)[-1]
            )
            # if this is a GitHub/Gist blob url, convert to a raw URL first.
            target = url_util.process_gitblob_url(target)
            _download_remote(main_script_path, target)
            _main_run(main_script_path, args, flag_options=kwargs)
    else:
        if not os.path.exists(target):
            raise click.BadParameter(f"File does not exist: {target}")
        _main_run(target, args, flag_options=kwargs)


def _get_command_line_as_string() -> Optional[str]:
    import subprocess

    parent = click.get_current_context().parent
    if parent is None:
        return None

    if "streamlit.cli" in parent.command_path:
        raise RuntimeError(
            "Running streamlit via `python -m streamlit.cli <command>` is"
            " unsupported. Please use `python -m streamlit <command>` instead."
        )

    cmd_line_as_list = [parent.command_path]
    cmd_line_as_list.extend(sys.argv[1:])
    return subprocess.list2cmdline(cmd_line_as_list)


def _main_run(
    file,
    args: Optional[List[str]] = None,
    flag_options: Optional[Dict[str, Any]] = None,
) -> None:
    if args is None:
        args = []

    if flag_options is None:
        flag_options = {}

    command_line = _get_command_line_as_string()

    check_credentials()

    bootstrap.run(file, command_line, args, flag_options)


# SUBCOMMAND: cache


@main.group("cache")
def cache():
    """Manage the Streamlit cache."""
    pass


@cache.command("clear")
def cache_clear():
    """Clear st.cache, st.cache_data, and st.cache_resource caches."""
    result = legacy_caching.clear_cache()
    cache_path = legacy_caching.get_cache_path()
    if result:
        print(f"Cleared directory {cache_path}.")
    else:
        print(f"Nothing to clear at {cache_path}.")

    # in this `streamlit cache clear` cli command we cannot use the
    # `cache_storage_manager from runtime (since runtime is not initialized)
    # so we create a new cache_storage_manager instance that used in runtime,
    # and call clear_all() method for it.
    # This will not remove the in-memory cache.
    cache_storage_manager = create_default_cache_storage_manager()
    cache_storage_manager.clear_all()
    caching.cache_resource.clear()


# SUBCOMMAND: config


@main.group("config")
def config():
    """Manage Streamlit's config settings."""
    pass


@config.command("show")
@configurator_options
def config_show(**kwargs):
    """Show all of Streamlit's config settings."""

    bootstrap.load_config_options(flag_options=kwargs)

    _config.show_config()


# SUBCOMMAND: activate


@main.group("activate", invoke_without_command=True)
@click.pass_context
def activate(ctx):
    """Activate Streamlit by entering your email."""
    if not ctx.invoked_subcommand:
        Credentials.get_current().activate()


@activate.command("reset")
def activate_reset():
    """Reset Activation Credentials."""
    Credentials.get_current().reset()


# SUBCOMMAND: test


@main.group("test", hidden=True)
def test():
    """Internal-only commands used for testing.

    These commands are not included in the output of `streamlit help`.
    """
    pass


@test.command("prog_name")
def test_prog_name():
    """Assert that the program name is set to `streamlit test`.

    This is used by our cli-smoke-tests to verify that the program name is set
    to `streamlit ...` whether the streamlit binary is invoked directly or via
    `python -m streamlit ...`.
    """
    # We use _get_command_line_as_string to run some error checks but don't do
    # anything with its return value.
    _get_command_line_as_string()

    parent = click.get_current_context().parent

    assert parent is not None
    assert parent.command_path == "streamlit test"


if __name__ == "__main__":
    main()
