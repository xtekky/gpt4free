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

"""Manage the user's Streamlit credentials."""

import os
import sys
import textwrap
from collections import namedtuple
from typing import Optional

import click
import toml

from streamlit import env_util, file_util, util
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


if env_util.IS_WINDOWS:
    _CONFIG_FILE_PATH = r"%userprofile%/.streamlit/config.toml"
else:
    _CONFIG_FILE_PATH = "~/.streamlit/config.toml"

_Activation = namedtuple(
    "_Activation",
    [
        "email",  # str : the user's email.
        "is_valid",  # boolean : whether the email is valid.
    ],
)


def email_prompt() -> str:
    # Emoji can cause encoding errors on non-UTF-8 terminals
    # (See https://github.com/streamlit/streamlit/issues/2284.)
    # WT_SESSION is a Windows Terminal specific environment variable. If it exists,
    # we are on the latest Windows Terminal that supports emojis
    show_emoji = sys.stdout.encoding == "utf-8" and (
        not env_util.IS_WINDOWS or os.environ.get("WT_SESSION")
    )

    # IMPORTANT: Break the text below at 80 chars.
    return """
      {0}%(welcome)s

      If you’d like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.

      %(email)s""".format(
        "👋 " if show_emoji else ""
    ) % {
        "welcome": click.style("Welcome to Streamlit!", bold=True),
        "email": click.style("Email: ", fg="blue"),
    }


# IMPORTANT: Break the text below at 80 chars.
_TELEMETRY_TEXT = """
  You can find our privacy policy at %(link)s

  Summary:
  - This open source library collects usage statistics.
  - We cannot see and do not store information contained inside Streamlit apps,
    such as text, charts, images, etc.
  - Telemetry data is stored in servers in the United States.
  - If you'd like to opt out, add the following to %(config)s,
    creating that file if necessary:

    [browser]
    gatherUsageStats = false
""" % {
    "link": click.style("https://streamlit.io/privacy-policy", underline=True),
    "config": click.style(_CONFIG_FILE_PATH),
}

_TELEMETRY_HEADLESS_TEXT = """
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.
"""

# IMPORTANT: Break the text below at 80 chars.
_INSTRUCTIONS_TEXT = """
  %(start)s
  %(prompt)s %(hello)s
""" % {
    "start": click.style("Get started by typing:", fg="blue", bold=True),
    "prompt": click.style("$", fg="blue"),
    "hello": click.style("streamlit hello", bold=True),
}


class Credentials(object):
    """Credentials class."""

    _singleton: Optional["Credentials"] = None

    @classmethod
    def get_current(cls):
        """Return the singleton instance."""
        if cls._singleton is None:
            Credentials()

        return Credentials._singleton

    def __init__(self):
        """Initialize class."""
        if Credentials._singleton is not None:
            raise RuntimeError(
                "Credentials already initialized. Use .get_current() instead"
            )

        self.activation = None
        self._conf_file = _get_credential_file_path()

        Credentials._singleton = self

    def __repr__(self) -> str:
        return util.repr_(self)

    def load(self, auto_resolve=False) -> None:
        """Load from toml file."""
        if self.activation is not None:
            LOGGER.error("Credentials already loaded. Not rereading file.")
            return

        try:
            with open(self._conf_file, "r") as f:
                data = toml.load(f).get("general")
            if data is None:
                raise Exception
            self.activation = _verify_email(data.get("email"))
        except FileNotFoundError:
            if auto_resolve:
                self.activate(show_instructions=not auto_resolve)
                return
            raise RuntimeError(
                'Credentials not found. Please run "streamlit activate".'
            )
        except Exception:
            if auto_resolve:
                self.reset()
                self.activate(show_instructions=not auto_resolve)
                return
            raise Exception(
                textwrap.dedent(
                    """
                Unable to load credentials from %s.
                Run "streamlit reset" and try again.
                """
                )
                % (self._conf_file)
            )

    def _check_activated(self, auto_resolve=True):
        """Check if streamlit is activated.

        Used by `streamlit run script.py`
        """
        try:
            self.load(auto_resolve)
        except (Exception, RuntimeError) as e:
            _exit(str(e))

        if self.activation is None or not self.activation.is_valid:
            _exit("Activation email not valid.")

    @classmethod
    def reset(cls):
        """Reset credentials by removing file.

        This is used by `streamlit activate reset` in case a user wants
        to start over.
        """
        c = Credentials.get_current()
        c.activation = None

        try:
            os.remove(c._conf_file)
        except OSError as e:
            LOGGER.error("Error removing credentials file: %s" % e)

    def save(self):
        """Save to toml file."""
        if self.activation is None:
            return

        # Create intermediate directories if necessary
        os.makedirs(os.path.dirname(self._conf_file), exist_ok=True)

        # Write the file
        data = {"email": self.activation.email}
        with open(self._conf_file, "w") as f:
            toml.dump({"general": data}, f)

    def activate(self, show_instructions: bool = True) -> None:
        """Activate Streamlit.

        Used by `streamlit activate`.
        """
        try:
            self.load()
        except RuntimeError:
            # Runtime Error is raised if credentials file is not found. In that case,
            # `self.activation` is None and we will show the activation prompt below.
            pass

        if self.activation:
            if self.activation.is_valid:
                _exit("Already activated")
            else:
                _exit(
                    "Activation not valid. Please run "
                    "`streamlit activate reset` then `streamlit activate`"
                )
        else:
            activated = False

            while not activated:
                email = click.prompt(
                    text=email_prompt(),
                    prompt_suffix="",
                    default="",
                    show_default=False,
                )

                self.activation = _verify_email(email)
                if self.activation.is_valid:
                    self.save()
                    click.secho(_TELEMETRY_TEXT)
                    if show_instructions:
                        click.secho(_INSTRUCTIONS_TEXT)
                    activated = True
                else:  # pragma: nocover
                    LOGGER.error("Please try again.")


def _verify_email(email: str) -> _Activation:
    """Verify the user's email address.

    The email can either be an empty string (if the user chooses not to enter
    it), or a string with a single '@' somewhere in it.

    Parameters
    ----------
    email : str

    Returns
    -------
    _Activation
        An _Activation object. Its 'is_valid' property will be True only if
        the email was validated.

    """
    email = email.strip()

    # We deliberately use simple email validation here
    # since we do not use email address anywhere to send emails.
    if len(email) > 0 and email.count("@") != 1:
        LOGGER.error("That doesn't look like an email :(")
        return _Activation(None, False)

    return _Activation(email, True)


def _exit(message):  # pragma: nocover
    """Exit program with error."""
    LOGGER.error(message)
    sys.exit(-1)


def _get_credential_file_path():
    return file_util.get_streamlit_file_path("credentials.toml")


def _check_credential_file_exists():
    return os.path.exists(_get_credential_file_path())


def check_credentials():
    """Check credentials and potentially activate.

    Note
    ----
    If there is no credential file and we are in headless mode, we should not
    check, since credential would be automatically set to an empty string.

    """
    from streamlit import config

    if not _check_credential_file_exists() and config.get_option("server.headless"):
        if not config.is_manually_set("browser.gatherUsageStats"):
            # If not manually defined, show short message about usage stats gathering.
            click.secho(_TELEMETRY_HEADLESS_TEXT)
        return
    Credentials.get_current()._check_activated()
