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

import socket
from typing import Optional

import requests
from typing_extensions import Final

from streamlit import util
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

# URL for checking the current machine's external IP address.
_AWS_CHECK_IP: Final = "http://checkip.amazonaws.com"

_external_ip: Optional[str] = None
_internal_ip: Optional[str] = None


def get_external_ip() -> Optional[str]:
    """Get the *external* IP address of the current machine.

    Returns
    -------
    string
        The external IPv4 address of the current machine.

    """
    global _external_ip

    if _external_ip is not None:
        return _external_ip

    response = _make_blocking_http_get(_AWS_CHECK_IP, timeout=5)

    if _looks_like_an_ip_adress(response):
        _external_ip = response
    else:
        LOGGER.warning(
            # fmt: off
            "Did not auto detect external IP.\n"
            "Please go to %s for debugging hints.",
            # fmt: on
            util.HELP_DOC
        )
        _external_ip = None

    return _external_ip


def get_internal_ip() -> Optional[str]:
    """Get the *local* IP address of the current machine.

    From: https://stackoverflow.com/a/28950776

    Returns
    -------
    string
        The local IPv4 address of the current machine.

    """
    global _internal_ip

    if _internal_ip is not None:
        return _internal_ip

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            _internal_ip = s.getsockname()[0]
        except Exception:
            _internal_ip = "127.0.0.1"

    return _internal_ip


def _make_blocking_http_get(url: str, timeout: float = 5) -> Optional[str]:
    try:
        text = requests.get(url, timeout=timeout).text
        if isinstance(text, str):
            text = text.strip()
        return text
    except Exception:
        return None


def _looks_like_an_ip_adress(address: Optional[str]) -> bool:
    if address is None:
        return False

    try:
        socket.inet_pton(socket.AF_INET, address)
        return True  # Yup, this is an IPv4 address!
    except (AttributeError, OSError):
        pass

    try:
        socket.inet_pton(socket.AF_INET6, address)
        return True  # Yup, this is an IPv6 address!
    except (AttributeError, OSError):
        pass

    # Nope, this is not an IP address.
    return False
