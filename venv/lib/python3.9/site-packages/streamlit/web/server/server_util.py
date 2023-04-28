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

"""Server related utility functions"""

from typing import Optional
from urllib.parse import urljoin

import tornado.web

from streamlit import config, net_util, url_util


def is_url_from_allowed_origins(url: str) -> bool:
    """Return True if URL is from allowed origins (for CORS purpose).

    Allowed origins:
    1. localhost
    2. The internal and external IP addresses of the machine where this
       function was called from.

    If `server.enableCORS` is False, this allows all origins.
    """
    if not config.get_option("server.enableCORS"):
        # Allow everything when CORS is disabled.
        return True

    hostname = url_util.get_hostname(url)

    allowed_domains = [  # List[Union[str, Callable[[], Optional[str]]]]
        # Check localhost first.
        "localhost",
        "0.0.0.0",
        "127.0.0.1",
        # Try to avoid making unnecessary HTTP requests by checking if the user
        # manually specified a server address.
        _get_server_address_if_manually_set,
        # Then try the options that depend on HTTP requests or opening sockets.
        net_util.get_internal_ip,
        net_util.get_external_ip,
    ]

    for allowed_domain in allowed_domains:
        if callable(allowed_domain):
            allowed_domain = allowed_domain()

        if allowed_domain is None:
            continue

        if hostname == allowed_domain:
            return True

    return False


def _get_server_address_if_manually_set() -> Optional[str]:
    if config.is_manually_set("browser.serverAddress"):
        return url_util.get_hostname(config.get_option("browser.serverAddress"))
    return None


def make_url_path_regex(*path, **kwargs) -> str:
    """Get a regex of the form ^/foo/bar/baz/?$ for a path (foo, bar, baz)."""
    path = [x.strip("/") for x in path if x]  # Filter out falsely components.
    path_format = r"^/%s/?$" if kwargs.get("trailing_slash", True) else r"^/%s$"
    return path_format % "/".join(path)


def get_url(host_ip: str) -> str:
    """Get the URL for any app served at the given host_ip.

    Parameters
    ----------
    host_ip : str
        The IP address of the machine that is running the Streamlit Server.

    Returns
    -------
    str
        The URL.
    """
    protocol = "https" if config.get_option("server.sslCertFile") else "http"

    port = _get_browser_address_bar_port()
    base_path = config.get_option("server.baseUrlPath").strip("/")

    if base_path:
        base_path = "/" + base_path

    host_ip = host_ip.strip("/")
    return f"{protocol}://{host_ip}:{port}{base_path}"


def _get_browser_address_bar_port() -> int:
    """Get the app URL that will be shown in the browser's address bar.

    That is, this is the port where static assets will be served from. In dev,
    this is different from the URL that will be used to connect to the
    server-browser websocket.

    """
    if config.get_option("global.developmentMode"):
        return 3000
    return int(config.get_option("browser.serverPort"))


def emit_endpoint_deprecation_notice(
    handler: tornado.web.RequestHandler, new_path: str
) -> None:
    """
    Emits the warning about deprecation of HTTP endpoint in the HTTP header.
    """
    handler.set_header("Deprecation", True)
    new_url = urljoin(f"{handler.request.protocol}://{handler.request.host}", new_path)
    handler.set_header("Link", f'<{new_url}>; rel="alternate"')
