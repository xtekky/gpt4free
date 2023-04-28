import sys
import contextlib
import inspect
import io
import json
import os
import re
import ssl
import time


# We need files() from Python 3.10 or higher
if sys.version_info >= (3, 10):
    import importlib.resources as ilr
else:
    import importlib_resources as ilr

from urllib.error import URLError
from urllib.parse import quote_plus
from urllib import request
from fake_useragent.log import logger

# Fallback method for retrieving data file
try:
    from pkg_resources import resource_filename
except:
    pass

str_types = (str,)
text = str
urlopen_args = inspect.getfullargspec(request.urlopen).kwonlyargs
urlopen_has_ssl_context = "context" in urlopen_args


def get(url, verify_ssl=True):
    attempt = 0

    while True:
        requestObj = request.Request(url)

        attempt += 1

        try:
            if urlopen_has_ssl_context:
                if not verify_ssl:
                    context = ssl._create_unverified_context()
                else:
                    context = None

                with contextlib.closing(
                    request.urlopen(
                        requestObj,
                        timeout=settings.HTTP_TIMEOUT,
                        context=context,
                    )
                ) as response:
                    return response.read()
            else:  # ssl context is not supported ;(
                with contextlib.closing(
                    request.urlopen(
                        requestObj,
                        timeout=settings.HTTP_TIMEOUT,
                    )
                ) as response:
                    return response.read()
        except (URLError, OSError) as exc:
            logger.debug(
                "Error occurred during fetching %s",
                url,
                exc_info=exc,
            )

            if attempt == settings.HTTP_RETRIES:
                raise FakeUserAgentError("Maximum amount of retries reached")
            else:
                logger.debug(
                    "Sleeping for %s seconds",
                    settings.HTTP_DELAY,
                )
                time.sleep(settings.HTTP_DELAY)


def get_browser_user_agents_online(browser, verify_ssl=True):
    """
    Retrieve browser user agent strings from website
    """
    html = get(
        settings.BROWSER_BASE_PAGE.format(browser=quote_plus(browser)),
        verify_ssl=verify_ssl,
    )
    try:
        html = html.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        pass
    html = html.split("<div id='liste'>")[1]
    html = html.split("</div>")[0]

    pattern = r"<a href=\'/.*?>(.+?)</a>"
    browsers_iter = re.finditer(pattern, html, re.UNICODE)

    browsers = []

    for browser in browsers_iter:
        if "more" in browser.group(1).lower():
            continue

        browsers.append(browser.group(1))

        if len(browsers) == settings.BROWSERS_COUNT_LIMIT:
            break

    if not browsers:
        raise FakeUserAgentError(
            "No browser user-agent strings found for browser: {browser}".format(
                browser=browser
            )
        )

    return browsers


def load(browsers, use_local_file=True, verify_ssl=True):
    data = {}
    fetch_online = True
    if use_local_file:
        try:
            json_lines = (
                ilr.files("fake_useragent.data").joinpath("browsers.json").read_text()
            )
            for line in json_lines.splitlines():
                data.update(json.loads(line))
            fetch_online = False
            ret = data
        except Exception as exc:
            # Empty data just to be sure
            data = {}
            logger.warning(
                "Unable to find local data/json file or could not parse the contents using importlib-resources. Try pkg-resource next.",
                exc_info=exc,
            )
            try:
                with open(
                    resource_filename("fake_useragent", "data/browsers.json")
                ) as file:
                    json_lines = file.read()
                    for line in json_lines.splitlines():
                        data.update(json.loads(line))
                fetch_online = False
                ret = data
            except Exception as exc2:
                # Empty data just to be sure
                data = {}
                logger.warning(
                    "Could not find local data/json file or could not parse the contents using pkg-resource. Fallback to external resource.",
                    exc_info=exc2,
                )

    # Fallback behaviour or use_external_data parameter is explicitly set to True
    if fetch_online:
        try:
            # For each browser receive the user-agent strings
            for browser_name in browsers:
                browser_name = browser_name.lower().strip()
                data[browser_name] = get_browser_user_agents_online(
                    browser_name,
                    verify_ssl=verify_ssl,
                )
        except Exception:
            raise FakeUserAgentError("Could not load data from external website")
        else:
            ret = data

    if not ret:
        raise FakeUserAgentError("Data dictionary is empty", ret)

    if not isinstance(ret, dict):
        raise FakeUserAgentError("Data is not dictionary ", ret)

    return ret


def write(path, data):
    with open(path, encoding="utf-8", mode="w") as fp:
        dumped = json.dumps(data)

        if not isinstance(dumped, text):  # Python 2
            dumped = dumped.decode("utf-8")

        fp.write(dumped)


def read(path):
    with open(path, encoding="utf-8") as fp:
        return json.loads(fp.read())


def exist(path):
    return os.path.isfile(path)


def rm(path):
    if exist(path):
        os.remove(path)


def update(cache_path, browsers, verify_ssl=True):
    rm(cache_path)

    write(cache_path, load(browsers, use_local_file=False, verify_ssl=verify_ssl))


def load_cached(cache_path, browsers, verify_ssl=True):
    if not exist(cache_path):
        update(cache_path, browsers, verify_ssl=verify_ssl)

    return read(cache_path)


from fake_useragent import settings  # noqa # isort:skip
from fake_useragent.errors import FakeUserAgentError  # noqa # isort:skip
