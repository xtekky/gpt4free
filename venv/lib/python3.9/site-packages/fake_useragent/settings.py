import os
import tempfile

try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata

__version__ = metadata.version("fake-useragent")

DB = os.path.join(
    tempfile.gettempdir(),
    "fake_useragent_{version}.json".format(
        version=__version__,
    ),
)

BROWSER_BASE_PAGE = "https://useragentstring.com/pages/{browser}/"  # noqa

BROWSERS_COUNT_LIMIT = 50

REPLACEMENTS = {
    " ": "",
    "_": "",
}

SHORTCUTS = {
    "internetexplorer": "internet explorer",
    "ie": "internet explorer",
    "msie": "internet explorer",
    "microsoft edge": "edge",
    "google": "chrome",
    "googlechrome": "chrome",
    "ff": "firefox",
}

OVERRIDES = {
    "Edge/IE": "Internet Explorer",
    "IE/Edge": "Internet Explorer",
}

HTTP_TIMEOUT = 5

HTTP_RETRIES = 2

HTTP_DELAY = 0.1
