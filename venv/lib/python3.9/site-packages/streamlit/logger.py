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

"""Logging module."""

import logging
import sys
from typing import Dict, Union

from typing_extensions import Final

DEFAULT_LOG_MESSAGE: Final = "%(asctime)s %(levelname) -7s " "%(name)s: %(message)s"

# Loggers for each name are saved here.
_loggers: Dict[str, logging.Logger] = {}

# The global log level is set here across all names.
_global_log_level = logging.INFO


def set_log_level(level: Union[str, int]) -> None:
    """Set log level."""
    logger = get_logger(__name__)

    if isinstance(level, str):
        level = level.upper()
    if level == "CRITICAL" or level == logging.CRITICAL:
        log_level = logging.CRITICAL
    elif level == "ERROR" or level == logging.ERROR:
        log_level = logging.ERROR
    elif level == "WARNING" or level == logging.WARNING:
        log_level = logging.WARNING
    elif level == "INFO" or level == logging.INFO:
        log_level = logging.INFO
    elif level == "DEBUG" or level == logging.DEBUG:
        log_level = logging.DEBUG
    else:
        msg = 'undefined log level "%s"' % level
        logger.critical(msg)
        sys.exit(1)

    for log in _loggers.values():
        log.setLevel(log_level)

    global _global_log_level
    _global_log_level = log_level


def setup_formatter(logger: logging.Logger) -> None:
    """Set up the console formatter for a given logger."""
    # Deregister any previous console loggers.
    if hasattr(logger, "streamlit_console_handler"):
        logger.removeHandler(logger.streamlit_console_handler)

    logger.streamlit_console_handler = logging.StreamHandler()  # type: ignore[attr-defined]

    # Import here to avoid circular imports
    from streamlit import config

    if config._config_options:
        # logger is required in ConfigOption.set_value
        # Getting the config option before the config file has been parsed
        # can create an infinite loop
        message_format = config.get_option("logger.messageFormat")
    else:
        message_format = DEFAULT_LOG_MESSAGE
    formatter = logging.Formatter(fmt=message_format)
    formatter.default_msec_format = "%s.%03d"
    logger.streamlit_console_handler.setFormatter(formatter)  # type: ignore[attr-defined]

    # Register the new console logger.
    logger.addHandler(logger.streamlit_console_handler)  # type: ignore[attr-defined]


def update_formatter() -> None:
    for log in _loggers.values():
        setup_formatter(log)


def init_tornado_logs() -> None:
    """Set Tornado log levels.

    This function does not import any Tornado code, so it's safe to call even
    when Server is not running.
    """
    # http://www.tornadoweb.org/en/stable/log.html
    for log in ("access", "application", "general"):
        # get_logger will set the log level for the logger with the given name.
        get_logger(f"tornado.{log}")


def get_logger(name: str) -> logging.Logger:
    """Return a logger.

    Parameters
    ----------
    name : str
        The name of the logger to use. You should just pass in __name__.

    Returns
    -------
    Logger

    """
    if name in _loggers.keys():
        return _loggers[name]

    if name == "root":
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    logger.setLevel(_global_log_level)
    logger.propagate = False
    setup_formatter(logger)

    _loggers[name] = logger

    return logger
