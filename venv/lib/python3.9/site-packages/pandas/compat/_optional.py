from __future__ import annotations

import importlib
import sys
import types
import warnings

from pandas.util._exceptions import find_stack_level

from pandas.util.version import Version

# Update install.rst when updating versions!

VERSIONS = {
    "bs4": "4.9.3",
    "blosc": "1.21.0",
    "bottleneck": "1.3.2",
    "brotli": "0.7.0",
    "fastparquet": "0.4.0",
    "fsspec": "2021.07.0",
    "html5lib": "1.1",
    "hypothesis": "6.13.0",
    "gcsfs": "2021.07.0",
    "jinja2": "3.0.0",
    "lxml.etree": "4.6.3",
    "matplotlib": "3.3.2",
    "numba": "0.53.1",
    "numexpr": "2.7.3",
    "odfpy": "1.4.1",
    "openpyxl": "3.0.7",
    "pandas_gbq": "0.15.0",
    "psycopg2": "2.8.6",  # (dt dec pq3 ext lo64)
    "pymysql": "1.0.2",
    "pyarrow": "1.0.1",
    "pyreadstat": "1.1.2",
    "pytest": "6.0",
    "pyxlsb": "1.0.8",
    "s3fs": "2021.08.0",
    "scipy": "1.7.1",
    "snappy": "0.6.0",
    "sqlalchemy": "1.4.16",
    "tables": "3.6.1",
    "tabulate": "0.8.9",
    "xarray": "0.19.0",
    "xlrd": "2.0.1",
    "xlwt": "1.3.0",
    "xlsxwriter": "1.4.3",
    "zstandard": "0.15.2",
    "tzdata": "2022.1",
}

# A mapping from import name to package name (on PyPI) for packages where
# these two names are different.

INSTALL_MAPPING = {
    "bs4": "beautifulsoup4",
    "bottleneck": "Bottleneck",
    "brotli": "brotlipy",
    "jinja2": "Jinja2",
    "lxml.etree": "lxml",
    "odf": "odfpy",
    "pandas_gbq": "pandas-gbq",
    "snappy": "python-snappy",
    "sqlalchemy": "SQLAlchemy",
    "tables": "pytables",
}


def get_version(module: types.ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if version is None:
        # xlrd uses a capitalized attribute name
        version = getattr(module, "__VERSION__", None)

    if version is None:
        if module.__name__ == "brotli":
            # brotli doesn't contain attributes to confirm it's version
            return ""
        if module.__name__ == "snappy":
            # snappy doesn't contain attributes to confirm it's version
            # See https://github.com/andrix/python-snappy/pull/119
            return ""
        raise ImportError(f"Can't determine version for {module.__name__}")
    if module.__name__ == "psycopg2":
        # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
        version = version.split()[0]
    return version


def import_optional_dependency(
    name: str,
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'``.
    """

    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError:
        if errors == "raise":
            raise ImportError(msg)
        else:
            return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return None
            elif errors == "raise":
                raise ImportError(msg)

    return module
