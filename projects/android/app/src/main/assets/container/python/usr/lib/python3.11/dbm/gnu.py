"""Provide the _gdbm module as a dbm submodule."""

try:
    from _gdbm import *
except ImportError as msg:
    raise ImportError(str(msg) + ', please install the python3-gdbm package')
