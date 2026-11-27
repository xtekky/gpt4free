"""Provide the _dbm module as a dbm submodule."""

try:
    from _dbm import *
except ImportError as msg:
    raise ImportError(str(msg) + ', please install the python3-gdbm package')
