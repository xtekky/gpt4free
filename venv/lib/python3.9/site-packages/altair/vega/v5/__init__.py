# flake8: noqa
import warnings
from ...utils.deprecation import AltairDeprecationWarning
from .display import vega, Vega, renderers
from .schema import *

from .data import (
    pipe,
    curry,
    limit_rows,
    sample,
    to_json,
    to_csv,
    to_values,
    default_data_transformer,
)

warnings.warn(
    "The module altair.vega.v5 is deprecated and will be removed in Altair 5.",
    AltairDeprecationWarning,
)
