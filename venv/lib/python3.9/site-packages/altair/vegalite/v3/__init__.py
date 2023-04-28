# flake8: noqa
import warnings
from ...utils.deprecation import AltairDeprecationWarning

from .schema import *
from .api import *
from ._deprecated import *

from ...datasets import list_datasets, load_dataset

from ... import expr
from ...expr import datum

from .display import VegaLite, renderers

from .data import (
    MaxRowsError,
    pipe,
    curry,
    limit_rows,
    sample,
    to_json,
    to_csv,
    to_values,
    default_data_transformer,
    data_transformers,
)

warnings.warn(
    "The module altair.vegalite.v3 is deprecated and will be removed in Altair 5. "
    "Use `import altair as alt` instead of `import altair.vegalite.v3 as alt`.",
    AltairDeprecationWarning,
)
