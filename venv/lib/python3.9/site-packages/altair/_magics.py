"""
Magic functions for rendering vega/vega-lite specifications
"""
__all__ = ["vega", "vegalite"]

import json
import warnings

import IPython
from IPython.core import magic_arguments
import pandas as pd
from toolz import curried

from altair.vegalite import v3 as vegalite_v3
from altair.vegalite import v4 as vegalite_v4
from altair.vega import v5 as vega_v5

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


RENDERERS = {
    "vega": {"5": vega_v5.Vega},
    "vega-lite": {"3": vegalite_v3.VegaLite, "4": vegalite_v4.VegaLite},
}


TRANSFORMERS = {
    "vega": {
        # Vega doesn't yet have specific data transformers; use vegalite
        "5": vegalite_v4.data_transformers,
    },
    "vega-lite": {
        "3": vegalite_v3.data_transformers,
        "4": vegalite_v4.data_transformers,
    },
}


def _prepare_data(data, data_transformers):
    """Convert input data to data for use within schema"""
    if data is None or isinstance(data, dict):
        return data
    elif isinstance(data, pd.DataFrame):
        return curried.pipe(data, data_transformers.get())
    elif isinstance(data, str):
        return {"url": data}
    else:
        warnings.warn("data of type {} not recognized".format(type(data)))
        return data


def _get_variable(name):
    """Get a variable from the notebook namespace."""
    ip = IPython.get_ipython()
    if ip is None:
        raise ValueError(
            "Magic command must be run within an IPython "
            "environemnt, in which get_ipython() is defined."
        )
    if name not in ip.user_ns:
        raise NameError(
            "argument '{}' does not match the "
            "name of any defined variable".format(name)
        )
    return ip.user_ns[name]


@magic_arguments.magic_arguments()
@magic_arguments.argument(
    "data",
    nargs="*",
    help="local variable name of a pandas DataFrame to be used as the dataset",
)
@magic_arguments.argument("-v", "--version", dest="version", default="5")
@magic_arguments.argument("-j", "--json", dest="json", action="store_true")
def vega(line, cell):
    """Cell magic for displaying Vega visualizations in CoLab.

    %%vega [name1:variable1 name2:variable2 ...] [--json] [--version='5']

    Visualize the contents of the cell using Vega, optionally specifying
    one or more pandas DataFrame objects to be used as the datasets.

    If --json is passed, then input is parsed as json rather than yaml.
    """
    args = magic_arguments.parse_argstring(vega, line)

    version = args.version
    assert version in RENDERERS["vega"]
    Vega = RENDERERS["vega"][version]
    data_transformers = TRANSFORMERS["vega"][version]

    def namevar(s):
        s = s.split(":")
        if len(s) == 1:
            return s[0], s[0]
        elif len(s) == 2:
            return s[0], s[1]
        else:
            raise ValueError("invalid identifier: '{}'".format(s))

    try:
        data = list(map(namevar, args.data))
    except ValueError:
        raise ValueError("Could not parse arguments: '{}'".format(line))

    if args.json:
        spec = json.loads(cell)
    elif not YAML_AVAILABLE:
        try:
            spec = json.loads(cell)
        except json.JSONDecodeError:
            raise ValueError(
                "%%vega: spec is not valid JSON. "
                "Install pyyaml to parse spec as yaml"
            )
    else:
        spec = yaml.load(cell, Loader=yaml.FullLoader)

    if data:
        spec["data"] = []
        for name, val in data:
            val = _get_variable(val)
            prepped = _prepare_data(val, data_transformers)
            prepped["name"] = name
            spec["data"].append(prepped)

    return Vega(spec)


@magic_arguments.magic_arguments()
@magic_arguments.argument(
    "data",
    nargs="?",
    help="local variablename of a pandas DataFrame to be used as the dataset",
)
@magic_arguments.argument("-v", "--version", dest="version", default="4")
@magic_arguments.argument("-j", "--json", dest="json", action="store_true")
def vegalite(line, cell):
    """Cell magic for displaying vega-lite visualizations in CoLab.

    %%vegalite [dataframe] [--json] [--version=3]

    Visualize the contents of the cell using Vega-Lite, optionally
    specifying a pandas DataFrame object to be used as the dataset.

    if --json is passed, then input is parsed as json rather than yaml.
    """
    args = magic_arguments.parse_argstring(vegalite, line)
    version = args.version
    assert version in RENDERERS["vega-lite"]
    VegaLite = RENDERERS["vega-lite"][version]
    data_transformers = TRANSFORMERS["vega-lite"][version]

    if args.json:
        spec = json.loads(cell)
    elif not YAML_AVAILABLE:
        try:
            spec = json.loads(cell)
        except json.JSONDecodeError:
            raise ValueError(
                "%%vegalite: spec is not valid JSON. "
                "Install pyyaml to parse spec as yaml"
            )
    else:
        spec = yaml.load(cell, Loader=yaml.FullLoader)

    if args.data is not None:
        data = _get_variable(args.data)
        spec["data"] = _prepare_data(data, data_transformers)

    return VegaLite(spec)
