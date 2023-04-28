"""Datasets for use with Altair (deprecated)

The functions here are merely thin wrappers for data access routines in the
vega_datasets package.
"""
from altair.utils.deprecation import deprecated


@deprecated("load_dataset is deprecated. " "Use the vega_datasets package instead.")
def load_dataset(name):
    """Load a dataset by name as a pandas.DataFrame."""
    import vega_datasets

    return vega_datasets.data(name)


@deprecated("load_dataset is deprecated. " "Use the vega_datasets package instead.")
def list_datasets():
    """List the available datasets."""
    import vega_datasets

    return vega_datasets.data.list_datasets()
