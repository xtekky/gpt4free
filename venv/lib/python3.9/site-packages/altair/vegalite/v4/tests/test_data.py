import os

import pandas as pd
import pytest

from .. import data as alt


@pytest.fixture
def sample_data():
    return pd.DataFrame({"x": range(10), "y": range(10)})


def test_disable_max_rows(sample_data):
    with alt.data_transformers.enable("default", max_rows=5):
        # Ensure max rows error is raised.
        with pytest.raises(alt.MaxRowsError):
            alt.data_transformers.get()(sample_data)

        # Ensure that max rows error is properly disabled.
        with alt.data_transformers.disable_max_rows():
            alt.data_transformers.get()(sample_data)

    try:
        with alt.data_transformers.enable("json"):
            # Ensure that there is no TypeError for non-max_rows transformers.
            with alt.data_transformers.disable_max_rows():
                jsonfile = alt.data_transformers.get()(sample_data)
    except TypeError:
        jsonfile = {}
    finally:
        if jsonfile:
            os.remove(jsonfile["url"])
