from ..data import (
    MaxRowsError,
    curry,
    default_data_transformer,
    limit_rows,
    pipe,
    sample,
    to_csv,
    to_json,
    to_values,
    DataTransformerRegistry,
)


# ==============================================================================
# VegaLite 3 data transformers
# ==============================================================================


ENTRY_POINT_GROUP = "altair.vegalite.v3.data_transformer"  # type: str


data_transformers = DataTransformerRegistry(
    entry_point_group=ENTRY_POINT_GROUP
)  # type: DataTransformerRegistry
data_transformers.register("default", default_data_transformer)
data_transformers.register("json", to_json)
data_transformers.register("csv", to_csv)
data_transformers.enable("default")


__all__ = (
    "MaxRowsError",
    "curry",
    "default_data_transformer",
    "limit_rows",
    "pipe",
    "sample",
    "to_csv",
    "to_json",
    "to_values",
    "data_transformers",
)
