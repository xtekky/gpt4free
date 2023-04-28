from collections import defaultdict

import numpy as np


# Grafted from
# https://github.com/maartenbreddels/ipyvolume/blob/d13828dfd8b57739004d5daf7a1d93ad0839ed0f/ipyvolume/serialize.py#L219
def array_to_binary(ar, obj=None, force_contiguous=True):
    if ar is None:
        return None
    if ar.dtype.kind not in ["u", "i", "f"]:  # ints and floats
        raise ValueError("unsupported dtype: %s" % (ar.dtype))
    # WebGL does not support float64, case it here
    if ar.dtype == np.float64:
        ar = ar.astype(np.float32)
    # JS does not support int64
    if ar.dtype == np.int64:
        ar = ar.astype(np.int32)
    # make sure it's contiguous
    if force_contiguous and not ar.flags["C_CONTIGUOUS"]:
        ar = np.ascontiguousarray(ar)
    return {
        # binary data representation of a numpy matrix
        "value": memoryview(ar),
        # dtype convertible to a typed array
        "dtype": str(ar.dtype),
        # height of np matrix
        "length": ar.shape[0],
        # width of np matrix
        "size": 1 if len(ar.shape) == 1 else ar.shape[1],
    }


def serialize_columns(data_set_cols, obj=None):
    if data_set_cols is None:
        return None
    layers = defaultdict(dict)
    # Number of records in data set
    length = {}
    for col in data_set_cols:
        accessor_attribute = array_to_binary(col["np_data"])
        if length.get(col["layer_id"]):
            length[col["layer_id"]] = max(length[col["layer_id"]], accessor_attribute["length"])
        else:
            length[col["layer_id"]] = accessor_attribute["length"]
        # attributes is deck.gl's expected argument name for
        # binary data transfer
        if not layers[col["layer_id"]].get("attributes"):
            layers[col["layer_id"]]["attributes"] = {}
        # Add new accessor
        layers[col["layer_id"]]["attributes"][col["accessor"]] = {
            "value": accessor_attribute["value"],
            "dtype": accessor_attribute["dtype"],
            "size": accessor_attribute["size"],
        }
    for layer_key, _ in layers.items():
        layers[layer_key]["length"] = length[layer_key]
    return layers


data_buffer_serialization = dict(to_json=serialize_columns, from_json=None)
