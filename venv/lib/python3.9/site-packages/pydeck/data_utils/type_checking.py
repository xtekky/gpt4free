def is_pandas_df(obj):
    """Check if an object is a Pandas DataFrame

    Returns
    -------
    bool
        Returns True if object is a Pandas DataFrame and False otherwise
    """
    return obj.__class__.__module__ == "pandas.core.frame" and obj.to_records and obj.to_dict


def has_geo_interface(obj):
    return hasattr(obj, "__geo_interface__")


def records_from_geo_interface(data):
    """Un-nest data from object implementing __geo_interface__ standard"""
    flattened_records = []
    for d in data.__geo_interface__.get("features"):
        record = d.get("properties", {})
        geom = d.get("geometry", {})
        record["geometry"] = geom
        flattened_records.append(record)
    return flattened_records
