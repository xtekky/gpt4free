import pytest
import altair.vegalite.v4 as alt


def geom_obj(geom):
    class Geom(object):
        pass

    geom_obj = Geom()
    setattr(geom_obj, "__geo_interface__", geom)
    return geom_obj


# correct translation of Polygon geometry to Feature type
def test_geo_interface_polygon_feature():
    geom = {
        "coordinates": [[(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]],
        "type": "Polygon",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"]["type"] == "Feature"


# merge geometry with empty properties dictionary
def test_geo_interface_removal_empty_properties():
    geom = {
        "geometry": {
            "coordinates": [
                [[6.90, 53.48], [5.98, 51.85], [6.07, 53.51], [6.90, 53.48]]
            ],
            "type": "Polygon",
        },
        "id": None,
        "properties": {},
        "type": "Feature",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"]["type"] == "Feature"


# only register metadata in the properties member
def test_geo_interface_register_foreign_member():
    geom = {
        "geometry": {
            "coordinates": [
                [[6.90, 53.48], [5.98, 51.85], [6.07, 53.51], [6.90, 53.48]]
            ],
            "type": "Polygon",
        },
        "id": 2,
        "properties": {"foo": "bah"},
        "type": "Feature",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    with pytest.raises(KeyError):
        spec["data"]["values"]["id"]
    assert spec["data"]["values"]["foo"] == "bah"


# correct serializing of arrays and nested tuples
def test_geo_interface_serializing_arrays_tuples():
    import array as arr

    geom = {
        "bbox": arr.array("d", [1, 2, 3, 4]),
        "geometry": {
            "coordinates": [
                tuple(
                    (
                        tuple((6.90, 53.48)),
                        tuple((5.98, 51.85)),
                        tuple((6.07, 53.51)),
                        tuple((6.90, 53.48)),
                    )
                )
            ],
            "type": "Polygon",
        },
        "id": 27,
        "properties": {},
        "type": "Feature",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"]["geometry"]["coordinates"][0][0] == [6.9, 53.48]


# overwrite existing 'type' value in properties with `Feature`
def test_geo_interface_reserved_members():
    geom = {
        "geometry": {
            "coordinates": [
                [[6.90, 53.48], [5.98, 51.85], [6.07, 53.51], [6.90, 53.48]]
            ],
            "type": "Polygon",
        },
        "id": 27,
        "properties": {"type": "foo"},
        "type": "Feature",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"]["type"] == "Feature"


# an empty FeatureCollection is valid
def test_geo_interface_empty_feature_collection():
    geom = {"type": "FeatureCollection", "features": []}
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"] == []


# Features in a FeatureCollection only keep properties and geometry
def test_geo_interface_feature_collection():
    geom = {
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": {
                    "coordinates": [
                        [[6.90, 53.48], [5.98, 51.85], [6.07, 53.51], [6.90, 53.48]]
                    ],
                    "type": "Polygon",
                },
                "id": 27,
                "properties": {"type": "foo", "id": 1, "geometry": 1},
                "type": "Feature",
            },
            {
                "geometry": {
                    "coordinates": [
                        [[8.90, 53.48], [7.98, 51.85], [8.07, 53.51], [8.90, 53.48]]
                    ],
                    "type": "Polygon",
                },
                "id": 28,
                "properties": {"type": "foo", "id": 2, "geometry": 1},
                "type": "Feature",
            },
        ],
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"][0]["id"] == 1
    assert spec["data"]["values"][1]["id"] == 2
    assert "coordinates" in spec["data"]["values"][0]["geometry"]
    assert "coordinates" in spec["data"]["values"][1]["geometry"]
    assert spec["data"]["values"][0]["type"] == "Feature"
    assert spec["data"]["values"][1]["type"] == "Feature"


# typical output of a __geo_interface__ from geopandas GeoDataFrame
# notic that the index value is registerd as a commonly used identifier
# with the name "id" (in this case 49). Similar to serialization of a
# pandas DataFrame is the index not included in the output
def test_geo_interface_feature_collection_gdf():
    geom = {
        "bbox": (19.89, -26.82, 29.43, -17.66),
        "features": [
            {
                "bbox": (19.89, -26.82, 29.43, -17.66),
                "geometry": {
                    "coordinates": [
                        [[6.90, 53.48], [5.98, 51.85], [6.07, 53.51], [6.90, 53.48]]
                    ],
                    "type": "Polygon",
                },
                "id": "49",
                "properties": {
                    "continent": "Africa",
                    "gdp_md_est": 35900.0,
                    "id": "BWA",
                    "iso_a3": "BWA",
                    "name": "Botswana",
                    "pop_est": 2214858,
                },
                "type": "Feature",
            }
        ],
        "type": "FeatureCollection",
    }
    feat = geom_obj(geom)

    with alt.data_transformers.enable(consolidate_datasets=False):
        spec = alt.Chart(feat).mark_geoshape().to_dict()
    assert spec["data"]["values"][0]["id"] == "BWA"
