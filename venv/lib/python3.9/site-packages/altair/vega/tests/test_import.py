from altair.vega import SCHEMA_VERSION, SCHEMA_URL


def test_schema_version():
    assert SCHEMA_VERSION in SCHEMA_URL
