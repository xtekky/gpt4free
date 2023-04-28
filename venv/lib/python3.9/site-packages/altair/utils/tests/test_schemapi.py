# The contents of this file are automatically written by
# tools/generate_schema_wrapper.py. Do not modify directly.
import copy
import io
import json
import jsonschema
import pickle
import pytest

import numpy as np

from ..schemapi import (
    UndefinedType,
    SchemaBase,
    Undefined,
    _FromDict,
    SchemaValidationError,
)

# Make tests inherit from _TestSchema, so that when we test from_dict it won't
# try to use SchemaBase objects defined elsewhere as wrappers.


class _TestSchema(SchemaBase):
    @classmethod
    def _default_wrapper_classes(cls):
        return _TestSchema.__subclasses__()


class MySchema(_TestSchema):
    _schema = {
        "definitions": {
            "StringMapping": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "StringArray": {"type": "array", "items": {"type": "string"}},
        },
        "properties": {
            "a": {"$ref": "#/definitions/StringMapping"},
            "a2": {"type": "object", "additionalProperties": {"type": "number"}},
            "b": {"$ref": "#/definitions/StringArray"},
            "b2": {"type": "array", "items": {"type": "number"}},
            "c": {"type": ["string", "number"]},
            "d": {
                "anyOf": [
                    {"$ref": "#/definitions/StringMapping"},
                    {"$ref": "#/definitions/StringArray"},
                ]
            },
            "e": {"items": [{"type": "string"}, {"type": "string"}]},
        },
    }


class StringMapping(_TestSchema):
    _schema = {"$ref": "#/definitions/StringMapping"}
    _rootschema = MySchema._schema


class StringArray(_TestSchema):
    _schema = {"$ref": "#/definitions/StringArray"}
    _rootschema = MySchema._schema


class Derived(_TestSchema):
    _schema = {
        "definitions": {
            "Foo": {"type": "object", "properties": {"d": {"type": "string"}}},
            "Bar": {"type": "string", "enum": ["A", "B"]},
        },
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "string"},
            "c": {"$ref": "#/definitions/Foo"},
        },
    }


class Foo(_TestSchema):
    _schema = {"$ref": "#/definitions/Foo"}
    _rootschema = Derived._schema


class Bar(_TestSchema):
    _schema = {"$ref": "#/definitions/Bar"}
    _rootschema = Derived._schema


class SimpleUnion(_TestSchema):
    _schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}


class DefinitionUnion(_TestSchema):
    _schema = {"anyOf": [{"$ref": "#/definitions/Foo"}, {"$ref": "#/definitions/Bar"}]}
    _rootschema = Derived._schema


class SimpleArray(_TestSchema):
    _schema = {
        "type": "array",
        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
    }


class InvalidProperties(_TestSchema):
    _schema = {
        "type": "object",
        "properties": {"for": {}, "as": {}, "vega-lite": {}, "$schema": {}},
    }


def test_construct_multifaceted_schema():
    dct = {
        "a": {"foo": "bar"},
        "a2": {"foo": 42},
        "b": ["a", "b", "c"],
        "b2": [1, 2, 3],
        "c": 42,
        "d": ["x", "y", "z"],
        "e": ["a", "b"],
    }

    myschema = MySchema.from_dict(dct)
    assert myschema.to_dict() == dct

    myschema2 = MySchema(**dct)
    assert myschema2.to_dict() == dct

    assert isinstance(myschema.a, StringMapping)
    assert isinstance(myschema.a2, dict)
    assert isinstance(myschema.b, StringArray)
    assert isinstance(myschema.b2, list)
    assert isinstance(myschema.d, StringArray)


def test_schema_cases():
    assert Derived(a=4, b="yo").to_dict() == {"a": 4, "b": "yo"}
    assert Derived(a=4, c={"d": "hey"}).to_dict() == {"a": 4, "c": {"d": "hey"}}
    assert Derived(a=4, b="5", c=Foo(d="val")).to_dict() == {
        "a": 4,
        "b": "5",
        "c": {"d": "val"},
    }
    assert Foo(d="hello", f=4).to_dict() == {"d": "hello", "f": 4}

    assert Derived().to_dict() == {}
    assert Foo().to_dict() == {}

    with pytest.raises(jsonschema.ValidationError):
        # a needs to be an integer
        Derived(a="yo").to_dict()

    with pytest.raises(jsonschema.ValidationError):
        # Foo.d needs to be a string
        Derived(c=Foo(4)).to_dict()

    with pytest.raises(jsonschema.ValidationError):
        # no additional properties allowed
        Derived(foo="bar").to_dict()


def test_round_trip():
    D = {"a": 4, "b": "yo"}
    assert Derived.from_dict(D).to_dict() == D

    D = {"a": 4, "c": {"d": "hey"}}
    assert Derived.from_dict(D).to_dict() == D

    D = {"a": 4, "b": "5", "c": {"d": "val"}}
    assert Derived.from_dict(D).to_dict() == D

    D = {"d": "hello", "f": 4}
    assert Foo.from_dict(D).to_dict() == D


def test_from_dict():
    D = {"a": 4, "b": "5", "c": {"d": "val"}}
    obj = Derived.from_dict(D)
    assert obj.a == 4
    assert obj.b == "5"
    assert isinstance(obj.c, Foo)


def test_simple_type():
    assert SimpleUnion(4).to_dict() == 4


def test_simple_array():
    assert SimpleArray([4, 5, "six"]).to_dict() == [4, 5, "six"]
    assert SimpleArray.from_dict(list("abc")).to_dict() == list("abc")


def test_definition_union():
    obj = DefinitionUnion.from_dict("A")
    assert isinstance(obj, Bar)
    assert obj.to_dict() == "A"

    obj = DefinitionUnion.from_dict("B")
    assert isinstance(obj, Bar)
    assert obj.to_dict() == "B"

    obj = DefinitionUnion.from_dict({"d": "yo"})
    assert isinstance(obj, Foo)
    assert obj.to_dict() == {"d": "yo"}


def test_invalid_properties():
    dct = {"for": 2, "as": 3, "vega-lite": 4, "$schema": 5}
    invalid = InvalidProperties.from_dict(dct)
    assert invalid["for"] == 2
    assert invalid["as"] == 3
    assert invalid["vega-lite"] == 4
    assert invalid["$schema"] == 5
    assert invalid.to_dict() == dct


def test_undefined_singleton():
    assert Undefined is UndefinedType()


@pytest.fixture
def dct():
    return {
        "a": {"foo": "bar"},
        "a2": {"foo": 42},
        "b": ["a", "b", "c"],
        "b2": [1, 2, 3],
        "c": 42,
        "d": ["x", "y", "z"],
    }


def test_copy_method(dct):
    myschema = MySchema.from_dict(dct)

    # Make sure copy is deep
    copy = myschema.copy(deep=True)
    copy["a"]["foo"] = "new value"
    copy["b"] = ["A", "B", "C"]
    copy["c"] = 164
    assert myschema.to_dict() == dct

    # If we ignore a value, changing the copy changes the original
    copy = myschema.copy(deep=True, ignore=["a"])
    copy["a"]["foo"] = "new value"
    copy["b"] = ["A", "B", "C"]
    copy["c"] = 164
    mydct = myschema.to_dict()
    assert mydct["a"]["foo"] == "new value"
    assert mydct["b"][0] == dct["b"][0]
    assert mydct["c"] == dct["c"]

    # If copy is not deep, then changing copy below top level changes original
    copy = myschema.copy(deep=False)
    copy["a"]["foo"] = "baz"
    copy["b"] = ["A", "B", "C"]
    copy["c"] = 164
    mydct = myschema.to_dict()
    assert mydct["a"]["foo"] == "baz"
    assert mydct["b"] == dct["b"]
    assert mydct["c"] == dct["c"]


def test_copy_module(dct):
    myschema = MySchema.from_dict(dct)

    cp = copy.deepcopy(myschema)
    cp["a"]["foo"] = "new value"
    cp["b"] = ["A", "B", "C"]
    cp["c"] = 164
    assert myschema.to_dict() == dct


def test_attribute_error():
    m = MySchema()
    with pytest.raises(AttributeError) as err:
        m.invalid_attribute
    assert str(err.value) == (
        "'MySchema' object has no attribute " "'invalid_attribute'"
    )


def test_to_from_json(dct):
    json_str = MySchema.from_dict(dct).to_json()
    new_dct = MySchema.from_json(json_str).to_dict()

    assert new_dct == dct


def test_to_from_pickle(dct):
    myschema = MySchema.from_dict(dct)
    output = io.BytesIO()
    pickle.dump(myschema, output)
    output.seek(0)
    myschema_new = pickle.load(output)

    assert myschema_new.to_dict() == dct


def test_class_with_no_schema():
    class BadSchema(SchemaBase):
        pass

    with pytest.raises(ValueError) as err:
        BadSchema(4)
    assert str(err.value).startswith("Cannot instantiate object")


@pytest.mark.parametrize("use_json", [True, False])
def test_hash_schema(use_json):
    classes = _TestSchema._default_wrapper_classes()

    for cls in classes:
        hsh1 = _FromDict.hash_schema(cls._schema, use_json=use_json)
        hsh2 = _FromDict.hash_schema(cls._schema, use_json=use_json)
        assert hsh1 == hsh2
        assert hash(hsh1) == hash(hsh2)


def test_schema_validation_error():
    try:
        MySchema(a={"foo": 4})
        the_err = None
    except jsonschema.ValidationError as err:
        the_err = err

    assert isinstance(the_err, SchemaValidationError)
    message = str(the_err)

    assert message.startswith("Invalid specification")
    assert "test_schemapi.MySchema->a" in message
    assert "validating {!r}".format(the_err.validator) in message
    assert the_err.message in message


def test_serialize_numpy_types():
    m = MySchema(
        a={"date": np.datetime64("2019-01-01")},
        a2={"int64": np.int64(1), "float64": np.float64(2)},
        b2=np.arange(4),
    )
    out = m.to_json()
    dct = json.loads(out)
    assert dct == {
        "a": {"date": "2019-01-01T00:00:00"},
        "a2": {"int64": 1, "float64": 2},
        "b2": [0, 1, 2, 3],
    }
