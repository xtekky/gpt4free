"""
Python representations of the JSON Schema Test Suite tests.
"""

from functools import partial
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import unittest

import attr

from jsonschema.validators import _VALIDATORS
import jsonschema


def _find_suite():
    root = os.environ.get("JSON_SCHEMA_TEST_SUITE")
    if root is not None:
        return Path(root)

    root = Path(jsonschema.__file__).parent.parent / "json"
    if not root.is_dir():  # pragma: no cover
        raise ValueError(
            (
                "Can't find the JSON-Schema-Test-Suite directory. "
                "Set the 'JSON_SCHEMA_TEST_SUITE' environment "
                "variable or run the tests from alongside a checkout "
                "of the suite."
            ),
        )
    return root


@attr.s(hash=True)
class Suite:

    _root = attr.ib(default=attr.Factory(_find_suite))

    def _remotes(self):
        jsonschema_suite = self._root.joinpath("bin", "jsonschema_suite")
        remotes = subprocess.check_output(
            [sys.executable, str(jsonschema_suite), "remotes"],
        )
        return json.loads(remotes.decode("utf-8"))

    def benchmark(self, runner):  # pragma: no cover
        for name, Validator in _VALIDATORS.items():
            self.version(name=name).benchmark(
                runner=runner,
                Validator=Validator,
            )

    def version(self, name):
        return Version(
            name=name,
            path=self._root.joinpath("tests", name),
            remotes=self._remotes(),
        )


@attr.s(hash=True)
class Version:

    _path = attr.ib()
    _remotes = attr.ib()

    name = attr.ib()

    def benchmark(self, runner, **kwargs):  # pragma: no cover
        for suite in self.tests():
            for test in suite:
                runner.bench_func(
                    test.fully_qualified_name,
                    partial(test.validate_ignoring_errors, **kwargs),
                )

    def tests(self):
        return (
            test
            for child in self._path.glob("*.json")
            for test in self._tests_in(
                subject=child.name[:-5],
                path=child,
            )
        )

    def format_tests(self):
        path = self._path.joinpath("optional", "format")
        return (
            test
            for child in path.glob("*.json")
            for test in self._tests_in(
                subject=child.name[:-5],
                path=child,
            )
        )

    def optional_tests_of(self, name):
        return self._tests_in(
            subject=name,
            path=self._path.joinpath("optional", name + ".json"),
        )

    def to_unittest_testcase(self, *suites, **kwargs):
        name = kwargs.pop("name", "Test" + self.name.title().replace("-", ""))
        methods = {
            test.method_name: test.to_unittest_method(**kwargs)
            for suite in suites
            for tests in suite
            for test in tests
        }
        cls = type(name, (unittest.TestCase,), methods)

        try:
            cls.__module__ = _someone_save_us_the_module_of_the_caller()
        except Exception:  # pragma: no cover
            # We're doing crazy things, so if they go wrong, like a function
            # behaving differently on some other interpreter, just make them
            # not happen.
            pass

        return cls

    def _tests_in(self, subject, path):
        for each in json.loads(path.read_text(encoding="utf-8")):
            yield (
                _Test(
                    version=self,
                    subject=subject,
                    case_description=each["description"],
                    schema=each["schema"],
                    remotes=self._remotes,
                    **test,
                ) for test in each["tests"]
            )


@attr.s(hash=True, repr=False)
class _Test:

    version = attr.ib()

    subject = attr.ib()
    case_description = attr.ib()
    description = attr.ib()

    data = attr.ib()
    schema = attr.ib(repr=False)

    valid = attr.ib()

    _remotes = attr.ib()

    comment = attr.ib(default=None)

    def __repr__(self):  # pragma: no cover
        return "<Test {}>".format(self.fully_qualified_name)

    @property
    def fully_qualified_name(self):  # pragma: no cover
        return " > ".join(
            [
                self.version.name,
                self.subject,
                self.case_description,
                self.description,
            ],
        )

    @property
    def method_name(self):
        delimiters = r"[\W\- ]+"
        return "test_{}_{}_{}".format(
            re.sub(delimiters, "_", self.subject),
            re.sub(delimiters, "_", self.case_description),
            re.sub(delimiters, "_", self.description),
        )

    def to_unittest_method(self, skip=lambda test: None, **kwargs):
        if self.valid:
            def fn(this):
                self.validate(**kwargs)
        else:
            def fn(this):
                with this.assertRaises(jsonschema.ValidationError):
                    self.validate(**kwargs)

        fn.__name__ = self.method_name
        reason = skip(self)
        if reason is None or os.environ.get("JSON_SCHEMA_DEBUG", "0") != "0":
            return fn
        elif os.environ.get("JSON_SCHEMA_EXPECTED_FAILURES", "0") != "0":
            return unittest.expectedFailure(fn)
        else:
            return unittest.skip(reason)(fn)

    def validate(self, Validator, **kwargs):
        Validator.check_schema(self.schema)
        resolver = jsonschema.RefResolver.from_schema(
            schema=self.schema,
            store=self._remotes,
            id_of=Validator.ID_OF,
        )

        # XXX: #693 asks to improve the public API for this, since yeah, it's
        #      bad. Figures that since it's hard for end-users, we experience
        #      the pain internally here too.
        def prevent_network_access(uri):
            raise RuntimeError(f"Tried to access the network: {uri}")
        resolver.resolve_remote = prevent_network_access

        validator = Validator(schema=self.schema, resolver=resolver, **kwargs)
        if os.environ.get("JSON_SCHEMA_DEBUG", "0") != "0":
            breakpoint()
        validator.validate(instance=self.data)

    def validate_ignoring_errors(self, Validator):  # pragma: no cover
        try:
            self.validate(Validator=Validator)
        except jsonschema.ValidationError:
            pass


def _someone_save_us_the_module_of_the_caller():
    """
    The FQON of the module 2nd stack frames up from here.

    This is intended to allow us to dynamically return test case classes that
    are indistinguishable from being defined in the module that wants them.

    Otherwise, trial will mis-print the FQON, and copy pasting it won't re-run
    the class that really is running.

    Save us all, this is all so so so so so terrible.
    """

    return sys._getframe(2).f_globals["__name__"]
