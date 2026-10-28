import unittest

import g4f.version
from g4f.errors import VersionNotFoundError

DEFAULT_MESSAGES = [{"role": "user", "content": "Hello"}]


class TestGetLastProvider(unittest.TestCase):
    def test_get_latest_version(self):
        current_version = g4f.version.utils.current_version
        if current_version is not None:
            self.assertIsInstance(g4f.version.utils.current_version, str)
        try:
            latest_version = g4f.version.utils.latest_version
        except VersionNotFoundError:
            return
        if latest_version is None:
            # Without package metadata, git tags or network access the version
            # cannot be resolved, so there is nothing to assert.
            self.skipTest("latest version is unavailable")
        self.assertIsInstance(latest_version, str)
