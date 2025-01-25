import unittest

import g4f.version
from g4f.errors import VersionNotFoundError

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class TestGetLastProvider(unittest.TestCase):
    def test_get_latest_version(self):
        current_version = g4f.version.utils.current_version
        if current_version is not None:
            self.assertIsInstance(g4f.version.utils.current_version, str)
        self.assertIsInstance(g4f.version.utils.latest_version, str)