from __future__ import annotations

import os
import tempfile
import unittest

from g4f.cli import get_api_parser, run_api_args
import g4f.cli as cli_mod
import g4f.cookies as cookies_mod


class TestCLI(unittest.TestCase):
    def test_api_parser_includes_cookies_dir(self):
        parser = get_api_parser(exit_on_error=False)
        args = parser.parse_args(["--cookies-dir", "/tmp/foo"])
        self.assertEqual(args.cookies_dir, "/tmp/foo")

    def test_run_api_args_sets_cookies_dir(self):
        # create a temporary directory to simulate user input
        tmpdir = tempfile.mkdtemp()
        parser = get_api_parser(exit_on_error=False)
        args = parser.parse_args(["--cookies-dir", tmpdir])

        called = {}

        # patch run_api so we don't actually start uvicorn
        orig_run = cli_mod.run_api
        orig_set = cookies_mod.set_cookies_dir
        try:
            cli_mod.run_api = lambda **kwargs: called.setdefault('ran', True)
            cookies_mod.set_cookies_dir = lambda d: called.setdefault('dir', d)

            run_api_args(args)

            self.assertTrue(called.get('ran'), "run_api should have been called")
            self.assertEqual(called.get('dir'), tmpdir,
                             "cookies directory should be passed to set_cookies_dir")
        finally:
            cli_mod.run_api = orig_run
            cookies_mod.set_cookies_dir = orig_set


if __name__ == "__main__":
    unittest.main()
