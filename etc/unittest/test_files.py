"""Tests for secure_filename sanitization."""

import unittest

from g4f.files import secure_filename


class SecureFilenameTests(unittest.TestCase):
    def test_reserved_windows_device_names(self):
        for name in (
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM9", "LPT1", "LPT9",
            "con", "aux", "Nul",
        ):
            with self.subTest(name=name):
                self.assertEqual(secure_filename(name), f"_{name}")

    def test_reserved_names_with_extension(self):
        self.assertEqual(secure_filename("CON.txt"), "_CON.txt")
        self.assertEqual(secure_filename("lpt9.log"), "_lpt9.log")
        self.assertEqual(secure_filename("NUL.tar.gz"), "_NUL.tar.gz")

    def test_non_reserved_names_are_unchanged(self):
        self.assertEqual(secure_filename("console.txt"), "console.txt")
        self.assertEqual(secure_filename("COM10"), "COM10")
        self.assertEqual(secure_filename("my_CONS"), "my_CONS")

    def test_existing_sanitization_still_applies(self):
        self.assertEqual(secure_filename("../../etc/passwd"), "etc_passwd")
        self.assertEqual(secure_filename("a b:c?.txt"), "a_b_c_.txt")
        self.assertIsNone(secure_filename(None))

    def test_max_length_is_respected_for_reserved_names(self):
        self.assertLessEqual(len(secure_filename("CON", max_length=8)), 8)


if __name__ == "__main__":
    unittest.main()
