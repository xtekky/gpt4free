import unittest

class TestImport(unittest.TestCase):

    def test_get_cookies(self):
        from g4f import get_cookies as get_cookies_alias
        from g4f.cookies import get_cookies
        self.assertEqual(get_cookies_alias, get_cookies)

    def test_requests(self):
        from g4f.requests import StreamSession
        self.assertIsInstance(StreamSession, type)

if __name__ == '__main__':
    unittest.main()