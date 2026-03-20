import unittest

from g4f.Provider.needs_auth.mini_max.MiniMax import MiniMax


class TestMiniMaxProvider(unittest.TestCase):
    def test_default_model_is_m27(self):
        self.assertEqual(MiniMax.default_model, "MiniMax-M2.7")

    def test_m27_highspeed_in_models(self):
        self.assertIn("MiniMax-M2.7-highspeed", MiniMax.models)

    def test_m27_is_first_in_models(self):
        self.assertEqual(MiniMax.models[0], "MiniMax-M2.7")

    def test_m27_highspeed_is_second(self):
        self.assertEqual(MiniMax.models[1], "MiniMax-M2.7-highspeed")

    def test_old_models_still_available(self):
        self.assertIn("MiniMax-Text-01", MiniMax.models)
        self.assertIn("abab6.5s-chat", MiniMax.models)

    def test_model_alias_points_to_m27(self):
        self.assertEqual(MiniMax.model_aliases["MiniMax"], "MiniMax-M2.7")

    def test_provider_is_working(self):
        self.assertTrue(MiniMax.working)

    def test_needs_auth(self):
        self.assertTrue(MiniMax.needs_auth)


if __name__ == "__main__":
    unittest.main()
