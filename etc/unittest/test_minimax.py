import unittest

from g4f.Provider.needs_auth.mini_max.MiniMax import MiniMax


class TestMiniMaxProvider(unittest.TestCase):
    def test_default_model_is_m3(self):
        self.assertEqual(MiniMax.default_model, "MiniMax-M3")

    def test_m3_in_models(self):
        self.assertIn("MiniMax-M3", MiniMax.models)

    def test_m3_is_first_in_models(self):
        self.assertEqual(MiniMax.models[0], "MiniMax-M3")

    def test_m27_still_available(self):
        self.assertIn("MiniMax-M2.7", MiniMax.models)

    def test_m27_highspeed_still_available(self):
        self.assertIn("MiniMax-M2.7-highspeed", MiniMax.models)

    def test_old_models_removed(self):
        self.assertNotIn("MiniMax-Text-01", MiniMax.models)
        self.assertNotIn("abab6.5s-chat", MiniMax.models)
        self.assertNotIn("MiniMax-M2.5", MiniMax.models)
        self.assertNotIn("MiniMax-M2.1", MiniMax.models)
        self.assertNotIn("MiniMax-M2", MiniMax.models)
        self.assertNotIn("MiniMax-M1", MiniMax.models)

    def test_model_alias_points_to_m3(self):
        self.assertEqual(MiniMax.model_aliases["MiniMax"], "MiniMax-M3")

    def test_provider_is_working(self):
        self.assertTrue(MiniMax.working)

    def test_needs_auth(self):
        self.assertTrue(MiniMax.needs_auth)


if __name__ == "__main__":
    unittest.main()
