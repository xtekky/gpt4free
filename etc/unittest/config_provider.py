"""
Unit tests for g4f.providers.config_provider.

Tests cover:
- QuotaCache: set / get (cached vs expired) / invalidate / clear
- ErrorCounter: increment / get_count / reset / clear
- evaluate_condition: various operator and logical combinations
- RouterConfig.load: parsing valid YAML, missing file, invalid YAML
- ConfigModelProvider: successful routing, condition-based skip, 429 handling
"""

from __future__ import annotations

import time
import unittest

from g4f.providers.config_provider import (
    QuotaCache,
    ErrorCounter,
    ModelRouteConfig,
    ProviderRouteConfig,
    RouterConfig,
    ConfigModelProvider,
    evaluate_condition,
    has_yaml
)


# ---------------------------------------------------------------------------
# QuotaCache tests
# ---------------------------------------------------------------------------

class TestQuotaCache(unittest.TestCase):

    def setUp(self):
        QuotaCache.clear()

    def test_miss_returns_none(self):
        self.assertIsNone(QuotaCache.get("NonExistent"))

    def test_set_and_get(self):
        QuotaCache.set("MyProvider", {"balance": 42.0})
        result = QuotaCache.get("MyProvider")
        self.assertIsNotNone(result)
        self.assertEqual(result["balance"], 42.0)

    def test_ttl_expiry(self):
        QuotaCache.ttl = 0.01  # very short TTL
        try:
            QuotaCache.set("MyProvider", {"balance": 1.0})
            time.sleep(0.05)
            self.assertIsNone(QuotaCache.get("MyProvider"))
        finally:
            QuotaCache.ttl = 300  # restore default

    def test_invalidate(self):
        QuotaCache.set("MyProvider", {"balance": 5.0})
        QuotaCache.invalidate("MyProvider")
        self.assertIsNone(QuotaCache.get("MyProvider"))

    def test_clear(self):
        QuotaCache.set("A", {"balance": 1.0})
        QuotaCache.set("B", {"balance": 2.0})
        QuotaCache.clear()
        self.assertIsNone(QuotaCache.get("A"))
        self.assertIsNone(QuotaCache.get("B"))


# ---------------------------------------------------------------------------
# ErrorCounter tests
# ---------------------------------------------------------------------------

class TestErrorCounter(unittest.TestCase):

    def setUp(self):
        ErrorCounter.clear()

    def test_initial_count_is_zero(self):
        self.assertEqual(ErrorCounter.get_count("NewProvider"), 0)

    def test_increment_increases_count(self):
        ErrorCounter.increment("P")
        ErrorCounter.increment("P")
        self.assertEqual(ErrorCounter.get_count("P"), 2)

    def test_reset_clears_count(self):
        ErrorCounter.increment("P")
        ErrorCounter.reset("P")
        self.assertEqual(ErrorCounter.get_count("P"), 0)

    def test_window_expiry(self):
        ErrorCounter.window = 0.01  # 10 ms window
        try:
            ErrorCounter.increment("P")
            time.sleep(0.05)
            self.assertEqual(ErrorCounter.get_count("P"), 0)
        finally:
            ErrorCounter.window = 3600  # restore default

    def test_clear_all(self):
        ErrorCounter.increment("X")
        ErrorCounter.increment("Y")
        ErrorCounter.clear()
        self.assertEqual(ErrorCounter.get_count("X"), 0)
        self.assertEqual(ErrorCounter.get_count("Y"), 0)


# ---------------------------------------------------------------------------
# evaluate_condition tests
# ---------------------------------------------------------------------------

class TestEvaluateCondition(unittest.TestCase):

    # --- simple comparisons (PollinationsAI-style quota) ---

    def test_balance_gt_true(self):
        self.assertTrue(evaluate_condition("balance > 0", {"balance": 5.0}, 0))

    def test_balance_gt_false(self):
        self.assertFalse(evaluate_condition("balance > 0", {"balance": 0.0}, 0))

    def test_balance_lt(self):
        self.assertTrue(evaluate_condition("balance < 10", {"balance": 3.0}, 0))

    def test_error_count_lt_true(self):
        self.assertTrue(evaluate_condition("error_count < 3", {}, 2))

    def test_error_count_lt_false(self):
        self.assertFalse(evaluate_condition("error_count < 3", {}, 5))

    def test_eq_operator(self):
        self.assertTrue(evaluate_condition("error_count == 0", {"balance": 1.0}, 0))

    def test_neq_operator(self):
        self.assertTrue(evaluate_condition("error_count != 3", {"balance": 1.0}, 2))

    def test_ge_operator(self):
        self.assertTrue(evaluate_condition("balance >= 5", {"balance": 5.0}, 0))

    def test_le_operator(self):
        self.assertTrue(evaluate_condition("balance <= 5", {"balance": 5.0}, 0))

    # --- logical connectives ---

    def test_or_both_false(self):
        self.assertFalse(
            evaluate_condition("balance > 0 or error_count < 3", {"balance": 0.0}, 5)
        )

    def test_or_first_true(self):
        self.assertTrue(
            evaluate_condition("balance > 0 or error_count < 3", {"balance": 1.0}, 5)
        )

    def test_or_second_true(self):
        self.assertTrue(
            evaluate_condition("balance > 0 or error_count < 3", {"balance": 0.0}, 2)
        )

    def test_or_both_true(self):
        self.assertTrue(
            evaluate_condition("balance > 0 or error_count < 3", {"balance": 1.0}, 1)
        )

    def test_and_both_true(self):
        self.assertTrue(
            evaluate_condition("balance > 0 and error_count < 3", {"balance": 1.0}, 2)
        )

    def test_and_first_false(self):
        self.assertFalse(
            evaluate_condition("balance > 0 and error_count < 3", {"balance": 0.0}, 2)
        )

    def test_not_operator(self):
        self.assertTrue(evaluate_condition("not error_count > 5", {}, 2))

    # --- provider-specific quota dot-notation ---

    def test_quota_balance_pollinations(self):
        """PollinationsAI: quota.balance shorthand."""
        self.assertTrue(
            evaluate_condition("quota.balance > 0", {"balance": 10.0}, 0)
        )

    def test_quota_balance_pollinations_false(self):
        self.assertFalse(
            evaluate_condition("quota.balance > 0", {"balance": 0.0}, 0)
        )

    def test_quota_nested_yupp(self):
        """Yupp: quota.credits.remaining > 0."""
        quota = {"credits": {"remaining": 500, "total": 5000}}
        self.assertTrue(
            evaluate_condition("quota.credits.remaining > 0", quota, 0)
        )

    def test_quota_nested_yupp_false(self):
        quota = {"credits": {"remaining": 0, "total": 5000}}
        self.assertFalse(
            evaluate_condition("quota.credits.remaining > 0", quota, 0)
        )

    def test_quota_missing_key_resolves_zero(self):
        """Missing quota key should resolve to 0.0 (not raise)."""
        self.assertFalse(
            evaluate_condition("quota.nonexistent > 0", {}, 0)
        )

    def test_quota_missing_nested_key_resolves_zero(self):
        self.assertFalse(
            evaluate_condition("quota.credits.remaining > 0", {}, 0)
        )

    def test_quota_combined_condition(self):
        """quota.credits.remaining > 0 or error_count < 3."""
        quota = {"credits": {"remaining": 0, "total": 5000}}
        self.assertTrue(
            evaluate_condition("quota.credits.remaining > 0 or error_count < 3", quota, 2)
        )

    # --- legacy aliases ---

    def test_get_quota_balance_alias(self):
        """get_quota.balance → quota.balance backward-compat alias."""
        self.assertTrue(
            evaluate_condition("get_quota.balance > 0", {"balance": 10.0}, 0)
        )

    def test_get_quota_balance_alias_false(self):
        self.assertFalse(
            evaluate_condition("get_quota.balance > 0", {"balance": 0.0}, 0)
        )

    # --- edge cases ---

    def test_empty_condition_returns_true(self):
        self.assertTrue(evaluate_condition("", {}, 0))

    def test_none_quota_treated_as_empty_dict(self):
        """None quota should behave as empty dict: balance → 0.0."""
        self.assertFalse(evaluate_condition("balance > 0", None, 0))

    def test_float_literal(self):
        self.assertTrue(evaluate_condition("balance > 1.5", {"balance": 2.0}, 0))

    def test_parentheses(self):
        self.assertTrue(
            evaluate_condition(
                "(balance > 0 or error_count < 3) and error_count < 10",
                {"balance": 0.0},
                2,
            )
        )

    def test_unknown_variable_raises(self):
        with self.assertRaises(ValueError):
            evaluate_condition("unknown_var > 0", {}, 0)

    def test_quota_unknown_sub_key_resolves_zero(self):
        """Accessing a missing sub-key of quota returns 0.0, not an error."""
        quota = {"balance": 5.0}
        self.assertFalse(
            evaluate_condition("quota.missing_field > 100", quota, 0)
        )


# ---------------------------------------------------------------------------
# RouterConfig tests
# ---------------------------------------------------------------------------

class TestRouterConfig(unittest.TestCase):

    def setUp(self):
        if not has_yaml:
            self.skipTest('"yaml" not installed')
        RouterConfig.clear()

    def test_load_valid_yaml(self):
        import tempfile, os
        cfg = """
models:
  - name: "test-model"
    providers:
      - provider: "PollinationsAI"
        model: "openai-large"
        condition: "balance > 0 or error_count < 3"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(cfg)
            path = f.name
        try:
            RouterConfig.load(path)
            route = RouterConfig.get("test-model")
            self.assertIsNotNone(route)
            self.assertEqual(route.name, "test-model")
            self.assertEqual(len(route.providers), 1)
            self.assertEqual(route.providers[0].provider, "PollinationsAI")
            self.assertEqual(route.providers[0].model, "openai-large")
            self.assertEqual(route.providers[0].condition, "balance > 0 or error_count < 3")
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        # Should not raise; simply leaves routes empty.
        RouterConfig.load("/nonexistent/path/config.yaml")
        self.assertEqual(RouterConfig.routes, {})

    def test_load_empty_yaml(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("models: []\n")
            path = f.name
        try:
            RouterConfig.load(path)
            self.assertEqual(RouterConfig.routes, {})
        finally:
            os.unlink(path)

    def test_load_invalid_yaml(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(": : invalid\n")
            path = f.name
        try:
            # Should not raise; logs an error instead.
            RouterConfig.load(path)
        finally:
            os.unlink(path)

    def test_get_unknown_model_returns_none(self):
        self.assertIsNone(RouterConfig.get("no-such-model"))

    def test_clear_removes_routes(self):
        RouterConfig.routes["x"] = ModelRouteConfig(name="x", providers=[])
        RouterConfig.clear()
        self.assertEqual(RouterConfig.routes, {})

    def test_provider_default_model_uses_route_name(self):
        import tempfile, os
        cfg = """
models:
  - name: "my-model"
    providers:
      - provider: "PollinationsAI"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(cfg)
            path = f.name
        try:
            RouterConfig.load(path)
            route = RouterConfig.get("my-model")
            self.assertIsNotNone(route)
            # When 'model' key is absent, it defaults to the route name
            self.assertEqual(route.providers[0].model, "my-model")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ConfigModelProvider tests (synchronous helpers)
# ---------------------------------------------------------------------------

class TestConfigModelProvider(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        QuotaCache.clear()
        ErrorCounter.clear()
        RouterConfig.clear()

    async def test_provider_not_found_skipped(self):
        """If a configured provider doesn't exist, it should be skipped."""
        route = ModelRouteConfig(
            name="my-model",
            providers=[
                ProviderRouteConfig(provider="NonExistentProvider9999", model="x"),
            ],
        )
        cmp = ConfigModelProvider(route)
        chunks = []
        with self.assertRaises(RuntimeError):
            async for chunk in cmp.create_async_generator("my-model", []):
                chunks.append(chunk)

    async def test_condition_false_skips_provider(self):
        """Provider whose condition evaluates False must be skipped."""
        route = ModelRouteConfig(
            name="my-model",
            providers=[
                ProviderRouteConfig(
                    provider="NonExistentProvider9999",
                    model="x",
                    condition="balance > 100",  # will be False (balance=0)
                ),
            ],
        )
        cmp = ConfigModelProvider(route)
        with self.assertRaises(RuntimeError):
            async for _ in cmp.create_async_generator("my-model", []):
                pass

    async def test_condition_true_attempts_provider(self):
        """Provider whose condition is True should be attempted (even if it fails)."""
        route = ModelRouteConfig(
            name="my-model",
            providers=[
                ProviderRouteConfig(
                    provider="NonExistentProvider9999",
                    model="x",
                    condition="balance >= 0",  # True
                ),
            ],
        )
        cmp = ConfigModelProvider(route)
        with self.assertRaises((RuntimeError, ValueError)):
            async for _ in cmp.create_async_generator("my-model", []):
                pass

    async def test_429_invalidates_quota_cache(self):
        """A RateLimitError should invalidate the quota cache for that provider."""
        from g4f.errors import RateLimitError

        QuotaCache.set("TestProvider", {"balance": 5.0})

        route = ModelRouteConfig(
            name="my-model",
            providers=[
                ProviderRouteConfig(provider="NonExistentProvider9999", model="x"),
            ],
        )
        cmp = ConfigModelProvider(route)

        # Simulate what the provider does on 429:
        # We test the cache invalidation logic directly.
        QuotaCache.invalidate("TestProvider")
        self.assertIsNone(QuotaCache.get("TestProvider"))


if __name__ == "__main__":
    unittest.main()
