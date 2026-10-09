from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

import g4f.api
from g4f.errors import MissingAuthError


class _DummyProvider:
    def __init__(self, quota=None, health=None, quota_error=None, health_error=None):
        self._quota = quota
        self._health = health
        self._quota_error = quota_error
        self._health_error = health_error

    async def get_quota(self, api_key=None):
        if self._quota_error:
            raise self._quota_error
        return self._quota

    async def get_health(self, api_key=None):
        if self._health_error:
            raise self._health_error
        return self._health


class _NoHealthProvider:
    async def get_quota(self, api_key=None):
        return {"foo": "bar"}


class TestApiQuota(unittest.TestCase):
    def setUp(self):
        # create fresh FastAPI app instance for each test
        self.app = g4f.api.create_app()
        self.client = TestClient(self.app)

    def test_nonexistent_provider_returns_404(self):
        resp = self.client.get("/api/NoSuchProvider/quota")
        self.assertEqual(resp.status_code, 404)

    def test_dummy_provider_quota_route(self):
        # monkeypatch the provider factory with a fake provider
        with patch(
            "g4f.api.AbstractClientFactory.create_provider",
            return_value=_DummyProvider(quota={"foo": "bar"}),
        ):
            resp = self.client.get("/api/dummy/quota")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"foo": "bar"})


class TestApiHealth(unittest.TestCase):
    def setUp(self):
        # create fresh FastAPI app instance for each test
        self.app = g4f.api.create_app()
        self.client = TestClient(self.app)

    def test_nonexistent_provider_returns_404(self):
        resp = self.client.get("/api/NoSuchProvider/health")
        self.assertEqual(resp.status_code, 404)

    def test_dummy_provider_health_route(self):
        with patch(
            "g4f.api.AbstractClientFactory.create_provider",
            return_value=_DummyProvider(health={"ok": True, "status": 200}),
        ):
            resp = self.client.get("/api/dummy/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True, "status": 200})

    def test_provider_without_health_returns_500(self):
        with patch(
            "g4f.api.AbstractClientFactory.create_provider",
            return_value=_NoHealthProvider(),
        ):
            resp = self.client.get("/api/dummy/health")
        self.assertEqual(resp.status_code, 500)

    def test_missing_auth_returns_401(self):
        with patch(
            "g4f.api.AbstractClientFactory.create_provider",
            return_value=_DummyProvider(health_error=MissingAuthError("API key is required.")),
        ):
            resp = self.client.get("/api/dummy/health")
        self.assertEqual(resp.status_code, 401)

    def test_not_implemented_returns_501(self):
        with patch(
            "g4f.api.AbstractClientFactory.create_provider",
            return_value=_DummyProvider(health_error=NotImplementedError("no health url")),
        ):
            resp = self.client.get("/api/dummy/health")
        self.assertEqual(resp.status_code, 501)

class TestOpenaiTemplateHealth(unittest.TestCase):
    """The /models fallback must never issue a chat completion request."""

    def _provider(self, **kwargs):
        from g4f.client.factory import create_custom_provider

        return create_custom_provider(
            base_url="https://upstream.example/v1",
            backup_url="https://proxy.example/v1",
            **kwargs,
        )

    def test_probes_backup_url_models_first(self):
        provider = self._provider()
        probed = []

        async def fake_probe(url, api_key=None):
            probed.append(url)
            return {"url": url, "status": 200, "ok": True, "quota": None}

        with patch.object(provider, "probe_health", staticmethod(fake_probe)):
            result = asyncio.run(provider.get_health())

        self.assertEqual(probed, ["https://proxy.example/v1/models"])
        self.assertTrue(result["ok"])

    def test_falls_back_to_base_url_models(self):
        provider = self._provider()
        probed = []

        async def fake_probe(url, api_key=None):
            probed.append(url)
            ok = url.startswith("https://upstream.example")
            return {"url": url, "status": 200 if ok else 402, "ok": ok, "quota": None}

        with patch.object(provider, "probe_health", staticmethod(fake_probe)):
            result = asyncio.run(provider.get_health())

        self.assertEqual(
            probed,
            ["https://proxy.example/v1/models", "https://upstream.example/v1/models"],
        )
        self.assertTrue(result["ok"])

    def test_health_url_wins_over_models_fallback(self):
        provider = self._provider(health_url="https://health.example/status")
        probed = []

        async def fake_probe(url, api_key=None):
            probed.append(url)
            return {"url": url, "status": 200, "ok": True, "quota": None}

        with patch.object(provider, "probe_health", staticmethod(fake_probe)):
            asyncio.run(provider.get_health())

        self.assertEqual(probed, ["https://health.example/status"])

    def test_no_urls_raises_not_implemented(self):
        from g4f.client.factory import create_custom_provider

        provider = create_custom_provider(base_url="")
        provider.base_url = ""
        provider.backup_url = None
        with self.assertRaises(NotImplementedError):
            asyncio.run(provider.get_health())
