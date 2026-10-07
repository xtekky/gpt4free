from __future__ import annotations

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
