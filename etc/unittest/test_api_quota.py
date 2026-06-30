from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

import g4f.api
from g4f import Provider

class TestApiQuota(unittest.TestCase):
    def setUp(self):
        # create fresh FastAPI app instance for each test
        self.app = g4f.api.create_app()
        self.client = TestClient(self.app)

    def test_nonexistent_provider_returns_404(self):
        resp = self.client.get("/api/NoSuchProvider/quota")
        self.assertEqual(resp.status_code, 404)

    def test_dummy_provider_quota_route(self):
        # monkeypatch a fake provider with async get_quota method
        class DummyProvider:
            async def get_quota(self, api_key=None):
                return {"foo": "bar"}

        Provider.__map__["dummy"] = DummyProvider()
        try:
            resp = self.client.get("/api/dummy/quota")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"foo": "bar"})
        finally:
            Provider.__map__.pop("dummy", None)
