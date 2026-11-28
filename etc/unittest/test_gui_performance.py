import gzip
import os
import tempfile
import unittest
from unittest.mock import patch

from flask import Response

from g4f.gui import get_gui_app
from g4f.gui.server import website
from g4f.gui.server.app import create_app


class TestGuiCompression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_gui_app()
        cls.app.testing = True

    def setUp(self):
        self.client = self.app.test_client()

    def test_chat_page_is_gzipped(self):
        response = self.client.get(
            "/chat/", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Encoding"), "gzip")
        self.assertIn("Accept-Encoding", response.headers.get("Vary", ""))
        self.assertIn(b"<!DOCTYPE html>", gzip.decompress(response.data))

    def test_assets_are_gzipped(self):
        response = self.client.get(
            "/dist/css/style.css", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Encoding"), "gzip")
        self.assertGreater(len(gzip.decompress(response.data)), len(response.data))

    def test_no_compression_without_accept_encoding(self):
        response = self.client.get("/chat/")
        self.assertIsNone(response.headers.get("Content-Encoding"))

    def test_streaming_responses_are_not_buffered(self):
        app = create_app(True)
        app.testing = True

        @app.route("/stream")
        def stream():
            return Response(
                (chunk for chunk in (b"event: ping\n\n", b"event: done\n\n")),
                mimetype="text/event-stream",
            )

        response = app.test_client().get(
            "/stream", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("Content-Encoding"))
        self.assertEqual(response.data, b"event: ping\n\nevent: done\n\n")


class TestChatPageCaching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_gui_app()
        cls.app.testing = True

    def setUp(self):
        self.client = self.app.test_client()
        self._tmp = tempfile.TemporaryDirectory()
        self.dist = os.path.join(self._tmp.name, "dist")
        os.makedirs(self.dist)
        os.makedirs(os.path.join(self._tmp.name, "chat"))
        with open(
            os.path.join(self._tmp.name, "chat", "index.html"), "w", encoding="utf-8"
        ) as file:
            file.write("<html><body>chat {{ v }}</body></html>")
        self._patches = [
            patch.object(website, "DIST_DIR", self.dist),
            patch.object(website, "_latest_version", return_value="9.9.9"),
        ]
        for item in self._patches:
            item.start()

    def tearDown(self):
        for item in self._patches:
            item.stop()
        self._tmp.cleanup()

    def test_local_chat_page_has_revalidatable_cache(self):
        response = self.client.get("/chat/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("9.9.9", response.get_data(as_text=True))
        self.assertIn("must-revalidate", response.headers["Cache-Control"])
        etag = response.headers.get("ETag")
        self.assertTrue(etag)

    def test_conditional_request_returns_304(self):
        first = self.client.get("/chat/")
        response = self.client.get(
            "/chat/", headers={"If-None-Match": first.headers["ETag"]}
        )
        self.assertEqual(response.status_code, 304)
        self.assertEqual(response.data, b"")

    def test_index_passes_through_304(self):
        # "/" is decorated with SEO metadata, but a conditional 304 has no
        # body to decorate and must not be turned into an empty 200 page.
        with patch.object(
            website, "render", return_value=Response("", status=304)
        ):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 304)


class TestLatestVersionCache(unittest.TestCase):
    def test_latest_version_is_cached(self):
        calls = []

        class FakeVersion:
            @property
            def latest_version(self):
                calls.append(True)
                return "1.2.3"

        with patch.object(website, "version") as version_module, \
             patch.object(website, "_latest_version_cache", None):
            version_module.utils = FakeVersion()
            self.assertEqual(website._latest_version(), "1.2.3")
            self.assertEqual(website._latest_version(), "1.2.3")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
