import unittest
from unittest.mock import patch

import requests
from flask import Response

from g4f.gui import get_gui_app
from g4f.gui.server.locale import normalize_text, translate_text


DOCUMENT = '''<!DOCTYPE html><html lang="en"><head><title>Original</title>
<meta name="description" content="Old">
<link rel="canonical" href="https://g4f.dev/">
</head><body><h1>G4F AI Tools & Dashboard</h1><a href="/">Home</a>
<a href="/providers/">Providers</a><script src="/dist/app.js"></script></body></html>'''


class TestGuiLocale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_gui_app()
        cls.app.testing = True

    def setUp(self):
        self.client = self.app.test_client()

    def test_localized_proxied_html_and_meta(self):
        with patch("g4f.gui.server.website.render", return_value=DOCUMENT):
            response = self.client.get("/es/home.html")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('<html lang="es">', html)
        self.assertNotIn("<base", html)
        self.assertIn('href="https://g4f.space/es/"', html)
        self.assertIn('hreflang="fr" href="https://g4f.space/fr/"', html)
        self.assertIn("Herramientas de IA de G4F", html)
        self.assertIn('href="/es/providers/"', html)
        self.assertIn('href="/es/"', html)
        self.assertIn('src="/es/dist/app.js"', html)
        self.assertNotIn('href="/es/es/', html)
        self.assertIn("loadGlobalTranslations", html)
        self.assertIn("data-g4f-locale-runtime", html)
        self.assertIn("setSelectedLanguage", html)
        self.assertNotIn('href="https://g4f.dev/"', html)
        self.assertEqual(response.headers["Content-Language"], "es")

        meta = self.client.get("/fr/meta").json
        self.assertEqual(meta["locale"], "fr")
        self.assertEqual(meta["routes"]["home"]["url"], "https://g4f.space/fr/")
        self.assertEqual(meta["routes"]["providers"]["alternates"]["en"],
                         "https://g4f.space/providers/")
        self.assertEqual(meta["routes"]["chat"]["url"], "https://g4f.space/fr/chat/")
        sitemap = self.client.get("/sitemap.xml").get_data(as_text=True)
        self.assertIn("https://g4f.space/fr/chat/", sitemap)
        self.assertIn('hreflang="de"', sitemap)

    def test_chat_playground_and_index_routes(self):
        with patch("g4f.gui.server.website.render", return_value=DOCUMENT):
            for path in ("/es/chat/", "/es/index.html"):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, path)
                html = response.get_data(as_text=True)
                self.assertNotIn("<base", html)
                self.assertIn('href="/es/providers/"', html)

        response = self.client.get("/de/playground/")
        try:
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertIn('<html lang="de">', html)
            self.assertNotIn("<base", html)
            self.assertIn("KI-Playground", html)
            self.assertNotIn('href="/css/', html)
        finally:
            response.close()

    def test_index_handles_prepared_response(self):
        # render() returns a prepared Response for pages served from the
        # local dist directory or the download cache. That response used to
        # reach seo_document() unconverted, raising a TypeError and turning
        # "/" into HTTP 500 on every launch after the page was cached.
        cached = Response(DOCUMENT, mimetype="text/html")
        cached.headers["Cache-Control"] = "public, max-age=31536000"
        with patch("g4f.gui.server.website.render", return_value=cached):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("G4F AI Tools & Dashboard", html)
        self.assertIn('<html lang="en">', html)
        self.assertEqual(
            response.headers["Cache-Control"], "public, max-age=31536000"
        )

    def test_invalid_locales_and_pages(self):
        self.assertEqual(self.client.get("/xx/home.html").status_code, 404)
        with patch(
            "g4f.gui.server.website.render",
            side_effect=requests.RequestException("missing upstream page"),
        ):
            self.assertEqual(self.client.get("/fr/unknown.html").status_code, 404)
        self.assertEqual(self.client.get("/xx/meta").status_code, 404)

    def test_provider_page_is_localized(self):
        with patch("g4f.gui.server.website.Website._get_providers", return_value=[]):
            response = self.client.get("/de/providers.html")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('<html lang="de">', html)
        self.assertIn("Verfügbare Anbieter", html)
        self.assertIn('rel="canonical" href="https://g4f.space/de/providers/"', html)
        self.assertIn('href="/de/providers/"', html)
        self.assertNotIn("<base", html)

    def test_fragment_and_query_links_stay_on_current_page(self):
        document = DOCUMENT.replace(
            '<a href="/">Home</a>',
            '<a href="/">Home</a><a href="#login">Login</a><a href="?q=1">Query</a>',
        )
        with patch("g4f.gui.server.website.render", return_value=document):
            response = self.client.get("/de/chat/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        # Without a <base> these must resolve against /de/chat/, not /de/.
        self.assertIn('href="#login"', html)
        self.assertIn('href="?q=1"', html)
        self.assertNotIn("<base", html)

    def test_headers_and_country_ip(self):
        response = self.client.get("/headers", headers={"Authorization": "secret", "X-Test": "ok"})
        self.assertEqual(response.json["Authorization"], "[redacted]")
        self.assertEqual(response.json["X-Test"], "ok")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

        response = self.client.get("/country-ip", headers={"CF-IPCountry": "fr", "X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(response.json, {"ip": "127.0.0.1", "country": "FR"})
        self.assertIsNone(self.client.get("/country-ip", headers={"CF-IPCountry": "<script>"}).json["country"])

    def test_trim_aware_translation(self):
        self.assertEqual(normalize_text("  Hello\n   world  "), "Hello world")
        self.assertEqual(
            translate_text("\n  Hello   world\n", {"Hello world": "Hola mundo"}),
            "\n  Hola mundo\n",
        )
        self.assertEqual(translate_text("  ", {"": "x"}), "  ")
        self.assertEqual(
            translate_text("https://example.com", {"https://example.com": "x"}),
            "https://example.com",
        )


if __name__ == "__main__":
    unittest.main()
