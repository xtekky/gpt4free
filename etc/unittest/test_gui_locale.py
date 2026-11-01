import unittest
from unittest.mock import patch

from g4f.gui import get_gui_app


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
        self.assertIn('href="https://g4f.space/es/"', html)
        self.assertIn('hreflang="fr" href="https://g4f.space/fr/"', html)
        self.assertIn('Herramientas de IA de G4F', html)
        self.assertIn('href="/es/providers/"', html)
        self.assertIn('src="/dist/app.js"', html)
        self.assertNotIn('href="https://g4f.dev/"', html)
        self.assertEqual(response.headers["Content-Language"], "es")

        meta = self.client.get("/fr/meta").json
        self.assertEqual(meta["routes"]["home"]["url"], "https://g4f.space/fr/")
        self.assertEqual(meta["routes"]["providers"]["alternates"]["en"],
                         "https://g4f.space/providers/")
        sitemap = self.client.get("/sitemap.xml").get_data(as_text=True)
        self.assertIn("https://g4f.space/fr/providers/", sitemap)
        self.assertIn('hreflang="de"', sitemap)

    def test_invalid_locales_and_pages(self):
        self.assertEqual(self.client.get("/xx/home.html").status_code, 404)
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

    def test_headers_and_country_ip(self):
        response = self.client.get("/headers", headers={"Authorization": "secret", "X-Test": "ok"})
        self.assertEqual(response.json["Authorization"], "[redacted]")
        self.assertEqual(response.json["X-Test"], "ok")
        self.assertEqual(response.headers["Cache-Control"], "no-store")

        response = self.client.get("/country-ip", headers={"CF-IPCountry": "fr", "X-Forwarded-For": "1.2.3.4"})
        self.assertEqual(response.json, {"ip": "127.0.0.1", "country": "FR"})
        self.assertIsNone(self.client.get("/country-ip", headers={"CF-IPCountry": "<script>"}).json["country"])


if __name__ == "__main__":
    unittest.main()
