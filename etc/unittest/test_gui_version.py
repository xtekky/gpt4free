import os
import unittest
from unittest.mock import patch

from g4f.gui import get_gui_app

from g4f.gui.server import versioned


HTML = """<!DOCTYPE html><html><head></head><body>
<script src="/dist/js/framework.js?v={{ v }}"></script>
<a href="/chat/pwa.js">pwa</a>
<a href="/privacy.html">privacy</a>
<a href="https://example.com/">external</a>
<a href="//cdn.example.com/app.js">protocol relative</a>
<button hx-get="/backend-api/v2/version">version</button>
<a href="/v1/models">api</a>
</body></html>"""


class TestVersionedHelpers(unittest.TestCase):
    def test_normalize_version(self):
        self.assertEqual(versioned.normalize_version("8.8.12"), "8.8.12")
        self.assertEqual(versioned.normalize_version("v8.8.12"), "8.8.12")
        self.assertEqual(versioned.normalize_version("1"), "1")
        for value in ("", "v", "8.8.12/../x", "8.8.12 beta", "latest", "../v1"):
            with self.assertRaises(ValueError, msg=value):
                versioned.normalize_version(value)

    def test_request_candidates(self):
        self.assertEqual(versioned.request_candidates(""), ["home.html"])
        self.assertEqual(
            versioned.request_candidates("chat/"), ["chat/index.html"]
        )
        self.assertEqual(
            versioned.request_candidates("chat"),
            ["chat/index.html", "chat.html"],
        )
        self.assertEqual(
            versioned.request_candidates("dist/js/app.js"),
            ["dist/js/app.js"],
        )
        with self.assertRaises(versioned.VersionedError):
            versioned.request_candidates("../secret.txt")
        with self.assertRaises(versioned.VersionedError):
            versioned.request_candidates("dist/%2e%2e/secret.txt")

    def test_rewrite_versioned_html(self):
        html = versioned.rewrite_versioned_html(HTML, "8.8.12")
        self.assertIn('/v8.8.12/dist/js/framework.js?v=v8.8.12', html)
        self.assertIn('href="/v8.8.12/chat/pwa.js"', html)
        self.assertIn('href="/v8.8.12/privacy.html"', html)
        self.assertNotIn('/vv8.8.12/', html)
        self.assertIn('href="https://example.com/"', html)
        self.assertIn('href="//cdn.example.com/app.js"', html)
        self.assertIn('hx-get="/backend-api/v2/version"', html)
        self.assertIn('href="/v1/models"', html)
        self.assertIn('<meta name="g4f-version" content="v8.8.12">', html)

    def test_older_chat_relative_assets_and_imports(self):
        html = versioned.rewrite_versioned_html(
            '<script src="../dist/js/framework.js?v={{ v }}"></script>'
            '<script type="module">import "../dist/js/providers.js"; '
            'import "/dist/js/model.js";</script>',
            "7.0.0",
        )
        self.assertIn('src="/v7.0.0/dist/js/framework.js?v=v7.0.0"', html)
        self.assertIn('import "/v7.0.0/dist/js/providers.js"', html)
        self.assertIn('import "/v7.0.0/dist/js/model.js"', html)
        self.assertNotIn('/vv7.0.0/', html)

    def test_resolve_site_ref_uses_earliest_commit_after_release(self):
        version = "99.99.98"
        versioned._REF_CACHE.pop(version, None)
        commits = [
            {"sha": "newest"},
            {"sha": "middle"},
            {"sha": "earliest-after-release"},
        ]
        with patch.object(versioned, "release_date", return_value="2026-01-01T00:00:00Z"), \
             patch.object(versioned, "_resolve_local_ref", return_value=None), \
             patch.object(versioned, "_github_get", return_value=(commits, {})) as request:
            self.assertEqual(
                versioned.resolve_site_ref(version),
                "earliest-after-release",
            )
        self.assertIn("since=", request.call_args.args[0])
        versioned._REF_CACHE.pop(version, None)

    def test_resolve_site_ref_reads_last_page_for_old_releases(self):
        version = "99.99.97"
        versioned._REF_CACHE.pop(version, None)
        newest_page = [{"sha": f"new-{index}"} for index in range(100)]
        oldest_page = [
            {"sha": "newest-in-oldest-page"},
            {"sha": "earliest-after-release"},
        ]
        links = {"last": {"url": "https://api.github.com/commits?page=9"}}
        with patch.object(versioned, "release_date", return_value="2025-01-01T00:00:00Z"), \
             patch.object(versioned, "_resolve_local_ref", return_value=None), \
             patch.object(
                 versioned,
                 "_github_get",
                 side_effect=[(newest_page, links), (oldest_page, {})],
             ):
            self.assertEqual(
                versioned.resolve_site_ref(version),
                "earliest-after-release",
            )
        versioned._REF_CACHE.pop(version, None)


class TestVersionUtils(unittest.TestCase):
    def test_no_version_check_skips_network_lookup(self):
        from g4f import version as version_module

        with patch.dict(os.environ, {"G4F_NO_VERSION_CHECK": "1"}), \
             patch.object(version_module, "get_pypi_version") as pypi, \
             patch.object(version_module, "get_github_version") as github:
            self.assertEqual(
                version_module.VersionUtils().latest_version,
                version_module.utils.current_version,
            )
        pypi.assert_not_called()
        github.assert_not_called()

    def test_version_check_enabled_by_default(self):
        from g4f import version as version_module

        with patch.dict(os.environ, {"G4F_NO_VERSION_CHECK": "0"}), \
             patch.object(version_module, "get_pypi_version", return_value="9.9.9"), \
             patch.object(version_module, "get_github_version", return_value="v9.9.9"):
            self.assertIn(
                version_module.VersionUtils().latest_version, ("9.9.9", "v9.9.9")
            )


class TestVersionedRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = get_gui_app()
        cls.app.testing = True

    def setUp(self):
        self.client = self.app.test_client()

    def test_render_versioned_chat(self):
        with patch(
            "g4f.gui.server.website.load_versioned_file",
            return_value=(HTML.encode(), "text/html", "abc123"),
        ) as load:
            response = self.client.get("/v8.8.12/chat/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-G4F-Version"], "8.8.12")
        self.assertEqual(response.headers["X-G4F-Gui-Commit"], "abc123")
        self.assertEqual(load.call_args.args, ("8.8.12", "chat/index.html"))
        html = response.get_data(as_text=True)
        self.assertIn('/v8.8.12/dist/js/framework.js?v=v8.8.12', html)
        self.assertIn('hx-get="/backend-api/v2/version"', html)

    def test_versioned_chat_without_trailing_slash(self):
        with patch(
            "g4f.gui.server.website.load_versioned_file",
            return_value=(b'<script src="../dist/js/framework.js"></script>', "text/html", "abc123"),
        ) as load:
            response = self.client.get("/v7.0.0/chat")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(load.call_args.args, ("7.0.0", "chat/index.html"))
        self.assertIn('src="/v7.0.0/dist/js/framework.js"', response.get_data(as_text=True))

    def test_versioned_asset(self):
        with patch(
            "g4f.gui.server.website.load_versioned_file",
            return_value=(b"console.log(1);", "text/javascript", "abc123"),
        ):
            response = self.client.get("/v8.8.12/dist/js/app.js")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/javascript")
        self.assertIn("immutable", response.headers["Cache-Control"])

    def test_invalid_and_unknown_versions(self):
        self.assertEqual(self.client.get("/v.latest/chat/").status_code, 404)
        with patch(
            "g4f.gui.server.website.load_versioned_file",
            side_effect=versioned.VersionedError("missing"),
        ):
            self.assertEqual(self.client.get("/v8.8.12/chat/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
