from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.docs_publish import (
    MANIFEST_NAME,
    DocsPublishError,
    extract_title,
    markdown_to_html,
    publish_docs,
    render_document,
)


class MarkdownRenderingTests(unittest.TestCase):
    def test_headings_inline_markup_and_links(self) -> None:
        html = markdown_to_html(
            "# Hello\n\n"
            "Use **bold**, *italic*, ~~old~~ and `code`.\n\n"
            "Read [the guide](guide.md#start) or [README](README.md).\n"
        )
        self.assertIn("<h1>Hello</h1>", html)
        self.assertIn("<strong>bold</strong>", html)
        self.assertIn("<em>italic</em>", html)
        self.assertIn("<del>old</del>", html)
        self.assertIn("<code>code</code>", html)
        self.assertIn('href="guide.html#start"', html)
        self.assertIn('href="index.html"', html)

    def test_lists_tables_quotes_and_code_fences(self) -> None:
        html = markdown_to_html(
            "- one\n"
            "  - nested\n"
            "- two\n\n"
            "1. first\n"
            "2. second\n\n"
            "| Key | Value |\n"
            "| :--- | ---: |\n"
            "| a | `b` |\n\n"
            "> quoted\n\n"
            '```python\nprint("<hi>")\n```\n'
        )
        self.assertIn("<ul><li>", html)
        self.assertIn("<ul><li><p>nested</p></li></ul>", html)
        self.assertIn("<ol><li>", html)
        self.assertIn('<th style="text-align: left">Key</th>', html)
        self.assertIn('<td style="text-align: right"><code>b</code></td>', html)
        self.assertIn("<blockquote>", html)
        self.assertIn('<code class="language-python">', html)
        self.assertIn("&lt;hi&gt;", html)

    def test_nested_four_backtick_fence(self) -> None:
        html = markdown_to_html(
            "````markdown\n"
            "```json\n"
            '{"a": 1}\n'
            "```\n"
            "````\n"
        )
        self.assertIn('<code class="language-markdown">', html)
        self.assertIn("```json", html)
        self.assertIn("&quot;a&quot;", html)

    def test_title_and_document_template(self) -> None:
        markdown = "# My `Guide`\n\nBody.\n"
        self.assertEqual(extract_title(markdown), "My Guide")
        page = render_document(
            markdown,
            "<title>{{ title }}</title><main>{{ article }}</main>"
            '<link href="/docs/{{ filename }}">',
            "sub/page.html",
        )
        self.assertIn("<title>My Guide</title>", page)
        self.assertIn("<h1>My <code>Guide</code></h1>", page)
        self.assertIn('href="/docs/sub/page.html"', page)


class PublishDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "docs"
        self.source.mkdir()
        self.target = self.root / "g4f.dev" / "docs"
        self.target.mkdir(parents=True)
        (self.target / "template.html").write_text(
            "<title>{{ title }}</title><main>{{ article }}</main>",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_doc(self, name: str, text: str) -> Path:
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_publish_copies_markdown_and_generates_html(self) -> None:
        self.write_doc("README.md", "# Docs\n")
        self.write_doc("guide.md", "# Guide\n\nSee [nested](sub/nested.md).\n")
        self.write_doc("sub/nested.md", "# Nested\n")

        result = publish_docs(self.source, self.target)

        self.assertEqual(sorted(result.copied), ["guide.md", "sub/nested.md"])
        self.assertEqual(
            sorted(result.rendered), ["guide.html", "sub/nested.html"]
        )
        self.assertTrue((self.target / "guide.md").is_file())
        self.assertTrue((self.target / "sub" / "nested.md").is_file())
        self.assertFalse((self.target / "README.md").exists())
        self.assertFalse((self.target / "index.html").exists())
        html = (self.target / "guide.html").read_text(encoding="utf-8")
        self.assertIn("<title>Guide</title>", html)
        self.assertIn('href="sub/nested.html"', html)

        manifest = json.loads(
            (self.target / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema"], 1)
        self.assertIn("guide.md", manifest["files"])

        check = publish_docs(self.source, self.target, check=True)
        self.assertEqual(check.stale, [])

    def test_include_readme_publishes_index(self) -> None:
        self.write_doc("README.md", "# Docs Home\n")
        self.write_doc("guide.md", "# Guide\n")

        result = publish_docs(self.source, self.target, include_readme=True)

        self.assertIn("README.md", result.copied)
        self.assertIn("index.html", result.rendered)
        index = (self.target / "index.html").read_text(encoding="utf-8")
        self.assertIn("<title>Docs Home</title>", index)

    def test_check_detects_stale_docs_then_publish_updates_them(self) -> None:
        self.write_doc("guide.md", "# Guide\n\nOld.\n")
        publish_docs(self.source, self.target)
        self.write_doc("guide.md", "# Guide\n\nNew.\n")

        check = publish_docs(self.source, self.target, check=True)
        self.assertIn("guide.md", check.stale)
        self.assertIn("guide.html", check.stale)

        publish_docs(self.source, self.target)
        updated = (self.target / "guide.html").read_text(encoding="utf-8")
        self.assertIn("New.", updated)
        self.assertNotIn("Old.", updated)
        self.assertEqual(
            publish_docs(self.source, self.target, check=True).stale, []
        )

    def test_clean_removes_only_previously_published_files(self) -> None:
        self.write_doc("guide.md", "# Guide\n")
        self.write_doc("old.md", "# Old\n")
        publish_docs(self.source, self.target)
        (self.target / "unrelated.html").write_text("keep", encoding="utf-8")
        (self.source / "old.md").unlink()

        result = publish_docs(self.source, self.target, clean=True)

        self.assertIn("old.md", result.removed)
        self.assertIn("old.html", result.removed)
        self.assertFalse((self.target / "old.md").exists())
        self.assertFalse((self.target / "old.html").exists())
        self.assertTrue((self.target / "unrelated.html").exists())
        self.assertTrue((self.target / "guide.html").exists())

    def test_dry_run_reports_changes_without_writing(self) -> None:
        self.write_doc("guide.md", "# Guide\n")

        result = publish_docs(self.source, self.target, dry_run=True)

        self.assertEqual(result.copied, ["guide.md"])
        self.assertEqual(result.rendered, ["guide.html"])
        self.assertFalse((self.target / "guide.md").exists())
        self.assertFalse((self.target / "guide.html").exists())

    def test_missing_target_fails_clearly(self) -> None:
        self.write_doc("guide.md", "# Guide\n")
        missing = self.root / "missing"
        with self.assertRaises(DocsPublishError) as raised:
            publish_docs(self.source, missing)
        self.assertIn("does not exist", str(raised.exception))

    def test_create_target_and_builtin_template(self) -> None:
        self.write_doc("guide.md", "# Guide\n\nBody.\n")
        target = self.root / "new-docs"

        publish_docs(self.source, target, create_target=True)

        html = (target / "guide.html").read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<title>Guide</title>", html)
        self.assertIn("<p>Body.</p>", html)

    def test_repository_docs_all_render(self) -> None:
        repository_docs = Path(__file__).resolve().parents[3] / "docs"
        target = self.root / "repository-docs"
        target.mkdir()

        result = publish_docs(repository_docs, target, include_readme=True)

        markdown_files = list(repository_docs.rglob("*.md"))
        self.assertEqual(len(result.copied), len(markdown_files))
        self.assertEqual(len(result.rendered), len(markdown_files))
        self.assertTrue((target / "index.html").is_file())
        self.assertTrue((target / "sandbox-desktop.html").is_file())


if __name__ == "__main__":
    unittest.main()
