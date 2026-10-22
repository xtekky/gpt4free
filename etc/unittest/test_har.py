"""Tests for stripping binary payloads out of HAR files."""

import base64
import json
import os
import tempfile
import unittest

from g4f.har import (
    decode_base64_text,
    is_binary_mime_type,
    load_har,
    looks_like_base64,
    sanitize_har_text,
    strip_har_binary,
)

PNG_BASE64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 600).decode()

def _har(entries: list) -> dict:
    return {"log": {"version": "1.2", "entries": entries}}

def _entry(url: str, content: dict = None, post_data: dict = None) -> dict:
    entry = {"request": {"url": url, "headers": []}, "response": {}}
    if content is not None:
        entry["response"]["content"] = content
    if post_data is not None:
        entry["request"]["postData"] = post_data
    return entry

class TestBinaryDetection(unittest.TestCase):
    def test_binary_mime_types(self):
        for mime_type in (
            "image/png",
            "audio/mpeg",
            "video/mp4",
            "font/woff2",
            "application/octet-stream",
            "application/pdf",
            "IMAGE/PNG; charset=binary",
        ):
            self.assertTrue(is_binary_mime_type(mime_type), mime_type)

    def test_text_mime_types(self):
        for mime_type in (
            None,
            "",
            "text/html",
            "text/plain; charset=utf-8",
            "application/json",
            "application/ld+json",
            "application/xhtml+xml",
            "image/svg+xml",
        ):
            self.assertFalse(is_binary_mime_type(mime_type), mime_type)

    def test_looks_like_base64(self):
        self.assertTrue(looks_like_base64(PNG_BASE64))
        self.assertFalse(looks_like_base64("short"))
        self.assertFalse(looks_like_base64("x" * 600))
        self.assertFalse(looks_like_base64('{"accessToken": "' + "a" * 600 + '"}'))

    def test_decode_base64_text(self):
        self.assertEqual(decode_base64_text(base64.b64encode(b"hello").decode()), "hello")
        self.assertIsNone(decode_base64_text(PNG_BASE64))
        self.assertIsNone(decode_base64_text("not base64!!"))

class TestStripHarBinary(unittest.TestCase):
    def test_base64_encoded_body_is_stripped(self):
        har = _har([_entry(
            "https://example.com/logo.png",
            content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64, "size": 608},
        )])
        strip_har_binary(har)
        content = har["log"]["entries"][0]["response"]["content"]
        self.assertNotIn("encoding", content)
        self.assertIn("binary data removed", content["text"])
        self.assertIn("image/png", content["text"])
        self.assertIn("608 bytes", content["text"])

    def test_binary_mime_type_without_encoding_is_stripped(self):
        har = _har([_entry(
            "https://example.com/font.woff2",
            content={"mimeType": "font/woff2", "text": "not-really-a-font"},
        )])
        strip_har_binary(har)
        self.assertIn("binary data removed", har["log"]["entries"][0]["response"]["content"]["text"])

    def test_long_base64_body_without_mime_type_is_stripped(self):
        har = _har([_entry(
            "https://example.com/blob",
            content={"text": PNG_BASE64},
        )])
        strip_har_binary(har)
        self.assertIn("binary data removed", har["log"]["entries"][0]["response"]["content"]["text"])

    def test_base64_encoded_text_body_is_decoded(self):
        body = '{"accessToken": "eyJhbGciOiJSUzI1NiJ9.payload.signature"}'
        encoded = base64.b64encode(body.encode()).decode()
        har = _har([_entry(
            "https://chatgpt.com/backend-api/f/conversation",
            content={"mimeType": "application/json", "encoding": "base64", "text": encoded},
        )])
        strip_har_binary(har)
        content = har["log"]["entries"][0]["response"]["content"]
        self.assertEqual(content["text"], body)
        self.assertNotIn("encoding", content)

    def test_text_body_is_preserved(self):
        body = '{"accessToken": "eyJhbGciOiJSUzI1NiJ9.payload.signature"}'
        har = _har([_entry(
            "https://chatgpt.com/backend-api/f/conversation",
            content={"mimeType": "application/json", "text": body},
        )])
        strip_har_binary(har)
        self.assertEqual(har["log"]["entries"][0]["response"]["content"]["text"], body)

    def test_html_body_is_preserved(self):
        body = "<html><body>" + "hello world " * 100 + "</body></html>"
        har = _har([_entry(
            "https://example.com/",
            content={"mimeType": "text/html", "text": body},
        )])
        strip_har_binary(har)
        self.assertEqual(har["log"]["entries"][0]["response"]["content"]["text"], body)

    def test_request_post_data_is_stripped(self):
        har = _har([_entry(
            "https://example.com/upload",
            post_data={"mimeType": "image/jpeg", "encoding": "base64", "text": PNG_BASE64},
        )])
        strip_har_binary(har)
        post_data = har["log"]["entries"][0]["request"]["postData"]
        self.assertIn("binary data removed", post_data["text"])
        self.assertNotIn("encoding", post_data)

    def test_headers_and_cookies_are_untouched(self):
        entry = _entry("https://example.com/", content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64})
        entry["request"]["headers"] = [{"name": "Authorization", "value": "Bearer token"}]
        entry["request"]["cookies"] = [{"name": "session", "value": "abc"}]
        har = _har([entry])
        strip_har_binary(har)
        self.assertEqual(entry["request"]["headers"][0]["value"], "Bearer token")
        self.assertEqual(entry["request"]["cookies"][0]["value"], "abc")

    def test_malformed_har_is_ignored(self):
        for har in (None, [], {}, {"log": {}}, {"log": {"entries": "nope"}}, {"log": {"entries": [None, 1]}}):
            self.assertIs(strip_har_binary(har), har)

class TestPruneHarFields(unittest.TestCase):
    def test_bookkeeping_fields_are_removed(self):
        entry = _entry("https://example.com/", content={"mimeType": "text/html", "text": "<html></html>"})
        entry.update({
            "startedDateTime": "2024-01-01T00:00:00Z",
            "time": 12.5,
            "cache": {},
            "timings": {"send": 0, "wait": 12.5, "receive": 0},
            "serverIPAddress": "1.2.3.4",
            "connection": "443",
            "pageref": "page_1",
            "_initiator": {"type": "script", "stack": {"callFrames": [{"url": "https://example.com/app.js"}]}},
            "_priority": "High",
            "_resourceType": "document",
            "_transferSize": 1234,
        })
        entry["request"].update({
            "method": "GET",
            "httpVersion": "HTTP/1.1",
            "queryString": [{"name": "q", "value": "1"}],
            "headersSize": -1,
            "bodySize": 0,
        })
        entry["response"].update({
            "status": 200,
            "statusText": "OK",
            "httpVersion": "HTTP/1.1",
            "redirectURL": "",
            "headersSize": -1,
            "bodySize": 42,
        })
        har = _har([entry])
        har["log"]["pages"] = [{"id": "page_1"}]
        strip_har_binary(har)
        self.assertEqual(entry["startedDateTime"], "2024-01-01T00:00:00Z")
        self.assertEqual(entry["time"], 12.5)
        for key in ("cache", "timings", "serverIPAddress", "connection", "pageref",
                    "_initiator", "_priority", "_resourceType", "_transferSize"):
            self.assertNotIn(key, entry)
        self.assertEqual(entry["request"]["method"], "GET")
        self.assertEqual(entry["request"]["url"], "https://example.com/")
        for key in ("httpVersion", "queryString", "headersSize", "bodySize"):
            self.assertNotIn(key, entry["request"])
        self.assertEqual(entry["response"]["status"], 200)
        for key in ("statusText", "httpVersion", "redirectURL", "headersSize", "bodySize"):
            self.assertNotIn(key, entry["response"])
        self.assertNotIn("pages", har["log"])
        self.assertEqual(har["log"]["version"], "1.2")

    def test_headers_and_cookies_survive_pruning(self):
        entry = _entry("https://example.com/", content={"mimeType": "text/html", "text": "<html></html>"})
        entry["request"]["headers"] = [{"name": "Authorization", "value": "******"}]
        entry["request"]["cookies"] = [{"name": "session", "value": "abc"}]
        entry["response"]["headers"] = [{"name": "content-type", "value": "text/html"}]
        har = _har([entry])
        strip_har_binary(har)
        self.assertEqual(entry["request"]["headers"][0]["value"], "******")
        self.assertEqual(entry["request"]["cookies"][0]["value"], "abc")
        self.assertEqual(entry["response"]["headers"][0]["value"], "text/html")

class TestSanitizeHarText(unittest.TestCase):
    def test_round_trip_strips_binary(self):
        text = json.dumps(_har([_entry(
            "https://example.com/logo.png",
            content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64},
        )]))
        result = sanitize_har_text(text)
        self.assertNotIn(PNG_BASE64, result)
        self.assertIn("binary data removed", result)
        self.assertEqual(json.loads(result)["log"]["entries"][0]["response"]["content"]["mimeType"], "image/png")

    def test_non_har_text_is_returned_unchanged(self):
        for text in ("not json at all", "", "[1, 2, 3]", "null"):
            self.assertEqual(sanitize_har_text(text), text)

    def test_load_har_reads_and_strips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            with open(path, "w") as file:
                json.dump(_har([_entry(
                    "https://example.com/logo.png",
                    content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64},
                )]), file)
            har = load_har(path)
        self.assertIn("binary data removed", har["log"]["entries"][0]["response"]["content"]["text"])

    def test_load_har_saves_stripped_file_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            original = _har([_entry(
                "https://example.com/logo.png",
                content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64},
            )])
            with open(path, "w") as file:
                json.dump(original, file)
            load_har(path)
            with open(path) as file:
                saved = json.load(file)
            with open(os.path.join(directory, "capture.har.orig")) as file:
                backup = json.load(file)
        self.assertIn("binary data removed", saved["log"]["entries"][0]["response"]["content"]["text"])
        self.assertEqual(backup, original)

    def test_load_har_keeps_first_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            original = _har([_entry(
                "https://example.com/logo.png",
                content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64},
            )])
            with open(path, "w") as file:
                json.dump(original, file)
            load_har(path)
            load_har(path)
            with open(os.path.join(directory, "capture.har.orig")) as file:
                backup = json.load(file)
        self.assertEqual(backup, original)

    def test_load_har_does_not_write_when_nothing_is_stripped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            with open(path, "w") as file:
                json.dump(_har([_entry(
                    "https://example.com/",
                    content={"mimeType": "text/html", "text": "<html></html>"},
                )]), file)
            before = os.path.getmtime(path)
            load_har(path)
            self.assertEqual(os.path.getmtime(path), before)
            self.assertFalse(os.path.exists(os.path.join(directory, "capture.har.orig")))

    def test_load_har_saves_when_only_fields_are_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            entry = _entry(
                "https://example.com/",
                content={"mimeType": "text/html", "text": "<html></html>"},
            )
            entry["timings"] = {"send": 0, "wait": 1, "receive": 0}
            with open(path, "w") as file:
                json.dump(_har([entry]), file)
            har = load_har(path)
            with open(path) as file:
                saved = json.load(file)
            backup_exists = os.path.exists(os.path.join(directory, "capture.har.orig"))
        self.assertNotIn("timings", har["log"]["entries"][0])
        self.assertNotIn("timings", saved["log"]["entries"][0])
        self.assertTrue(backup_exists)

    def test_load_har_can_skip_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture.har")
            with open(path, "w") as file:
                json.dump(_har([_entry(
                    "https://example.com/logo.png",
                    content={"mimeType": "image/png", "encoding": "base64", "text": PNG_BASE64},
                )]), file)
            har = load_har(path, save=False)
            with open(path) as file:
                on_disk = json.load(file)
            self.assertIn("binary data removed", har["log"]["entries"][0]["response"]["content"]["text"])
            self.assertEqual(on_disk["log"]["entries"][0]["response"]["content"]["text"], PNG_BASE64)
            self.assertFalse(os.path.exists(os.path.join(directory, "capture.har.orig")))