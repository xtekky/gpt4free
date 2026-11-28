"""Tests for the AnonDrop command-line helper."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import anondrop


def _fake_http_factory(responses, calls):
    def fake_http(method, url, data=None, headers=None, timeout=300.0):
        body = data
        if hasattr(body, "read"):
            buffered = bytearray()
            while True:
                part = body.read(65536)
                if not part:
                    break
                buffered.extend(part)
            body = bytes(buffered)
        calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": headers or {},
            }
        )
        for marker, response in responses:
            if marker in url:
                return response
        raise AssertionError("unexpected request: {0}".format(url))

    return fake_http


class LinkParsingTests(unittest.TestCase):
    def test_extracts_link_from_direct_upload(self) -> None:
        text = (
            "File Link: <a href='https://anondrop.io/abc123'>"
            "https://anondrop.io/abc123</a>"
        )
        self.assertEqual(anondrop.extract_link(text), "https://anondrop.io/abc123")

    def test_extracts_link_from_remote_upload_stream(self) -> None:
        text = "Starting...\nFile Link: <a href='https://anondrop.io/xyz789'>"
        self.assertEqual(anondrop.extract_link(text), "https://anondrop.io/xyz789")

    def test_missing_link_raises(self) -> None:
        with self.assertRaises(anondrop.AnonDropError):
            anondrop.extract_link("Invalid key")

    def test_reserved_site_links_are_not_share_links(self) -> None:
        with self.assertRaises(anondrop.AnonDropError):
            anondrop.extract_link("<a href='https://anondrop.io/info'>Info</a>")

    def test_file_id_accepts_ids_and_links(self) -> None:
        self.assertEqual(anondrop.extract_file_id("abc123"), "abc123")
        self.assertEqual(
            anondrop.extract_file_id("https://anondrop.io/abc123/"), "abc123"
        )
        self.assertEqual(
            anondrop.extract_file_id(
                "https://anondrop.io/abc123/notes.txt?download=true"
            ),
            "abc123",
        )

    def test_direct_url_encodes_filename(self) -> None:
        self.assertEqual(
            anondrop.build_direct_url("abc123", "my file.txt"),
            "https://anondrop.io/abc123/my%20file.txt",
        )
        self.assertEqual(
            anondrop.build_direct_url("abc123", "my file.txt", download=True),
            "https://anondrop.io/abc123/my%20file.txt?download=true",
        )

    def test_direct_link_uses_filename_from_direct_input(self) -> None:
        self.assertEqual(
            anondrop.resolve_direct_link(
                "https://anondrop.io/abc123/notes.txt?download=true"
            ),
            "https://anondrop.io/abc123/notes.txt",
        )

    def test_direct_link_reads_filename_from_share_page(self) -> None:
        calls = []
        responses = [
            ("/abc123", (
                b"<a href='/abc123/report final.pdf?download=true'>Download</a>",
                {},
            ))
        ]
        with mock.patch.object(
            anondrop, "_http", _fake_http_factory(responses, calls)
        ):
            link = anondrop.resolve_direct_link("https://anondrop.io/abc123")
        self.assertEqual(
            link, "https://anondrop.io/abc123/report%20final.pdf"
        )
        self.assertEqual(calls[0]["url"], "https://anondrop.io/abc123")

    def test_file_id_rejects_paths(self) -> None:
        with self.assertRaises(anondrop.AnonDropError):
            anondrop.extract_file_id("https://anondrop.io/upload?x=1")


class KeyStorageTests(unittest.TestCase):
    def test_save_and_load_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "key"
            saved = anondrop.save_key("abc123", path=path)
            self.assertEqual(saved, path)
            self.assertEqual(anondrop.load_key(path), "abc123")
            if os.name == "posix":
                mode = stat.S_IMODE(path.stat().st_mode)
                self.assertEqual(mode, 0o600)

    def test_load_missing_key_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(anondrop.load_key(Path(tmp) / "missing"))

    def test_empty_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(anondrop.AnonDropError):
                anondrop.save_key("   ", path=Path(tmp) / "key")

    def test_resolve_key_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            anondrop.save_key("file-key", path=path)
            with mock.patch.dict(os.environ, {"ANONDROP_KEY": "env-key"}):
                self.assertEqual(anondrop.resolve_key("explicit", path), "explicit")
                self.assertEqual(anondrop.resolve_key(None, path), "env-key")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(anondrop.resolve_key(None, path), "file-key")


class MultipartTests(unittest.TestCase):
    def test_streams_file_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_bytes(b"hello anondrop")
            body, content_type, fileobj = anondrop._multipart_file_body(path)
            try:
                self.assertIn("multipart/form-data; boundary=", content_type)
                data = bytearray()
                while True:
                    chunk = body.read(5)
                    if not chunk:
                        break
                    data.extend(chunk)
                self.assertEqual(len(data), len(body))
            finally:
                fileobj.close()
        self.assertIn(b'name="file"; filename="note.txt"', data)
        self.assertIn(b"Content-Type: text/plain", data)
        self.assertIn(b"hello anondrop", data)
        self.assertTrue(data.endswith(b"--\r\n"))


class UploadTests(unittest.TestCase):
    def test_direct_upload_sends_key_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            calls = []
            responses = [
                ("/upload", (
                    b"File Link: <a href='https://anondrop.io/abc123'>",
                    {"X-Account-Key": "issued-key"},
                ))
            ]
            with mock.patch.object(
                anondrop, "_http", _fake_http_factory(responses, calls)
            ):
                result = anondrop.direct_upload(path, key="key 1")
        self.assertEqual(result.url, "https://anondrop.io/abc123")
        self.assertEqual(
            result.direct_url, "https://anondrop.io/abc123/drop.txt"
        )
        self.assertEqual(result.account_key, "issued-key")
        self.assertEqual(result.name, "drop.txt")
        self.assertEqual(result.size, 7)
        request = calls[0]
        self.assertEqual(request["method"], "POST")
        self.assertIn("?key=key%201", request["url"])
        self.assertIn(b"payload", request["body"])
        self.assertEqual(
            request["headers"]["Content-Length"], str(len(request["body"]))
        )

    def test_direct_upload_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            calls = []
            responses = [("/upload", (b"https://anondrop.io/abc123", {}))]
            with mock.patch.object(
                anondrop, "_http", _fake_http_factory(responses, calls)
            ):
                anondrop.direct_upload(path)
        self.assertNotIn("key=", calls[0]["url"])

    def test_chunked_upload_splits_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.bin"
            path.write_bytes(b"0123456789")
            calls = []
            responses = [
                ("/initiateupload", (b"session123", {})),
                ("/endupload", (b"File Link: <a href='https://anondrop.io/done1'>", {})),
                ("/uploadchunk", (b"", {})),
            ]
            progress = []
            with mock.patch.object(
                anondrop, "_http", _fake_http_factory(responses, calls)
            ):
                result = anondrop.chunked_upload(
                    path,
                    "key",
                    chunk_size=4,
                    progress=lambda sent, total: progress.append((sent, total)),
                )
        self.assertEqual(result.url, "https://anondrop.io/done1")
        self.assertEqual(result.direct_url, "https://anondrop.io/done1/big.bin")
        self.assertEqual(result.chunks, 3)
        chunks = [call for call in calls if "/uploadchunk" in call["url"]]
        self.assertEqual([len(call["body"]) for call in chunks], [4, 4, 2])
        self.assertIn("chunk_index=0", chunks[0]["url"])
        self.assertIn("chunk_index=2", chunks[2]["url"])
        self.assertIn("size=10", calls[-1]["url"])
        self.assertEqual(progress, [(4, 10), (8, 10), (10, 10)])

    def test_chunked_upload_needs_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.bin"
            path.write_bytes(b"data")
            with self.assertRaises(anondrop.AnonDropError):
                anondrop.chunked_upload(path, "")

    def test_register_reads_key_from_page(self) -> None:
        calls = []
        responses = [
            ("/register", (
                b"<script>localStorage.setItem('userkey', 'newkey1');</script>",
                {},
            ))
        ]
        with mock.patch.object(
            anondrop, "_http", _fake_http_factory(responses, calls)
        ):
            key = anondrop.register()
        self.assertEqual(key, "newkey1")

    def test_delete_reports_server_result(self) -> None:
        calls = []
        responses = [("/delete/abc123", (b"not deleted", {}))]
        with mock.patch.object(
            anondrop, "_http", _fake_http_factory(responses, calls)
        ):
            message, deleted = anondrop.delete_file(
                "https://anondrop.io/abc123", "key"
            )
        self.assertEqual(message, "not deleted")
        self.assertFalse(deleted)
        self.assertIn("/delete/abc123?key=key", calls[0]["url"])
        self.assertEqual(calls[0]["method"], "POST")

    def test_list_files_returns_payload(self) -> None:
        payload = {"files": [{"id": "abc123", "name": "drop.txt"}], "pages": []}
        calls = []
        responses = [("/files", (json.dumps(payload).encode(), {}))]
        with mock.patch.object(
            anondrop, "_http", _fake_http_factory(responses, calls)
        ):
            result = anondrop.list_files("key")
        self.assertEqual(result, payload)
        self.assertIn("/files?key=key", calls[0]["url"])


class CommandLineTests(unittest.TestCase):
    def test_bare_file_argument_defaults_to_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            output = io.StringIO()
            result = anondrop.UploadResult(
                name="drop.txt", size=7, url="https://anondrop.io/abc123"
            )
            with mock.patch.object(anondrop, "upload", return_value=result) as upload:
                with mock.patch.object(anondrop, "resolve_key", return_value=None):
                    with contextlib.redirect_stdout(output):
                        code = anondrop.main([str(path), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            [
                {
                    "name": "drop.txt",
                    "size": 7,
                    "url": "https://anondrop.io/abc123",
                    "chunks": 1,
                    "direct_url": None,
                    "download_url": None,
                }
            ],
        )
        upload.assert_called_once()

    def test_drop_direct_prints_direct_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            output = io.StringIO()
            result = anondrop.UploadResult(
                name="drop.txt",
                size=7,
                url="https://anondrop.io/abc123",
                direct_url="https://anondrop.io/abc123/drop.txt",
            )
            with mock.patch.object(anondrop, "upload", return_value=result):
                with mock.patch.object(anondrop, "resolve_key", return_value=None):
                    with contextlib.redirect_stdout(output):
                        code = anondrop.main([str(path), "--direct"])
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().strip(),
            "https://anondrop.io/abc123/drop.txt",
        )

    def test_drop_download_appends_download_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            output = io.StringIO()
            result = anondrop.UploadResult(
                name="drop.txt",
                size=7,
                url="https://anondrop.io/abc123",
                direct_url="https://anondrop.io/abc123/drop.txt",
            )
            with mock.patch.object(anondrop, "upload", return_value=result):
                with mock.patch.object(anondrop, "resolve_key", return_value=None):
                    with contextlib.redirect_stdout(output):
                        code = anondrop.main([str(path), "--download"])
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().strip(),
            "https://anondrop.io/abc123/drop.txt?download=true",
        )

    def test_direct_command_builds_download_link(self) -> None:
        output = io.StringIO()
        with mock.patch.object(anondrop, "resolve_key", return_value=None):
            with contextlib.redirect_stdout(output):
                code = anondrop.main([
                    "direct",
                    "https://anondrop.io/abc123",
                    "--filename",
                    "my file.txt",
                    "--download",
                ])
        self.assertEqual(code, 0)
        self.assertEqual(
            output.getvalue().strip(),
            "https://anondrop.io/abc123/my%20file.txt?download=true",
        )

    def test_failed_upload_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "drop.txt"
            path.write_bytes(b"payload")
            with mock.patch.object(
                anondrop,
                "upload",
                side_effect=anondrop.AnonDropError("boom"),
            ):
                with mock.patch.object(anondrop, "resolve_key", return_value=None):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with contextlib.redirect_stdout(io.StringIO()):
                            code = anondrop.main([str(path)])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
