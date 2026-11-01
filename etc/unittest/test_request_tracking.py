"""Tests for outgoing request tracking used by the API request log."""

import os
import tempfile
import unittest
from unittest.mock import patch

import g4f.debug
from g4f.requests.tracking import (
    RequestTracker,
    add_request_sink,
    remove_request_sink,
)


class RequestTrackerTest(unittest.TestCase):
    def setUp(self):
        self.entries = []
        self.sink = self.entries.append
        add_request_sink(self.sink)

    def tearDown(self):
        remove_request_sink(self.sink)

    def test_finish_emits_metadata_and_optional_body(self):
        tracker = RequestTracker(
            "post",
            "https://example.com/v1/chat",
            {"Authorization": "secret"},
            '{"model":"auto"}',
        )
        tracker.finish(
            status=200,
            headers={"content-type": "application/json"},
            url="https://example.com/v1/chat?stream=true",
        )

        self.assertEqual(len(self.entries), 1)
        entry = self.entries[0]
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["url"], "https://example.com/v1/chat?stream=true")
        self.assertEqual(entry["status"], 200)
        self.assertEqual(entry["request_headers"], {"Authorization": "secret"})
        self.assertEqual(entry["response_headers"], {"content-type": "application/json"})
        self.assertEqual(entry["request_body"], '{"model":"auto"}')
        self.assertIsInstance(entry["duration_ms"], int)
        self.assertTrue(entry["timestamp"].endswith("Z"))
        self.assertIsNone(entry["error"])
        self.assertNotIn("response_body", entry)

    def test_finish_only_emits_once(self):
        tracker = RequestTracker("GET", "https://example.com/")
        tracker.finish(status=200)
        tracker.finish(status=500)

        self.assertEqual([entry["status"] for entry in self.entries], [200])

    def test_error_is_captured(self):
        tracker = RequestTracker("GET", "https://example.com/")
        tracker.finish(error=ConnectionError("connection refused"))

        self.assertEqual(self.entries[0]["error"], "connection refused")
        self.assertIsNone(self.entries[0]["status"])


class ApiOutgoingEntryTest(unittest.TestCase):
    def test_outgoing_entry_matches_api_log_shape(self):
        from g4f.api import _record_outgoing_request, _request_log

        _record_outgoing_request(
            {
                "method": "get",
                "url": "https://example.com/v1/models?limit=1",
                "status": 200,
                "duration_ms": 15,
                "timestamp": "2026-01-01T00:00:00Z",
                "request_headers": {"Authorization": "secret", "X-Test": "1"},
                "response_headers": {"content-type": "application/json"},
            }
        )
        entry = _request_log[-1]
        _request_log.pop()

        self.assertEqual(entry["direction"], "outgoing")
        self.assertEqual(entry["method"], "GET")
        self.assertEqual(entry["path"], "/v1/models")
        self.assertEqual(entry["host"], "example.com")
        self.assertEqual(entry["query"], "limit=1")
        self.assertEqual(entry["status"], 200)
        self.assertIsNone(entry["request_body"])
        self.assertIsNone(entry["response_body"])
        self.assertEqual(entry["request_headers"]["Authorization"], "***")
        self.assertEqual(entry["request_headers"]["X-Test"], "1")

    def test_outgoing_body_is_written_to_har_and_can_be_loaded(self):
        from g4f.api import (
            _read_har_request_body,
            _record_outgoing_request,
            _request_log,
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"G4F_HAR_LOG_DIR": directory}), patch.object(
                g4f.debug, "logging", True
            ):
                _record_outgoing_request(
                    {
                        "method": "POST",
                        "url": "https://example.com/v1/chat",
                        "status": 200,
                        "duration_ms": 10,
                        "timestamp": "2026-01-01T00:00:00Z",
                        "request_headers": {"content-type": "application/json"},
                        "response_headers": {"content-type": "application/json"},
                        "request_body": '{"model":"auto","messages":[]}',
                    }
                )
                entry = _request_log[-1]
                har_path = entry.get("_har_path")
                body = _read_har_request_body(har_path)
                _request_log.pop()

            self.assertIsNotNone(har_path)
            self.assertTrue(har_path.startswith(directory))
            self.assertEqual(body, {"model": "auto", "messages": []})
            self.assertIsNone(entry["request_body"])


if __name__ == "__main__":
    unittest.main()
