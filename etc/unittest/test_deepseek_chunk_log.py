from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from g4f.providers.response import JsonResponse, ProviderInfo, Reasoning
from projects.test.deepseek_chunk_log import (
    ChunkJsonlWriter,
    read_chunk_records,
    store_or_collect_chunk,
)


class DeepSeekChunkLogTest(unittest.TestCase):
    def test_stores_metadata_chunks_and_collects_visible_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "chunks.jsonl"
            visible_chunks = []

            with ChunkJsonlWriter(log_path, run_id="run-1") as writer:
                store_or_collect_chunk(
                    ProviderInfo(name="DeepSeek"), writer, visible_chunks
                )
                store_or_collect_chunk(
                    Reasoning(token="thinking"), writer, visible_chunks
                )
                store_or_collect_chunk(
                    JsonResponse(result="metadata"), writer, visible_chunks
                )
                store_or_collect_chunk("visible answer", writer, visible_chunks)

            self.assertEqual(visible_chunks, ["visible answer"])
            self.assertEqual(
                [record["chunk_type"] for record in read_chunk_records(log_path)],
                ["ProviderInfo", "Reasoning", "JsonResponse"],
            )

    def test_chunk_exception_is_raised_instead_of_stored(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "chunks.jsonl"

            with ChunkJsonlWriter(log_path, run_id="run-1") as writer:
                with self.assertRaisesRegex(RuntimeError, "stream failed"):
                    store_or_collect_chunk(
                        RuntimeError("stream failed"), writer, []
                    )

            self.assertEqual(read_chunk_records(log_path), [])

    def test_writes_each_supported_chunk_as_flushed_unicode_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "nested" / "chunks.jsonl"

            with ChunkJsonlWriter(log_path, run_id="run-1") as writer:
                writer.write(ProviderInfo(name="DeepSeek", model="deepseek-v3"))
                writer.write(Reasoning(token="تفكير"))
                writer.write(JsonResponse(result={"answer": "إجابة"}))

                records_before_close = read_chunk_records(log_path)

            self.assertEqual(writer.count, 3)
            self.assertEqual(
                [record["chunk_type"] for record in records_before_close],
                ["ProviderInfo", "Reasoning", "JsonResponse"],
            )
            self.assertEqual(
                [record["sequence"] for record in records_before_close],
                [1, 2, 3],
            )
            self.assertTrue(
                all(record["run_id"] == "run-1" for record in records_before_close)
            )
            self.assertEqual(records_before_close[1]["data"], {"token": "تفكير"})
            self.assertEqual(
                records_before_close[2]["data"],
                {"result": {"answer": "إجابة"}},
            )
            self.assertTrue(
                all(record["recorded_at"].endswith("Z") for record in records_before_close)
            )
            self.assertIn("تفكير", log_path.read_text(encoding="utf-8"))

    def test_appends_runs_and_can_read_one_run_later(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "chunks.jsonl"

            with ChunkJsonlWriter(log_path, run_id="first-run") as writer:
                writer.write(Reasoning(token="first"))
            with ChunkJsonlWriter(log_path, run_id="second-run") as writer:
                writer.write(Reasoning(token="second"))

            self.assertEqual(len(read_chunk_records(log_path)), 2)
            self.assertEqual(
                read_chunk_records(log_path, run_id="second-run")[0]["data"],
                {"token": "second"},
            )


if __name__ == "__main__":
    unittest.main()
