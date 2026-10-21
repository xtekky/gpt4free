from __future__ import annotations

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from g4f import tracing


class _FakeObservation:
    def __init__(self, trace_id="trace-1"):
        self.trace_id = trace_id
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self

    def end(self, **kwargs):
        self.ended = True


class _FakeClient:
    def __init__(self):
        self.observations = []
        self.flushed = False
        self.shut_down = False

    def start_observation(self, **kwargs):
        observation = _FakeObservation()
        observation.kwargs = kwargs
        self.observations.append(observation)
        return observation

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.shut_down = True


@contextlib.contextmanager
def _fake_langfuse(client):
    """Install a stub `langfuse` module so the tracing code can be exercised."""
    module = types.ModuleType("langfuse")
    module.get_client = lambda *args, **kwargs: client
    module.propagate_attributes = lambda **kwargs: contextlib.nullcontext()
    with patch.dict(sys.modules, {"langfuse": module}):
        yield module


class TestLangfuseTracingDisabled(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""},
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_disabled_without_credentials(self):
        self.assertFalse(tracing.is_enabled())

    def test_disabled_by_env_flag(self):
        with patch.dict(
            "os.environ",
            {
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
                "G4F_LANGFUSE_TRACING": "0",
            },
        ):
            self.assertFalse(tracing.is_enabled())

    def test_trace_is_inert(self):
        trace = tracing.start_request_trace("chat.completions", input=[{"role": "user"}])
        self.assertFalse(trace.enabled)
        self.assertIsNone(trace.trace_id)
        trace.record_output(SimpleNamespace(choices=[]))
        trace.record_error(RuntimeError("boom"))
        trace.finish()
        trace.finish()

    def test_flush_and_shutdown_are_noops(self):
        tracing.flush()
        tracing.shutdown()


class TestModelParameters(unittest.TestCase):
    def test_collects_sampling_fields(self):
        config = SimpleNamespace(
            temperature=0.5,
            top_p=0.9,
            top_k=None,
            max_tokens=128,
            presence_penalty=None,
            frequency_penalty=None,
            stop="END",
            stop_sequences=None,
            reasoning_effort="low",
        )
        self.assertEqual(
            tracing.model_parameters(config),
            {
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 128,
                "stop": ["END"],
                "reasoning_effort": "low",
            },
        )

    def test_returns_none_when_empty(self):
        self.assertIsNone(tracing.model_parameters(SimpleNamespace()))

    def test_skips_dict_values(self):
        config = SimpleNamespace(temperature={"nested": 1}, top_p=0.1)
        self.assertEqual(tracing.model_parameters(config), {"top_p": 0.1})


class TestExtraction(unittest.TestCase):
    def test_extract_text_from_message(self):
        result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
        )
        self.assertEqual(tracing._extract_text(result), "hello")

    def test_extract_text_from_delta(self):
        result = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))]
        )
        self.assertEqual(tracing._extract_text(result), "hi")

    def test_extract_text_without_choices(self):
        self.assertIsNone(tracing._extract_text(SimpleNamespace(choices=[])))

    def test_extract_usage_mapping(self):
        result = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=4, total_tokens=7
            )
        )
        self.assertEqual(
            tracing._extract_usage(result), {"input": 3, "output": 4, "total": 7}
        )

    def test_extract_usage_missing(self):
        self.assertIsNone(tracing._extract_usage(SimpleNamespace()))


class TestRequestTraceAccumulation(unittest.TestCase):
    def test_accumulates_stream_chunks(self):
        observation = _FakeObservation()
        trace = tracing.RequestTrace(observation, trace_id="t")
        self.assertTrue(trace.enabled)
        trace.record_output(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="a"))],
                model="gpt-4o",
                provider="Openai",
            )
        )
        trace.record_output(
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="b"))],
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=2, total_tokens=3
                ),
            )
        )
        trace.finish()
        self.assertTrue(observation.ended)
        self.assertEqual(
            observation.updates,
            [
                {
                    "output": "ab",
                    "usage_details": {"input": 1, "output": 2, "total": 3},
                    "model": "gpt-4o",
                    "metadata": {"provider": "Openai"},
                }
            ],
        )

    def test_finish_is_idempotent(self):
        observation = _FakeObservation()
        trace = tracing.RequestTrace(observation)
        trace.finish()
        trace.finish()
        self.assertEqual(len(observation.updates), 0)
        self.assertTrue(observation.ended)

    def test_record_error_marks_observation(self):
        observation = _FakeObservation()
        trace = tracing.RequestTrace(observation)
        trace.record_error(RuntimeError("boom"))
        self.assertEqual(
            observation.updates, [{"level": "ERROR", "status_message": "boom"}]
        )


class TestStartRequestTrace(unittest.TestCase):
    def test_creates_generation_observation(self):
        client = _FakeClient()
        with _fake_langfuse(client), patch.dict(
            "os.environ",
            {
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
                "G4F_LANGFUSE_TAGS": "g4f, api",
            },
        ):
            trace = tracing.start_request_trace(
                "chat.completions",
                input=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                provider="Openai",
                model_parameters={"temperature": 0.2},
                metadata={"stream": False},
                user_id="user-1",
                session_id="conv-1",
                tags=["extra"],
            )
        self.assertTrue(trace.enabled)
        self.assertEqual(trace.trace_id, "trace-1")
        self.assertEqual(len(client.observations), 1)
        kwargs = client.observations[0].kwargs
        self.assertEqual(kwargs["as_type"], "generation")
        self.assertEqual(kwargs["name"], "chat.completions")
        self.assertEqual(kwargs["model"], "gpt-4o")
        self.assertEqual(kwargs["model_parameters"], {"temperature": 0.2})

    def test_flush_and_shutdown_delegate_to_client(self):
        client = _FakeClient()
        with _fake_langfuse(client), patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        ):
            tracing.flush()
            tracing.shutdown()
        self.assertTrue(client.flushed)
        self.assertTrue(client.shut_down)

    def test_none_metadata_values_are_dropped(self):
        client = _FakeClient()
        captured = {}

        @contextlib.contextmanager
        def capture(**kwargs):
            captured.update(kwargs)
            yield

        module = types.ModuleType("langfuse")
        module.get_client = lambda *a, **k: client
        module.propagate_attributes = capture
        with patch.dict(sys.modules, {"langfuse": module}), patch.dict(
            "os.environ",
            {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
        ):
            tracing.start_request_trace(
                "chat.completions", metadata={"mode": None, "stream": True}
            )
        self.assertEqual(captured["metadata"], {"stream": True})


if __name__ == "__main__":
    unittest.main()