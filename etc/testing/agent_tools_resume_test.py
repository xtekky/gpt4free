"""Smoke test: AgentTools always yields a resumable session token and passes
inner ProviderInfo through."""
import asyncio
import sys

from g4f.providers.response import ProviderInfo, FinishReason, Usage
from g4f.Provider.AgentTools import AgentTools, _agent_sessions


class MockInnerProvider:
    """Inner provider: answers in plain text on the first call, then a
    different answer on the next call (proves resume replays the cache)."""
    supports_native_tools = True
    calls = 0

    @classmethod
    async def create_async_generator(cls, model, messages, stream=True, media=None, **kwargs):
        yield ProviderInfo(name="MockInner", url=None, label="Mock Inner", model="mock-model")
        cls.calls += 1
        yield f"answer-{cls.calls}"
        yield Usage(promptTokens=1, completionTokens=2, totalTokens=3)
        yield FinishReason("stop")


class MockNoToolsProvider:
    """Inner provider without native tool support (emulation fallback path)."""
    supports_native_tools = False
    calls = 0

    @classmethod
    async def create_async_generator(cls, model, messages, stream=True, media=None, **kwargs):
        yield ProviderInfo(name="MockNoTools", url=None, label=None, model="mock-2")
        cls.calls += 1
        yield f"plain-{cls.calls}"
        yield FinishReason("stop")


async def collect(gen):
    return [chunk async for chunk in gen]


def get_chunks(chunks, cls):
    return [c for c in chunks if isinstance(c, cls)]


def main():
    messages = [{"role": "user", "content": "hello agent"}]

    # --- 1. Normal completion: must end with a session token (status=done) ---
    chunks = asyncio.run(collect(AgentTools.create_async_generator(
        model="agent-tools", messages=messages, provider=MockInnerProvider,
    )))
    from g4f.providers.response import JsonConversation
    convs = get_chunks(chunks, JsonConversation)
    infos = get_chunks(chunks, ProviderInfo)
    assert chunks[-1].reason == "stop", f"last chunk must be FinishReason, got {chunks[-1]!r}"
    assert len(convs) == 1, f"expected exactly 1 conversation token, got {len(convs)}"
    assert convs[0].get_dict().get("status") == "done", convs[0].get_dict()
    assert convs[0].get_dict().get("agent_session"), convs[0].get_dict()
    # Inner provider info passed through
    inner_infos = [i for i in infos if getattr(i, "name", "") == "MockInner"]
    assert inner_infos, f"inner ProviderInfo not passed through: {infos}"
    assert getattr(inner_infos[0], "model", "") == "mock-model"
    assert "answer-1" in chunks, chunks
    print("1. normal completion yields session token + inner ProviderInfo: OK")

    # --- 2. Resume: same messages replay the cached result, no new model call ---
    calls_before = MockInnerProvider.calls
    chunks2 = asyncio.run(collect(AgentTools.create_async_generator(
        model="agent-tools", messages=messages, provider=MockInnerProvider,
    )))
    assert MockInnerProvider.calls == calls_before, "resume must not re-run the agent"
    assert "answer-1" in chunks2, chunks2
    convs2 = get_chunks(chunks2, JsonConversation)
    assert convs2 and convs2[0].get_dict().get("status") == "done", convs2
    inner_infos2 = [i for i in get_chunks(chunks2, ProviderInfo) if getattr(i, "name", "") == "MockInner"]
    assert inner_infos2, "inner ProviderInfo missing on resume"
    print("2. resume replays cached result + ProviderInfo: OK")

    # --- 3. Emulation fallback path also yields the token ---
    chunks3 = asyncio.run(collect(AgentTools.create_async_generator(
        model="agent-tools", messages=[{"role": "user", "content": "other task"}],
        provider=MockNoToolsProvider,
    )))
    convs3 = get_chunks(chunks3, JsonConversation)
    assert convs3 and convs3[0].get_dict().get("status") == "done", convs3
    assert "plain-1" in chunks3, chunks3
    assert any(getattr(i, "name", "") == "MockNoTools" for i in get_chunks(chunks3, ProviderInfo))
    print("3. emulation fallback yields session token + ProviderInfo: OK")

    # --- 4. Background continuation still yields status="running" token ---
    import sys as _sys
    agent_tools_module = _sys.modules["g4f.Provider.AgentTools"]
    old_bg = agent_tools_module.AGENT_BACKGROUND
    agent_tools_module.AGENT_BACKGROUND = True
    try:
        class SlowProvider(MockInnerProvider):
            @classmethod
            async def create_async_generator(cls, model, messages, stream=True, media=None, **kwargs):
                yield ProviderInfo(name="SlowInner", url=None, label=None, model="slow")
                await asyncio.sleep(60)
                yield "late"
        chunks4 = asyncio.run(collect(AgentTools.create_async_generator(
            model="agent-tools", messages=[{"role": "user", "content": "slow task"}],
            provider=SlowProvider, timeout=0.2,
        )))
        convs4 = get_chunks(chunks4, JsonConversation)
        assert convs4 and convs4[0].get_dict().get("status") == "running", convs4
        assert chunks4[-1].reason == "stop"
        print("4. timeout path yields status=running token: OK")
    finally:
        agent_tools_module.AGENT_BACKGROUND = old_bg

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    sys.exit(main())
