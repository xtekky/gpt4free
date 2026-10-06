"""Client registry, frame fan-out and control arbitration."""

from __future__ import annotations

import asyncio

import pytest

from remote_desktop.config import Settings
from remote_desktop.session import Client, SessionHub

from .fakes import FakeCapture, FakeController


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.json = []

    async def send_bytes(self, payload):
        self.sent.append(payload)

    async def send_json(self, message):
        self.json.append(message)


def make_hub(**overrides):
    settings = Settings(**overrides).clamp()
    settings.token = "test"
    return SessionHub(settings, capture=FakeCapture(), controller=FakeController())


def test_offer_drops_oldest_frame_when_full():
    client = Client("c1", FakeSocket(), queue_size=2)
    client.offer(b"one")
    client.offer(b"two")
    client.offer(b"three")
    assert client.queue.qsize() == 2
    assert client.queue.get_nowait() == b"two"
    assert client.queue.get_nowait() == b"three"
    assert client.frames_dropped == 1


@pytest.mark.asyncio
async def test_add_client_enforces_max_clients():
    hub = make_hub(max_clients=1)
    first = await hub.add_client(FakeSocket())
    assert first is not None
    assert await hub.add_client(FakeSocket()) is None
    await hub.remove_client(first)
    assert await hub.add_client(FakeSocket()) is not None
    await hub.shutdown()


@pytest.mark.asyncio
async def test_auto_control_grants_first_client_only():
    hub = make_hub(auto_control=True)
    first = await hub.add_client(FakeSocket())
    second = await hub.add_client(FakeSocket())
    assert first.has_control is True
    assert second.has_control is False
    assert hub.control_owner == first.id
    assert hub.request_control(second) is False
    await hub.shutdown()


@pytest.mark.asyncio
async def test_control_can_be_released_and_taken():
    hub = make_hub(auto_control=False)
    first = await hub.add_client(FakeSocket())
    second = await hub.add_client(FakeSocket())
    assert hub.control_owner is None
    assert hub.request_control(first) is True
    hub.release_control(first)
    assert hub.control_owner is None
    assert hub.request_control(second) is True
    await hub.shutdown()


@pytest.mark.asyncio
async def test_control_is_refused_when_input_disabled():
    hub = make_hub(allow_input=False, auto_control=False)
    client = await hub.add_client(FakeSocket())
    assert hub.request_control(client) is False
    assert hub.control_owner is None
    await hub.shutdown()


@pytest.mark.asyncio
async def test_stale_control_holder_is_replaced():
    hub = make_hub(auto_control=False)
    first = await hub.add_client(FakeSocket())
    second = await hub.add_client(FakeSocket())
    hub.request_control(first)
    first.last_activity -= 120.0
    assert hub.request_control(second) is True
    assert first.has_control is False
    assert hub.control_owner == second.id
    await hub.shutdown()


@pytest.mark.asyncio
async def test_disconnect_releases_control_and_input():
    hub = make_hub(auto_control=True)
    client = await hub.add_client(FakeSocket())
    assert client.has_control is True
    await hub.remove_client(client)
    assert hub.control_owner is None
    assert hub.controller.releases == 1
    await hub.shutdown()


@pytest.mark.asyncio
async def test_capture_loop_streams_frames_to_clients():
    hub = make_hub(fps=60)
    client = await hub.add_client(FakeSocket())
    payload = await asyncio.wait_for(client.queue.get(), timeout=5.0)
    assert payload[:1] == b"\x01"
    assert hub.last_frame is not None
    assert hub.status()["frames"] >= 1
    await hub.shutdown()


@pytest.mark.asyncio
async def test_writer_loop_sends_queued_frames():
    hub = make_hub()
    socket = FakeSocket()
    client = Client("c1", socket)
    task = asyncio.create_task(hub.writer_loop(client))
    client.offer(b"payload")
    await asyncio.sleep(0.05)
    task.cancel()
    assert socket.sent == [b"payload"]
    assert client.frames_sent == 1


@pytest.mark.asyncio
async def test_status_reports_backends():
    hub = make_hub()
    status = hub.status()
    assert status["ok"] is True
    assert status["screen"] == {"width": 800, "height": 600}
    assert status["allow_input"] is True
    assert status["input"]["enabled"] is True
    await hub.shutdown()


@pytest.mark.asyncio
async def test_hello_describes_the_session():
    hub = make_hub()
    client = await hub.add_client(FakeSocket())
    hello = hub.hello(client)
    assert hello["t"] == "hello"
    assert hello["width"] == 800
    assert hello["control"] is True
    await hub.shutdown()


@pytest.mark.asyncio
async def test_idle_stop_pauses_capture():
    hub = make_hub(idle_timeout=5.0)
    client = await hub.add_client(FakeSocket())
    await hub.remove_client(client)
    assert hub._idle_task is not None
    hub._idle_task.cancel()
    await hub.shutdown()
    assert hub.streaming is False
