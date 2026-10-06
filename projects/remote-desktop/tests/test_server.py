"""End-to-end checks over the real ASGI app with fake backends."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from remote_desktop.capture import unpack_frame
from remote_desktop.config import Settings
from remote_desktop.server import create_app, lan_ip, qr_ascii
from remote_desktop.session import SessionHub

from .fakes import FakeCapture, FakeController

TOKEN = "test-token"


@pytest.fixture
def client():
    settings = Settings(token=TOKEN, fps=60).clamp()
    hub = SessionHub(settings, capture=FakeCapture(), controller=FakeController())
    app = create_app(settings, hub=hub)
    with TestClient(app) as test_client:
        test_client.hub = hub
        yield test_client


def auth():
    return {"X-RD-Token": TOKEN}


def receive_json(socket):
    """Read the next *text* message, skipping the binary frame stream."""
    while True:
        message = socket.receive()
        if message.get("type") == "websocket.close":
            raise AssertionError("socket closed before a text message arrived")
        if "text" in message:
            return json.loads(message["text"])


def test_status_requires_a_token(client):
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-RD-Token": "wrong"}).status_code == 401


def test_status_accepts_bearer_and_query_tokens(client):
    assert client.get("/api/status", headers=auth()).status_code == 200
    assert client.get(
        "/api/status", headers={"Authorization": f"Bearer {TOKEN}"}
    ).status_code == 200
    assert client.get(f"/api/status?token={TOKEN}").status_code == 200


def test_status_payload(client):
    body = client.get("/api/status", headers=auth()).json()
    assert body["ok"] is True
    assert body["screen"] == {"width": 800, "height": 600}
    assert body["allow_input"] is True
    assert body["max_clients"] == 4


def test_screen_endpoint_returns_jpeg(client):
    response = client.get("/api/screen", headers=auth())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_screen_endpoint_requires_a_token(client):
    assert client.get("/api/screen").status_code == 401


def test_control_endpoint_needs_a_known_client(client):
    response = client.post(
        "/api/control", headers=auth(), json={"action": "request", "client": "nope"}
    )
    assert response.status_code == 404


def test_websocket_rejects_a_bad_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=wrong") as socket:
            socket.receive_text()


def test_websocket_streams_frames_and_accepts_input(client):
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        hello = json.loads(socket.receive_text())
        assert hello["t"] == "hello"
        assert hello["width"] == 800
        assert hello["control"] is True

        payload = socket.receive_bytes()
        sequence, jpeg = unpack_frame(payload)
        assert sequence >= 1
        assert jpeg[:2] == b"\xff\xd8"

        socket.send_text(json.dumps({"t": "mouse", "action": "move", "x": 0.5, "y": 0.5}))
        socket.send_text(json.dumps({"t": "key", "action": "press", "key": "Enter"}))
        socket.send_text(json.dumps({"t": "text", "text": "hi"}))

        for _ in range(50):
            if len(client.hub.controller.events) >= 3:
                break
            socket.receive_bytes()

    events = client.hub.controller.events
    assert [event["t"] for event in events] == ["mouse", "key", "text"]


def test_websocket_ping_pong(client):
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        receive_json(socket)
        socket.send_text(json.dumps({"t": "ping"}))
        while True:
            if receive_json(socket)["t"] == "pong":
                break


def test_websocket_ignores_malformed_messages(client):
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        receive_json(socket)
        socket.send_text("not json")
        socket.send_text(json.dumps([1, 2, 3]))
        socket.send_text(json.dumps({"t": "unknown"}))
        socket.send_text(json.dumps({"t": "ping"}))
        while True:
            if receive_json(socket)["t"] == "pong":
                break
    assert client.hub.controller.events == []


def test_websocket_denies_input_without_control(client):
    client.hub.settings.auto_control = False
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        hello = receive_json(socket)
        assert hello["control"] is False
        socket.send_text(json.dumps({"t": "mouse", "action": "move", "x": 0.1, "y": 0.1}))
        while True:
            message = receive_json(socket)
            if message["t"] == "error":
                assert message["message"] == "no control"
                break
    assert client.hub.controller.events == []


def test_websocket_control_handshake(client):
    client.hub.settings.auto_control = False
    with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
        receive_json(socket)
        socket.send_text(json.dumps({"t": "control", "action": "request"}))
        while True:
            message = receive_json(socket)
            if message["t"] == "control":
                assert message["action"] == "granted"
                break
        assert client.hub.control_owner is not None

        socket.send_text(json.dumps({"t": "control", "action": "release"}))
        while True:
            message = receive_json(socket)
            if message["t"] == "control" and message["action"] == "released":
                break
        assert client.hub.control_owner is None


def test_websocket_rejects_extra_clients(client):
    client.hub.settings.max_clients = 1
    with client.websocket_connect(f"/ws?token={TOKEN}") as first:
        first.receive_text()
        with client.websocket_connect(f"/ws?token={TOKEN}") as second:
            message = json.loads(second.receive_text())
            assert message["t"] == "error"
            assert "too many" in message["message"]


def test_static_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Remote Desktop" in response.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/manifest.json").status_code == 200


def test_lan_ip_is_an_address():
    parts = lan_ip().split(".")
    assert len(parts) == 4
    assert all(part.isdigit() for part in parts)


def test_qr_ascii_renders_blocks():
    code = qr_ascii("http://192.168.1.10:8765/?token=abc")
    assert code
    assert "\u2588" in code
    assert len(code.splitlines()) > 5
