"""Tests for :mod:`remote_desktop.app` (HTTP surface and signaling relay)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from remote_desktop.app import create_app, local_addresses, qr_svg, share_urls
from remote_desktop.config import Settings


def make_settings(**overrides) -> Settings:
    values = {"token": "", "allow_input": False, "auto_control": True, "max_viewers": 2}
    values.update(overrides)
    return Settings(**values).clamp()


@pytest.fixture
def client():
    with TestClient(create_app(make_settings())) as test_client:
        yield test_client


def hello(ws, role: str, room: str = "", token: str = "") -> dict:
    ws.send_json({"type": "hello", "role": role, "room": room, "token": token})
    return ws.receive_json()


def test_index_redirects_to_host(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/host"


def test_host_and_viewer_pages_are_served(client):
    host = client.get("/host")
    view = client.get("/view")
    assert host.status_code == 200
    assert view.status_code == 200
    assert "getDisplayMedia" in host.text
    assert "viewer.js" in view.text


def test_static_assets_are_served(client):
    for path in ("/static/css/app.css", "/static/js/host.js", "/static/js/viewer.js", "/static/icon.svg"):
        assert client.get(path).status_code == 200, path


def test_manifest_and_service_worker(client):
    manifest = client.get("/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["start_url"] == "/view"
    assert client.get("/sw.js").status_code == 200


def test_status_reports_configuration(client):
    payload = client.get("/api/status").json()
    assert payload["allow_input"] is False
    assert payload["input"] is False
    assert payload["max_viewers"] == 2
    assert payload["rooms"] == 0
    assert isinstance(payload["urls"], list) and payload["urls"]


def test_status_exposes_ice_servers():
    settings = make_settings(turn_url="turn:turn.example.com:3478", turn_secret="s3cret")
    with TestClient(create_app(settings)) as test_client:
        servers = test_client.get("/api/status").json()["ice_servers"]
    assert servers[0]["urls"] == ["stun:stun.l.google.com:19302"]
    assert servers[1]["urls"] == ["turn:turn.example.com:3478"]
    assert servers[1]["credential"]


def test_status_ice_servers_are_fresh_per_request():
    settings = make_settings(turn_url="turn:turn.example.com:3478", turn_secret="s3cret")
    with TestClient(create_app(settings)) as test_client:
        first = test_client.get("/api/status").json()["ice_servers"][1]["username"]
        second = test_client.get("/api/status").json()["ice_servers"][1]["username"]
    assert first != second


def test_status_without_turn_has_no_relay():
    with TestClient(create_app(make_settings(turn_url=""))) as test_client:
        servers = test_client.get("/api/status").json()["ice_servers"]
    assert all("turn" not in url for entry in servers for url in entry["urls"])

def test_qr_endpoint_returns_svg(client):
    response = client.get("/api/qr.svg", params={"url": "http://192.168.1.5:8765/view?room=ABC123"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in response.content


def test_qr_endpoint_rejects_long_urls(client):
    assert client.get("/api/qr.svg", params={"url": "x" * 600}).status_code == 400


def test_qr_svg_is_well_formed():
    assert qr_svg("http://example.test/view").startswith(b"<?xml")


def test_host_hello_creates_room(client):
    with client.websocket_connect("/ws") as ws:
        message = hello(ws, "host")
        assert message["type"] == "room"
        assert message["role"] == "host"
        assert len(message["room"]) == 6
        assert message["viewers"] == 0


def test_viewer_joins_and_host_is_notified(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            joined = hello(viewer_ws, "viewer", room)
            assert joined["type"] == "joined"
            assert joined["room"] == room
            assert joined["control"] == joined["peer"]
            notice = host_ws.receive_json()
            assert notice["type"] == "viewer-joined"
            assert notice["viewers"] == 1
            assert notice["control"] == joined["peer"]

def test_viewer_joined_reports_control_owner_to_host(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as first_ws:
            first = hello(first_ws, "viewer", room)
            assert host_ws.receive_json()["control"] == first["peer"]
            with client.websocket_connect("/ws") as second_ws:
                second = hello(second_ws, "viewer", room)
                notice = host_ws.receive_json()
                assert notice["type"] == "viewer-joined"
                assert notice["peer"] == second["peer"]
                assert notice["control"] == first["peer"]
                assert notice["viewers"] == 2


def test_viewer_with_unknown_room_is_rejected(client):
    with client.websocket_connect("/ws") as ws:
        message = hello(ws, "viewer", "ZZZZZZ")
        assert message["type"] == "error"
        assert message["code"] == "no-room"


def test_bad_role_is_rejected(client):
    with client.websocket_connect("/ws") as ws:
        message = hello(ws, "spy")
        assert message["code"] == "bad-role"


def test_second_hello_is_rejected(client):
    with client.websocket_connect("/ws") as ws:
        hello(ws, "host")
        ws.send_json({"type": "hello", "role": "host"})
        assert ws.receive_json()["code"] == "already-joined"


def test_malformed_json_is_reported(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        assert ws.receive_json()["code"] == "bad-json"


def test_unknown_message_type_is_reported(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "teleport"})
        assert ws.receive_json()["code"] == "unknown-type"


def test_ping_is_answered(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping", "t": 42})
        assert ws.receive_json() == {"type": "pong", "t": 42}


def test_token_is_enforced():
    with TestClient(create_app(make_settings(token="s3cret"))) as client:
        with client.websocket_connect("/ws") as ws:
            assert hello(ws, "host", token="wrong")["code"] == "bad-token"
        with client.websocket_connect("/ws") as ws:
            assert hello(ws, "host", token="s3cret")["type"] == "room"


def test_viewer_limit_is_enforced():
    with TestClient(create_app(make_settings(max_viewers=1))) as client:
        with client.websocket_connect("/ws") as host_ws:
            room = hello(host_ws, "host")["room"]
            with client.websocket_connect("/ws") as first:
                assert hello(first, "viewer", room)["type"] == "joined"
                with client.websocket_connect("/ws") as second:
                    assert hello(second, "viewer", room)["code"] == "room-full"


def test_signals_are_relayed_to_the_host(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            joined = hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            viewer_ws.send_json({"type": "signal", "to": joined["host"], "data": {"kind": "request"}})
            relayed = host_ws.receive_json()
            assert relayed["type"] == "signal"
            assert relayed["from"] == joined["peer"]
            assert relayed["data"] == {"kind": "request"}


def test_signals_are_relayed_to_a_viewer(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            joined = hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            host_ws.send_json({"type": "signal", "to": joined["peer"], "data": {"kind": "offer", "sdp": "x"}})
            relayed = viewer_ws.receive_json()
            assert relayed["data"]["kind"] == "offer"


def test_quality_signal_reaches_the_host(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            joined = hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            viewer_ws.send_json({"type": "signal", "to": joined["host"], "data": {"kind": "quality", "level": "low"}})
            relayed = host_ws.receive_json()
            assert relayed["from"] == joined["peer"]
            assert relayed["data"] == {"kind": "quality", "level": "low"}


def test_signal_to_unknown_peer_is_dropped(client):
    with client.websocket_connect("/ws") as host_ws:
        hello(host_ws, "host")
        host_ws.send_json({"type": "signal", "to": "nobody", "data": {"kind": "offer"}})
        host_ws.send_json({"type": "ping", "t": 1})
        assert host_ws.receive_json() == {"type": "pong", "t": 1}


def test_control_request_and_release(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as first:
            joined = hello(first, "viewer", room)
            host_ws.receive_json()
            first.send_json({"type": "control", "action": "release"})
            assert host_ws.receive_json()["owner"] is None
            assert first.receive_json()["owner"] is None
            first.send_json({"type": "control", "action": "request"})
            assert host_ws.receive_json()["owner"] == joined["peer"]
            assert first.receive_json()["owner"] == joined["peer"]


def test_control_request_is_denied_when_taken(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as first:
            hello(first, "viewer", room)
            host_ws.receive_json()
            with client.websocket_connect("/ws") as second:
                hello(second, "viewer", room)
                host_ws.receive_json()
                second.send_json({"type": "control", "action": "request"})
                denied = second.receive_json()
                assert denied["denied"] is True


def test_host_can_grant_control(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            joined = hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            viewer_ws.send_json({"type": "control", "action": "release"})
            host_ws.receive_json()
            viewer_ws.receive_json()
            host_ws.send_json({"type": "control", "action": "grant", "target": joined["peer"]})
            assert host_ws.receive_json()["owner"] == joined["peer"]
            assert viewer_ws.receive_json()["owner"] == joined["peer"]


def test_input_is_ignored_without_control(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            viewer_ws.send_json({"type": "control", "action": "release"})
            host_ws.receive_json()
            assert viewer_ws.receive_json()["owner"] is None
            viewer_ws.send_json({"type": "input", "event": {"type": "move", "x": 0.5, "y": 0.5}})
            viewer_ws.send_json({"type": "ping", "t": 7})
            assert viewer_ws.receive_json() == {"type": "pong", "t": 7}
            assert client.app.state.bridge.events == 0


def test_input_is_applied_for_the_controller(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            viewer_ws.send_json({"type": "input", "event": {"type": "move", "x": 0.5, "y": 0.5}})
            viewer_ws.send_json({"type": "ping", "t": 8})
            assert viewer_ws.receive_json() == {"type": "pong", "t": 8}
            assert client.app.state.bridge.events == 0  # input disabled in the fixture


def test_input_reaches_the_bridge_when_enabled():
    with TestClient(create_app(make_settings(allow_input=True))) as client:
        with client.websocket_connect("/ws") as host_ws:
            room = hello(host_ws, "host")["room"]
            with client.websocket_connect("/ws") as viewer_ws:
                hello(viewer_ws, "viewer", room)
                host_ws.receive_json()
                viewer_ws.send_json({"type": "input", "event": {"type": "text", "text": "hi"}})
                viewer_ws.send_json({"type": "ping", "t": 9})
                assert viewer_ws.receive_json() == {"type": "pong", "t": 9}
                assert client.app.state.bridge.events == 1


def test_input_from_a_non_controller_is_dropped():
    with TestClient(create_app(make_settings(allow_input=True, auto_control=False))) as client:
        with client.websocket_connect("/ws") as host_ws:
            room = hello(host_ws, "host")["room"]
            with client.websocket_connect("/ws") as viewer_ws:
                hello(viewer_ws, "viewer", room)
                host_ws.receive_json()
                viewer_ws.send_json({"type": "input", "event": {"type": "text", "text": "hi"}})
                viewer_ws.send_json({"type": "ping", "t": 10})
                assert viewer_ws.receive_json() == {"type": "pong", "t": 10}
                assert client.app.state.bridge.events == 0


def test_viewer_leave_notifies_host(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
        notice = host_ws.receive_json()
        assert notice["type"] == "peer-left"
        assert notice["role"] == "viewer"
        assert notice["viewers"] == 0


def test_host_leave_notifies_viewers(client):
    with client.websocket_connect("/ws") as host_ws:
        room = hello(host_ws, "host")["room"]
        with client.websocket_connect("/ws") as viewer_ws:
            hello(viewer_ws, "viewer", room)
            host_ws.receive_json()
            host_ws.close()
            notice = viewer_ws.receive_json()
            assert notice["type"] == "peer-left"
            assert notice["role"] == "host"


def test_room_is_removed_after_everyone_leaves(client):
    with client.websocket_connect("/ws") as host_ws:
        hello(host_ws, "host")
    assert client.get("/api/status").json()["rooms"] == 0


def test_local_addresses_excludes_loopback():
    assert all(not address.startswith("127.") for address in local_addresses())


def test_share_urls_prefers_public_url():
    settings = make_settings(public_url="https://desk.example.test/")
    assert share_urls(settings) == ["https://desk.example.test"]


def test_share_urls_uses_port():
    settings = make_settings(port=9001)
    assert all(url.endswith(":9001") for url in share_urls(settings))
