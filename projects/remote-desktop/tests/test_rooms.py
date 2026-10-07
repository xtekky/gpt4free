"""Tests for :mod:`remote_desktop.rooms`."""

from __future__ import annotations

import asyncio
import re

import pytest

from remote_desktop.rooms import ROLE_HOST, ROLE_VIEWER, Peer, RoomError, RoomRegistry


def make_peer(peer_id: str, role: str = ROLE_VIEWER) -> Peer:
    sent: list[dict] = []

    async def send_json(message: dict) -> None:
        sent.append(message)

    peer = Peer(peer_id, role, "", send_json)
    peer.sent = sent  # type: ignore[attr-defined]
    return peer


def test_create_room_assigns_host_and_code():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    assert len(room.id) == 6
    assert re.fullmatch(r"[0-9A-F]{6}", room.id)
    assert room.host is host
    assert host.room_id == room.id
    assert registry.get(room.id) is room


def test_room_codes_are_unique():
    registry = RoomRegistry()
    codes = {registry.create_room(make_peer(f"h{i}", ROLE_HOST)).id for i in range(50)}
    assert len(codes) == 50


def test_join_unknown_room_raises():
    registry = RoomRegistry()
    with pytest.raises(RoomError) as info:
        registry.join("NOPE00", make_peer("v1"))
    assert info.value.code == "no-room"


def test_join_room_without_host_raises():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    viewer = make_peer("v1")
    registry.join(room.id, viewer)
    registry.leave(host)
    with pytest.raises(RoomError) as info:
        registry.join(room.id, make_peer("v2"))
    assert info.value.code == "no-host"


def test_join_room_that_closed_raises_no_room():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    registry.leave(host)
    with pytest.raises(RoomError) as info:
        registry.join(room.id, make_peer("v1"))
    assert info.value.code == "no-room"


def test_join_enforces_viewer_limit():
    registry = RoomRegistry(max_viewers=2)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    registry.join(room.id, make_peer("v2"))
    with pytest.raises(RoomError) as info:
        registry.join(room.id, make_peer("v3"))
    assert info.value.code == "room-full"


def test_first_viewer_gets_control_when_auto_control():
    registry = RoomRegistry(auto_control=True)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    viewer = make_peer("v1")
    registry.join(room.id, viewer)
    assert room.control_owner == "v1"


def test_auto_control_off_leaves_control_free():
    registry = RoomRegistry(auto_control=False)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    assert room.control_owner is None


def test_grant_control_is_exclusive():
    registry = RoomRegistry(auto_control=False)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    registry.join(room.id, make_peer("v2"))
    assert registry.grant_control(room, "v1") is True
    assert registry.grant_control(room, "v2") is False
    assert room.control_owner == "v1"
    assert registry.grant_control(room, "v1") is True


def test_release_control_only_by_owner():
    registry = RoomRegistry(auto_control=False)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    registry.grant_control(room, "v1")
    assert registry.release_control(room, "v2") is False
    assert registry.release_control(room, "v1") is True
    assert room.control_owner is None


def test_leave_clears_control_owner():
    registry = RoomRegistry(auto_control=True)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    viewer = make_peer("v1")
    registry.join(room.id, viewer)
    registry.leave(viewer)
    assert room.control_owner is None
    assert room.viewer_count == 0


def test_leave_returns_none_when_room_closes():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    assert registry.leave(host) is None
    assert registry.get(room.id) is None


def test_leave_keeps_room_alive_for_remaining_viewers():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    viewer = make_peer("v1")
    registry.join(room.id, viewer)
    surviving = registry.leave(host)
    assert surviving is room
    assert room.host is None
    assert room.viewer_count == 1


def test_leave_unknown_peer_is_safe():
    registry = RoomRegistry()
    assert registry.leave(make_peer("ghost")) is None


def test_prune_removes_empty_and_stale_rooms():
    registry = RoomRegistry(room_ttl=0.0)
    empty = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.leave(empty.host)
    # An empty room is dropped as soon as its last peer leaves.
    assert registry.get(empty.id) is None

    stale = registry.create_room(make_peer("h2", ROLE_HOST))
    registry.join(stale.id, make_peer("v1"))
    registry.leave(stale.host)
    assert registry.prune() == [stale.id]
    assert registry.stats()["rooms"] == 0


def test_prune_keeps_room_with_viewers_within_ttl():
    registry = RoomRegistry(room_ttl=3600.0)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    registry.leave(room.host)
    assert registry.prune() == []
    assert registry.get(room.id) is room


def test_prune_keeps_live_rooms():
    registry = RoomRegistry(room_ttl=3600.0)
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    assert registry.prune() == []
    assert registry.get(room.id) is room


def test_peer_send_reports_failure():
    async def boom(_message: dict) -> None:
        raise RuntimeError("closed")

    peer = Peer("p", ROLE_VIEWER, "", boom)
    assert asyncio.run(peer.send({"type": "x"})) is False


def test_peer_send_reports_success():
    sent: list[dict] = []

    async def sink(message: dict) -> None:
        sent.append(message)

    peer = Peer("p", ROLE_VIEWER, "", sink)
    assert asyncio.run(peer.send({"type": "x"})) is True
    assert sent == [{"type": "x"}]


def test_peer_send_binary_without_sink():
    peer = Peer("p", ROLE_VIEWER, "", lambda _m: None)
    assert asyncio.run(peer.send_binary(b"x")) is False


def test_peer_send_binary_uses_sink():
    received: list[bytes] = []

    async def sink(payload: bytes) -> None:
        received.append(payload)

    peer = Peer("p", ROLE_VIEWER, "", lambda _m: None, sink)
    assert asyncio.run(peer.send_binary(b"frame")) is True
    assert received == [b"frame"]


def test_peer_touch_updates_last_seen():
    peer = make_peer("p")
    before = peer.last_seen
    peer.touch()
    assert peer.last_seen >= before


def test_peer_snapshot_shape():
    peer = make_peer("p")
    snapshot = peer.snapshot()
    assert snapshot["id"] == "p"
    assert snapshot["role"] == ROLE_VIEWER
    assert set(snapshot) == {"id", "role", "connected_at", "last_seen"}


def test_room_peers_lists_host_first():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    registry.join(room.id, make_peer("v1"))
    assert [peer.id for peer in room.peers] == ["h1", "v1"]


def test_room_get_finds_host_and_viewers():
    registry = RoomRegistry()
    host = make_peer("h1", ROLE_HOST)
    room = registry.create_room(host)
    viewer = make_peer("v1")
    registry.join(room.id, viewer)
    assert room.get("h1") is host
    assert room.get("v1") is viewer
    assert room.get("nobody") is None


def test_stats_counts_hosts_and_viewers():
    registry = RoomRegistry()
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    registry.join(room.id, make_peer("v1"))
    registry.join(room.id, make_peer("v2"))
    assert registry.stats() == {"rooms": 1, "hosts": 1, "viewers": 2}


def test_snapshot_shape():
    registry = RoomRegistry()
    room = registry.create_room(make_peer("h1", ROLE_HOST))
    snapshot = room.snapshot()
    assert snapshot["room"] == room.id
    assert snapshot["host"] == "h1"
    assert snapshot["viewer_count"] == 0
    assert snapshot["control_owner"] is None
