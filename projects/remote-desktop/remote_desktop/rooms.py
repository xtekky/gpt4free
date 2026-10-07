"""Room registry: pairs one host with its viewers and arbitrates control.

A *room* is created by a host page when it starts sharing. Viewers join with
the short room code. The registry owns the input lock so two phones cannot
fight over the pointer.
"""

from __future__ import annotations

import logging
import secrets
import time

logger = logging.getLogger(__name__)

ROLE_HOST = "host"
ROLE_VIEWER = "viewer"


class RoomError(Exception):
    """Raised when a room operation cannot be satisfied."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Peer:
    """One websocket connection inside a room."""

    def __init__(self, peer_id: str, role: str, room_id: str, send_json, send_binary=None) -> None:
        self.id = peer_id
        self.role = role
        self.room_id = room_id
        self.connected_at = time.monotonic()
        self.last_seen = self.connected_at
        self._send_json = send_json
        self._send_binary = send_binary

    async def send(self, message: dict) -> bool:
        """Send a JSON message, reporting whether the peer accepted it."""
        try:
            await self._send_json(message)
            return True
        except Exception:
            logger.debug("dropping peer %s: send failed", self.id)
            return False

    async def send_binary(self, payload: bytes) -> bool:
        if self._send_binary is None:
            return False
        try:
            await self._send_binary(payload)
            return True
        except Exception:
            return False

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "connected_at": round(self.connected_at, 3),
            "last_seen": round(self.last_seen, 3),
        }


class Room:
    """A host plus the viewers watching it."""

    def __init__(self, room_id: str) -> None:
        self.id = room_id
        self.host: Peer | None = None
        self.viewers: dict[str, Peer] = {}
        self.control_owner: str | None = None
        self.created_at = time.monotonic()
        self.last_activity = self.created_at

    @property
    def peers(self) -> list[Peer]:
        peers = list(self.viewers.values())
        if self.host is not None:
            peers.insert(0, self.host)
        return peers

    @property
    def viewer_count(self) -> int:
        return len(self.viewers)

    def get(self, peer_id: str) -> Peer | None:
        if self.host is not None and self.host.id == peer_id:
            return self.host
        return self.viewers.get(peer_id)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def snapshot(self) -> dict:
        return {
            "room": self.id,
            "host": self.host.id if self.host else None,
            "viewers": [peer.snapshot() for peer in self.viewers.values()],
            "viewer_count": self.viewer_count,
            "control_owner": self.control_owner,
            "created_at": round(self.created_at, 3),
            "last_activity": round(self.last_activity, 3),
        }


class RoomRegistry:
    """Owns every live room and the control lock inside each of them."""

    def __init__(self, max_viewers: int = 4, room_ttl: float = 120.0, auto_control: bool = True) -> None:
        self.max_viewers = max_viewers
        self.room_ttl = room_ttl
        self.auto_control = auto_control
        self.rooms: dict[str, Room] = {}

    def create_room(self, host: Peer) -> Room:
        room_id = self._new_room_id()
        room = Room(room_id)
        room.host = host
        host.room_id = room_id
        self.rooms[room_id] = room
        return room

    def _new_room_id(self) -> str:
        while True:
            room_id = secrets.token_hex(3).upper()
            if room_id not in self.rooms:
                return room_id

    def get(self, room_id: str) -> Room | None:
        return self.rooms.get(room_id)

    def join(self, room_id: str, peer: Peer) -> Room:
        room = self.rooms.get(room_id)
        if room is None:
            raise RoomError("no-room", "Unknown room code")
        if room.host is None:
            raise RoomError("no-host", "The host is not sharing right now")
        if len(room.viewers) >= self.max_viewers:
            raise RoomError("room-full", "Too many viewers in this room")
        room.viewers[peer.id] = peer
        peer.room_id = room.id
        room.touch()
        if self.auto_control and room.control_owner is None:
            room.control_owner = peer.id
        return room

    def leave(self, peer: Peer) -> Room | None:
        """Remove a peer; returns the surviving room, or ``None`` if it closed."""
        room = self.rooms.get(peer.room_id)
        if room is None:
            return None
        if room.host is not None and room.host.id == peer.id:
            room.host = None
        else:
            room.viewers.pop(peer.id, None)
        if room.control_owner == peer.id:
            room.control_owner = None
        room.touch()
        if room.host is None and not room.viewers:
            self.rooms.pop(room.id, None)
            return None
        return room

    def grant_control(self, room: Room, peer_id: str) -> bool:
        if room.control_owner not in (None, peer_id):
            return False
        room.control_owner = peer_id
        room.touch()
        return True

    def release_control(self, room: Room, peer_id: str) -> bool:
        if room.control_owner != peer_id:
            return False
        room.control_owner = None
        room.touch()
        return True

    def prune(self) -> list[str]:
        """Drop rooms that lost their host and have been idle for too long."""
        now = time.monotonic()
        removed: list[str] = []
        for room_id, room in list(self.rooms.items()):
            if room.host is None and not room.viewers:
                removed.append(room_id)
                del self.rooms[room_id]
            elif room.host is None and now - room.last_activity > self.room_ttl:
                removed.append(room_id)
                del self.rooms[room_id]
        return removed

    def stats(self) -> dict:
        return {
            "rooms": len(self.rooms),
            "hosts": sum(1 for room in self.rooms.values() if room.host is not None),
            "viewers": sum(room.viewer_count for room in self.rooms.values()),
        }
