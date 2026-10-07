"""Remote desktop: share this screen with a phone over the local network.

The desktop runs this small server, which serves a mobile web app and relays
WebRTC signaling. The screen itself is captured by the browser through the
native ``getDisplayMedia`` API, so no screen-scraping library is involved.
"""

from .config import Settings
from .input_bridge import InputBridge
from .rooms import ROLE_HOST, ROLE_VIEWER, Peer, Room, RoomError, RoomRegistry

__all__ = [
    "InputBridge",
    "Peer",
    "ROLE_HOST",
    "ROLE_VIEWER",
    "Room",
    "RoomError",
    "RoomRegistry",
    "Settings",
]

__version__ = "0.1.0"
