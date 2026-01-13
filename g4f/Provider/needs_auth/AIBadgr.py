from __future__ import annotations

from ..template import OpenaiTemplate

class AIBadgr(OpenaiTemplate):
    label = "AI Badgr"
    url = "https://aibadgr.com"
    login_url = "https://aibadgr.com/api-keys"
    base_url = "https://aibadgr.com/api/v1"
    working = True
    needs_auth = True
    models_needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
