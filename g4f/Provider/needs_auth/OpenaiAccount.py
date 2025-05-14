from __future__ import annotations

from .OpenaiChat import OpenaiChat

class OpenaiAccount(OpenaiChat):
    needs_auth = True
    parent = "OpenaiChat"
    use_nodriver = False # Show (Auth) in the model name