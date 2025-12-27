from __future__ import annotations

import secrets
import string

from .template import OpenaiTemplate

class OIVSCodeSer2(OpenaiTemplate):
    label = "OI VSCode Server 2"
    url = "https://oi-vscode-server-2.onrender.com"
    base_url = "https://oi-vscode-server-2.onrender.com/v1"

    working = False
    default_model = "*"
    default_vision_model = "gpt-4o-mini"
    vision_models = [default_vision_model]

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        # Generate a random user ID similar to the JavaScript code
        userid = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(21))
        
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "userid": userid,
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key else {}
            ),
            **({} if headers is None else headers)
        }

class OIVSCodeSer0501(OIVSCodeSer2):
    label = "OI VSCode Server 0501"
    url = "https://oi-vscode-server-0501.onrender.com"
    base_url = "https://oi-vscode-server-0501.onrender.com/v1"
    
    default_model = "gpt-4.1-mini"
    default_vision_model = default_model
    vision_models = [default_vision_model]