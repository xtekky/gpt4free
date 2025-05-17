from __future__ import annotations

import secrets
import string

from .template import OpenaiTemplate

class OIVSCodeSer2(OpenaiTemplate):
    label = "OI VSCode Server 2"
    url = "https://oi-vscode-server-2.onrender.com"
    api_base = "https://oi-vscode-server-2.onrender.com/v1"
    api_endpoint = "https://oi-vscode-server-2.onrender.com/v1/chat/completions"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    default_vision_model = default_model
    vision_models = [default_vision_model]
    models = vision_models
    
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
