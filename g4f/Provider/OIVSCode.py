from __future__ import annotations

from .template import OpenaiTemplate

class OIVSCode(OpenaiTemplate):
    label = "OI VSCode Server"
    url = "https://oi-vscode-server.onrender.com"
    api_base = "https://oi-vscode-server.onrender.com/v1"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini-2024-07-18"
    default_vision_model = default_model
    vision_models = [default_model, "gpt-4o-mini"]
    models = vision_models
    
    model_aliases = {"gpt-4o-mini": "gpt-4o-mini-2024-07-18"}
