from __future__ import annotations

from .template import OpenaiTemplate

class OIVSCode(OpenaiTemplate):
    label = "OI VSCode Server"
    url = "https://oi-vscode-server.onrender.com"
    api_base = "https://oi-vscode-server.onrender.com/v1"
    working = True
    needs_auth = False
    
    default_model = "gpt-4o-mini"
    default_vision_model = default_model
    vision_models = [default_model, "gpt-4o-mini"]    