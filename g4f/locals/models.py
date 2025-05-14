from __future__ import annotations

import os
import requests
import json

from ..requests.raise_for_status import raise_for_status

def load_models():
    response = requests.get("https://gpt4all.io/models/models3.json")
    raise_for_status(response)
    return format_models(response.json())

def get_model_name(filename: str) -> str:
    name = filename.split(".", 1)[0]
    for replace in ["-v1_5", "-v1", "-q4_0", "_v01", "-v0", "-f16", "-gguf2", "-newbpe"]:
        name = name.replace(replace, "")
    return name

def format_models(models: list) -> dict:
    return {get_model_name(model["filename"]): {
        "path": model["filename"],
        "ram": model["ramrequired"],
        "prompt": model["promptTemplate"] if "promptTemplate" in model else None,
        "system": model["systemPrompt"] if "systemPrompt" in model else None,
    } for model in models}

def read_models(file_path: str):
    with open(file_path, "rb") as f:
         return json.load(f)

def save_models(file_path: str, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_model_dir() -> str:
    local_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(local_dir))
    model_dir = os.path.join(project_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir


def get_models() -> dict[str, dict]:
    model_dir = get_model_dir()
    file_path = os.path.join(model_dir, "models.json")
    if os.path.isfile(file_path):
        return read_models(file_path)
    else:
        models = load_models()
        save_models(file_path, models)
        return models
