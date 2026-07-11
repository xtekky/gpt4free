from __future__ import annotations

import os
import json
from ..image.copy_images import secure_filename
from ..cookies import get_cookies_dir

class FileStorage():
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.join(get_cookies_dir(), ".models")
        self.storage_dir = storage_dir

    def get_file(self, key: str) -> str:
        return os.path.join(self.storage_dir, *[secure_filename(part) for part in key.split("/")]) + ".json"

    def set(self, key: str, value: str):
        file = self.get_file(key)
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(value, f)

    def get(self, key: str) -> str | None:
        try:
            with open(self.get_file(key), "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def delete(self, key: str):
        try:
            os.remove(self.get_file(key))
        except FileNotFoundError:
            pass

    def clear(self):
        for root, dirs, files in os.walk(self.storage_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))