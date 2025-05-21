from __future__ import annotations

import os
import requests
from datetime import datetime
from flask import send_from_directory, redirect
from ...image.copy_images import secure_filename
from ...cookies import get_cookies_dir
from ...errors import VersionNotFoundError
from ...constants import STATIC_URL, DIST_DIR
from ... import version

def redirect_home():
    return redirect('/chat')

def render(filename = "chat"):
    if os.path.exists(DIST_DIR):
        path = os.path.abspath(os.path.join(os.path.dirname(DIST_DIR), (filename + ("" if "." in filename else ".html"))))
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_dir = os.path.join(get_cookies_dir(), ".gui_cache")
    cache_file = os.path.join(cache_dir, f"{today}.{secure_filename(f'{filename}.{version.utils.current_version}-{latest_version}')}.html")
    if not os.path.exists(cache_file):
        os.makedirs(cache_dir, exist_ok=True)
        html = requests.get(f"{STATIC_URL}{filename}.html").text
        html = html.replace("../dist/", f"dist/")
        html = html.replace("\"dist/", f"\"{STATIC_URL}dist/")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(html)
    return send_from_directory(os.path.abspath(cache_dir), os.path.basename(cache_file))

class Website:
    def __init__(self, app) -> None:
        self.app = app
        self.routes = {
            '/': {
                'function': self._index,
                'methods': ['GET', 'POST']
            },
            '/chat/': {
                'function': self._chat,
                'methods': ['GET', 'POST']
            },
            '/qrcode.html': {
                'function': self._qrcode,
                'methods': ['GET', 'POST']
            },
            '/background.html': {
                'function': self._background,
                'methods': ['GET', 'POST']
            },
            '/chat/<conversation_id>': {
                'function': self._chat,
                'methods': ['GET', 'POST']
            },
            '/media/': {
                'function': redirect_home,
                'methods': ['GET', 'POST']
            },
            '/dist/<path:name>': {
                'function': self._dist,
                'methods': ['GET']
            },
        }

    def _index(self, filename = "home"):
        return render(filename)

    def _qrcode(self, filename = "qrcode"):
        return render(filename)

    def _background(self, filename = "background"):
        return render(filename)

    def _chat(self, filename = "chat"):
        filename = "chat/index" if filename == 'chat' else secure_filename(filename)
        return render(filename)

    def _dist(self, name: str):
        return send_from_directory(os.path.abspath(DIST_DIR), name)