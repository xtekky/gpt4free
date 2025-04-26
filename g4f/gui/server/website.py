from __future__ import annotations

import os
import requests
from datetime import datetime
from flask import send_from_directory, redirect, request
from ...image.copy_images import secure_filename, get_media_dir, ensure_media_dir
from ...errors import VersionNotFoundError
from ... import version

GPT4FREE_URL = "https://gpt4free.github.io"
DIST_DIR = "./gpt4free.github.io/dist"

def redirect_home():
    return redirect('/chat')

def render(filename = "chat", add_origion = True):
    if request.args.get("live"):
        add_origion = False
        if os.path.exists(DIST_DIR):
            path = os.path.abspath(os.path.join(os.path.dirname(DIST_DIR), (filename + ("" if "." in filename else ".html"))))
            print( f"Debug mode: {path}")
            return send_from_directory(os.path.dirname(path), os.path.basename(path))
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_file = os.path.join(get_media_dir(), f"{today}.{secure_filename(filename)}.{version.utils.current_version}-{latest_version}{'.live' if add_origion else ''}.html")
    if not os.path.exists(cache_file):
        ensure_media_dir()
        html = requests.get(f"{GPT4FREE_URL}/{filename}.html").text
        if add_origion:
            html = html.replace("../dist/", f"dist/")
            html = html.replace("\"dist/", f"\"{GPT4FREE_URL}/dist/")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(html)
    return send_from_directory(os.path.abspath(get_media_dir()), os.path.basename(cache_file))

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

    def _index(self, filename = "index"):
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