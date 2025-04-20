from __future__ import annotations

import os
import requests
from datetime import datetime
from flask import send_from_directory, redirect
from ...image.copy_images import secure_filename, get_media_dir, ensure_media_dir
from ...errors import VersionNotFoundError
from ... import version

GPT4FREE_URL = "https://gpt4free.github.io"

def redirect_home():
    return redirect('/chat')

def render(filename = "chat"):
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_file = os.path.join(get_media_dir(), f"{today}.{secure_filename(filename)}.{version.utils.current_version}-{latest_version}.html")
    if not os.path.exists(cache_file):
        ensure_media_dir()
        html = requests.get(f"{GPT4FREE_URL}/{filename}.html").text
        html = html.replace("../dist/", f"{GPT4FREE_URL}/dist/")
        html = html.replace('"dist/', f"\"{GPT4FREE_URL}/dist/")
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
