from __future__ import annotations

import os
import requests
from datetime import datetime
from flask import send_from_directory, redirect, request

from ...image.copy_images import secure_filename
from ...cookies import get_cookies_dir
from ...errors import VersionNotFoundError
from ...config import STATIC_URL, DOWNLOAD_URL, DIST_DIR
from ... import version

def redirect_home():
    return redirect('/chat/')

def render(filename = "home"):
    filename += ("" if "." in filename else ".html")
    if os.path.exists(DIST_DIR) and not request.args.get("debug"):
        path = os.path.abspath(os.path.join(os.path.dirname(DIST_DIR), filename))
        return send_from_directory(os.path.dirname(path), os.path.basename(path))
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_dir = os.path.join(get_cookies_dir(), ".gui_cache")
    cache_file = os.path.join(cache_dir, f"{secure_filename(filename)}.{today}.{secure_filename(f'{version.utils.current_version}-{latest_version}')}.html")
    is_temp = False
    if not os.path.exists(cache_file):
        if os.access(cache_file, os.W_OK):
            is_temp = True
        else:
            os.makedirs(cache_dir, exist_ok=True)
        response = requests.get(f"{DOWNLOAD_URL}{filename}")
        if not response.ok:
            found = None
            for root, _, files in os.walk(cache_dir):
                for file in files:
                    if file.startswith(secure_filename(filename)):
                        found = os.path.abspath(root), file
                break
            if found:
                return send_from_directory(found[0], found[1])
            else:
                response.raise_for_status()
        html = response.text
        html = html.replace("../dist/", f"dist/")
        html = html.replace("\"dist/", f"\"{STATIC_URL}dist/")
        if is_temp:
            return html
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
            '/chat/<filename>': {
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

    def _chat(self, filename = ""):
        filename = f"chat/{filename}" if filename else "chat/index"
        return render(filename)

    def _dist(self, name: str):
        return send_from_directory(os.path.abspath(DIST_DIR), name)