from __future__ import annotations

import os
import requests
from datetime import datetime
from urllib.parse import quote, unquote
from flask import send_from_directory, redirect, request

from ...image.copy_images import secure_filename
from ...cookies import get_cookies_dir
from ...errors import VersionNotFoundError
from ...config import STATIC_URL, DOWNLOAD_URL, DIST_DIR, JSDELIVR_URL, GITHUB_URL
from ... import version

def redirect_home():
    return redirect('/chat/')

def render(filename = "home", download_url: str = GITHUB_URL):
    if download_url == GITHUB_URL:
        filename += ("" if "." in filename else ".html")
    html = None
    is_temp = False
    if os.path.exists(DIST_DIR) and not request.args.get("debug"):
        path = os.path.abspath(os.path.join(os.path.dirname(DIST_DIR), filename))
        if os.path.exists(path):
            if download_url == GITHUB_URL:
                with open(path, 'r', encoding='utf-8') as f:
                    html = f.read()
                is_temp = True
            else:
                return send_from_directory(os.path.dirname(path), os.path.basename(path))
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_dir = os.path.join(get_cookies_dir(), ".gui_cache", today)
    latest_version = str(latest_version) +quote(unquote(request.query_string.decode())) or str(latest_version)
    cache_file = os.path.join(cache_dir, f"{secure_filename(f'{version.utils.current_version}-{latest_version}')}.{secure_filename(filename)}")
    if os.path.isfile(cache_file + ".js"):
        cache_file += ".js"
    if not os.path.exists(cache_file):
        if os.access(cache_file, os.W_OK):
            is_temp = True
        else:
            os.makedirs(cache_dir, exist_ok=True)
        if html is None:
            try:
                response = requests.get(f"{download_url}{filename}")
                response.raise_for_status()
            except requests.RequestException:
                try:
                    response = requests.get(f"{DOWNLOAD_URL}{filename}")
                    response.raise_for_status()
                except requests.RequestException:
                    found = None
                    for root, _, files in os.walk(cache_dir):
                        for file in files:
                            if file.startswith(secure_filename(filename)):
                                found = os.path.abspath(root), file
                        break
                    if found:
                        return send_from_directory(found[0], found[1])
                    else:
                        raise
            if not cache_file.endswith(".js") and response.headers.get("Content-Type", "").startswith("application/javascript"):
                cache_file += ".js"
            html = response.text
            html = html.replace("../dist/", f"dist/")
            html = html.replace("/dist/", f"dist/")
            html = html.replace(f"{STATIC_URL}dist/", "dist/")
            html = html.replace("dist/", f"{STATIC_URL}dist/")
        # html = html.replace(JSDELIVR_URL, "/")
        html = html.replace("{{ v }}", latest_version)
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
            '/private/': {
                'function': self._private,
                'methods': ['GET', 'POST']
            },
            '/private/<path:filename>': {
                'function': self._private,
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
            '/gh/<path:name>': {
                'function': self._gh,
                'methods': ['GET']
            },
            '/npm/<path:name>': {
                'function': self._npm,
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

    def _private(self, filename = ""):
        filename = f"private/{filename}" if filename else "private/index"
        return render(filename)

    def _dist(self, name: str):
        return send_from_directory(os.path.abspath(DIST_DIR), name)

    def _gh(self, name):
        return render(f"gh/{name}", JSDELIVR_URL)

    def _npm(self, name):
        return render(f"npm/{name}", JSDELIVR_URL)