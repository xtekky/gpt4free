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
        base_dir = os.path.abspath(os.path.dirname(DIST_DIR))
        path = os.path.abspath(os.path.join(base_dir, filename))
        if not path.startswith(base_dir + os.sep) and path != base_dir:
            return redirect('/')
        if os.path.exists(path):
            return send_from_directory(os.path.dirname(path), os.path.basename(path))
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime('%Y-%m-%d')
    cache_dir = os.path.join(get_cookies_dir(), ".gui_cache", today)
    if not request.args.get("g4f_session"):
        latest_version = str(latest_version) + quote(unquote(request.query_string.decode()))
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
            dist_url = "/dist/" if os.path.exists(DIST_DIR) else f"{STATIC_URL}dist/"
            html = html.replace('"../dist/', f'"{dist_url}')
            html = html.replace('"/dist/', f'"{dist_url}')
            html = html.replace('"dist/', f'"{dist_url}')
            html = html.replace("'../dist/", f"'{dist_url}")
            html = html.replace("'/dist/", f"'{dist_url}")
            html = html.replace("'dist/", f"'{dist_url}")
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
            '/home.html': {
                'function': self._home,
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
            '/playground/': {
                'function': self._playground,
                'methods': ['GET']
            },
            '/playground/<path:filename>': {
                'function': self._playground,
                'methods': ['GET']
            },
            '/apps/': {
                'function': self._apps,
                'methods': ['GET']
            },
            '/apps/<path:filename>': {
                'function': self._apps,
                'methods': ['GET']
            },
        }

    def _index(self, filename = "home"):
        return render(filename)

    def _qrcode(self, filename = "qrcode"):
        return render(filename)

    def _background(self, filename = "background"):
        return render(filename)

    def _home(self, filename = "home"):
        return render(filename)

    def _chat(self, filename = ""):
        filename = f"chat/{filename}" if filename else "chat/index"
        return render(filename)

    def _private(self, filename = ""):
        filename = f"private/{filename}" if filename else "private/index"
        return render(filename)

    def _dist(self, name: str):
        return render(f"dist/{name}")
    
    def _apps(self, filename: str = "index.html"):
        return render(f"apps/{filename}")

    def _playground(self, filename: str = "index.html"):
        PLAYGROUND_URL = "https://raw.githubusercontent.com/gpt4free/playground/refs/heads/main/"
        if not filename or filename.endswith("/"):
            filename = "index.html"
        filename += ("" if "." in filename else ".html")
        # Serve from local ./playground directory if present
        local_dir = os.path.abspath("./playground")
        local_path = os.path.normpath(os.path.join(local_dir, filename))
        if local_path.startswith(local_dir + os.sep) and os.path.isfile(local_path):
            return send_from_directory(os.path.dirname(local_path), os.path.basename(local_path))
        # Use cache dir
        cache_dir = os.path.join(get_cookies_dir(), ".playground_cache")
        safe_path = os.path.normpath(os.path.join(cache_dir, filename))
        if not safe_path.startswith(cache_dir + os.sep) and safe_path != cache_dir:
            return redirect("/playground/")
        # Serve from cache if present
        if os.path.isfile(safe_path):
            return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
        # Download and cache from GitHub
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        try:
            response = requests.get(f"{PLAYGROUND_URL}{filename}", timeout=10)
            response.raise_for_status()
            with open(safe_path, 'wb') as f:
                f.write(response.content)
            return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))
        except requests.RequestException:
            pass
        # SPA fallback: serve index.html for unknown sub-paths
        index_path = os.path.join(cache_dir, "index.html")
        if os.path.isfile(index_path):
            return send_from_directory(cache_dir, "index.html")
        local_index = os.path.join(local_dir, "index.html")
        if os.path.isfile(local_index):
            return send_from_directory(local_dir, "index.html")
        return redirect("https://gpt4free.github.io/playground")