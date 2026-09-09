from __future__ import annotations

import asyncio
import os
import inspect
import requests
import re
import time
import hashlib
from datetime import datetime
from urllib.parse import quote_plus
from flask import jsonify, send_from_directory, redirect, request

from ...files import secure_filename
from ...cookies import get_cookies_dir
from ...errors import VersionNotFoundError
from ...config import STATIC_URL, DOWNLOAD_URL, DIST_DIR, GITHUB_URL
from ... import version

_gui_session = requests.Session()
_CONTENT_PATTERN = re.compile(r"<!-- CONTENT_START -->.*?<!-- CONTENT_END -->", re.DOTALL)
_providers_cache: list[dict] | None = None
_providers_cache_time: float = 0.0
_providers_cards_cache: str | None = None
_PROVIDERS_TTL: float = 300.0
_template_cache: dict[str, str] = {}


def redirect_home():
    return redirect("/chat/")


def render(filename="home", download_url: str = GITHUB_URL):
    if download_url == GITHUB_URL:
        filename += "" if "." in filename else ".html"
    html = None
    is_temp = False
    if os.path.exists(DIST_DIR) and not request.args.get("debug"):
        base_dir = os.path.abspath(os.path.dirname(DIST_DIR))
        path = os.path.abspath(os.path.join(base_dir, filename))
        if not path.startswith(base_dir + os.sep) and path != base_dir:
            return redirect("/")
        if os.path.exists(path):
            if path.endswith(".html"):
                try:
                    latest_version = version.utils.latest_version
                except VersionNotFoundError:
                    latest_version = version.utils.current_version
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                return html.replace("{{ v }}", str(latest_version))
            return send_from_directory(
                os.path.dirname(path), os.path.basename(path), max_age=31536000
            )
    try:
        latest_version = version.utils.latest_version
    except VersionNotFoundError:
        latest_version = version.utils.current_version
    today = datetime.today().strftime("%Y-%m-%d")
    cache_dir = os.path.join(get_cookies_dir(), ".gui_cache", today)
    qs_suffix = ""
    if not request.args.get("g4f_session") and request.query_string:
        qs_suffix = "_" + hashlib.md5(request.query_string).hexdigest()[:8]
    safe_filename = secure_filename(os.path.basename(filename))
    safe_prefix = secure_filename(f"{version.utils.current_version}-{latest_version}")
    cache_file_name = f"{safe_prefix}{qs_suffix}.{safe_filename}"
    real_cache_dir = os.path.realpath(cache_dir)
    cache_file = os.path.realpath(os.path.join(cache_dir, cache_file_name))
    if not cache_file.startswith(real_cache_dir + os.sep):
        raise ValueError("Invalid cache path")
    if os.path.isfile(cache_file + ".js"):
        cache_file += ".js"
    if not os.path.exists(cache_file):
        if os.access(cache_file, os.W_OK):
            is_temp = True
        else:
            os.makedirs(cache_dir, exist_ok=True)
        if html is None:
            try:
                response = _gui_session.get(f"{download_url}{filename}", timeout=10)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                response = _gui_session.get(f"{download_url}{filename}", timeout=10, verify=False)
                response.raise_for_status()
            except requests.RequestException:
                try:
                    response = _gui_session.get(f"{DOWNLOAD_URL}{filename}", timeout=10)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    response = _gui_session.get(f"{DOWNLOAD_URL}{filename}", timeout=10, verify=False)
                    response.raise_for_status()
                except requests.RequestException:
                    found = None
                    for root, _, files in os.walk(cache_dir):
                        for file in files:
                            if file.startswith(secure_filename(filename)):
                                found = os.path.abspath(root), file
                                break
                        if found:
                            break
                    if found:
                        return send_from_directory(found[0], found[1], max_age=31536000)
                    else:
                        raise
            if not cache_file.endswith(".js") and response.headers.get(
                "Content-Type", ""
            ).startswith("application/javascript"):
                cache_file += ".js"
            if filename.endswith(".html"):
                html = response.text
                dist_url = (
                    "/dist/" if os.path.exists(DIST_DIR) else f"{STATIC_URL}dist/"
                )
                html = html.replace("'../dist/", f"'{dist_url}")
                html = html.replace("'/dist/", f"'{dist_url}")
                html = html.replace("'dist/", f"'{dist_url}")
                html = html.replace('<base href="/">', f'<base href="/sillytavern/">')
        if html is None:
            with open(cache_file, "wb") as f:
                f.write(response.content)
        else:
            html = html.replace("{{ v }}", latest_version)
            if is_temp:
                return html
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(html)
    return send_from_directory(
        os.path.abspath(cache_dir), os.path.basename(cache_file), max_age=31536000
    )


class Website:
    def __init__(self, app) -> None:
        self.app = app
        self.routes = {
            "/": {"function": self._index, "methods": ["GET", "POST"]},
            "/chat/": {"function": self._chat, "methods": ["GET", "POST"]},
            "/<filename>.html": {"function": self._index, "methods": ["GET", "POST"]},
            "/chat/<filename>": {"function": self._chat, "methods": ["GET", "POST"]},
            "/private/": {"function": self._private, "methods": ["GET", "POST"]},
            "/private/<path:filename>": {
                "function": self._private,
                "methods": ["GET", "POST"],
            },
            "/media/": {"function": redirect_home, "methods": ["GET", "POST"]},
            "/dist/<path:name>": {"function": self._dist, "methods": ["GET"]},
            "/playground/": {"function": self._playground, "methods": ["GET"]},
            "/playground/<path:filename>": {
                "function": self._playground,
                "methods": ["GET"],
            },
            "/apps/": {"function": self._apps, "methods": ["GET"]},
            "/apps/<path:filename>": {"function": self._apps, "methods": ["GET"]},
            "/stats/": {"function": self._stats, "methods": ["GET"]},
            "/providers/": {"function": self._providers, "methods": ["GET"]},
            "/providers/<name>": {"function": self._provider_detail, "methods": ["GET"]},
        }

        @app.route("/lib.js", methods=["GET"])
        def lib_js():
            return self._sillytavern("lib.js")

        @app.route("/script.js", methods=["GET"])
        def script_js():
            return self._sillytavern("script.js")

        @app.route("/lib/<path:filename>", methods=["GET"])
        def lib_files(filename):
            return self._sillytavern(f"lib/{filename}")

        @app.route("/scripts/<path:filename>", methods=["GET"])
        def script_files(filename):
            return self._sillytavern(f"scripts/{filename}")

    def _index(self, filename="home"):
        return render(filename)

    def _stats(self):
        return render("stats")

    def _get_providers(self):
        """Load all providers and return a list of dicts with their attributes (cached with 300s TTL)."""
        global _providers_cache, _providers_cache_time
        now = time.time()
        if _providers_cache is not None and (now - _providers_cache_time) < _PROVIDERS_TTL:
            return _providers_cache

        from g4f.Provider import ProviderLoader

        providers = []
        for name in ProviderLoader.names:
            try:
                provider = ProviderLoader.from_name(name)
                url = getattr(provider, "url", None)
                screenshot_url = getattr(provider, "screenshot_url", None)
                login_url = getattr(provider, "login_url", None)
                # Skip model list fetching here — it's slow for 108+ providers.
                # Models are loaded lazily in _provider_detail() for a single provider.
                models = getattr(provider, "models", None)
                if callable(models):
                    models = None
                needs_auth = getattr(provider, "needs_auth", False)
                working = getattr(provider, "working", False)
                supports_stream = getattr(provider, "supports_stream", False)
                supports_message_history = getattr(provider, "supports_message_history", False)
                supports_system_message = getattr(provider, "supports_system_message", False)
                params = getattr(provider, "params", [])
                if callable(params):
                    try:
                        params = params()
                    except Exception:
                        params = []
                providers.append({
                    "name": name,
                    "url": url,
                    "screenshot_url": screenshot_url,
                    "login_url": login_url,
                    "models": models if isinstance(models, list) else list(models) if models else [],
                    "needs_auth": needs_auth,
                    "working": working,
                    "supports_stream": supports_stream,
                    "supports_message_history": supports_message_history,
                    "supports_system_message": supports_system_message,
                    "params": params if isinstance(params, list) else list(params) if params else [],
                })
            except Exception:
                raise
        _providers_cache = providers
        _providers_cache_time = now
        return providers

    def _providers(self):
        global _providers_cards_cache
        providers = self._get_providers()

        template_path = os.path.join(os.path.dirname(__file__), "providers.html")
        if not os.path.exists(template_path):
            return "Providers template not found"

        if template_path not in _template_cache:
            with open(template_path, "r", encoding="utf-8") as f:
                _template_cache[template_path] = f.read()
        html = _template_cache[template_path]

        if _providers_cards_cache is None or (time.time() - _providers_cache_time) >= _PROVIDERS_TTL:
            # Build HTML cards
            cards_html = """
            <div class="page-header">
                <h1>Available Providers</h1>
                <p>Browse the list of AI providers supported by G4F</p>
            </div>

            <div class="providers-list">
            """
            for p in providers:
                if not p["working"]:
                    continue  # Skip non-working providers
                models_html = ""
                if p["models"]:
                    models_list = ", ".join(p["models"][:5]) if isinstance(p["models"], list) else ""
                    if len(p["models"]) > 5:
                        models_list += f" (+{len(p['models']) - 5} more)"
                    models_html = f"<div class='provider-details'><strong>Models:</strong> {models_list}</div>"
                else:
                    models_html = "<div class='provider-details'><em>No specific models</em></div>"

                url_html = f"<div class='provider-url'>{p['url']}</div>" if p["url"] else ""
                auth_html = "<div class='provider-details'><strong>Auth:</strong> Required</div>" if p["needs_auth"] else ""
                working_html = "<div class='provider-details'><strong>Status:</strong> Working</div>" if p["working"] else ""

                cards_html += f"""
                <div class="provider-card" onclick="window.location.href='/providers/{p['name']}'">
                    <div class="provider-name">{p['name']}</div>
                    {url_html}
                    {models_html}
                    {auth_html}
                    {working_html}
                    <div class="provider-actions">
                        <a href="/providers/{p['name']}" class="btn btn-primary">Details</a>
                        <a href="{p['url']}" target="_blank" class="btn btn-secondary">Website</a>
                    </div>
                </div>
                """
            cards_html += "\n        </div>"
            _providers_cards_cache = cards_html
        else:
            cards_html = _providers_cards_cache

        return _CONTENT_PATTERN.sub(f"<!-- CONTENT_START -->{cards_html}<!-- CONTENT_END -->", html)

    def _provider_detail(self, name: str = ""):
        from html import escape

        providers = self._get_providers()
        names = [p["name"] for p in providers]

        # Find the current provider (case-insensitive)
        idx = None
        for i, n in enumerate(names):
            if n.lower() == name.lower():
                idx = i
                break

        if idx is None:
            return self._providers()

        p = providers[idx]
        working_providers = [p for p in providers if p["working"]]
        prev_p = working_providers[idx - 1] if idx > 0 and idx < len(working_providers) - 1 else working_providers[-1]
        next_p = working_providers[idx + 1] if idx < len(working_providers) - 1 else working_providers[0]

        # Lazily load models for this single provider only
        from g4f.Provider import ProviderLoader
        models = []
        try:
            provider = ProviderLoader.from_name(p["name"])
            raw_models = getattr(provider, "models", None)
            if callable(raw_models):
                try:
                    raw_models = raw_models()
                except Exception:
                    raw_models = []
            if raw_models:
                models = raw_models if isinstance(raw_models, list) else list(raw_models)
            else:
                # Fall back to get_models() if available
                get_models = getattr(provider, "get_models", None)
                if callable(get_models):
                    try:
                        raw_models = get_models()
                        if inspect.isawaitable(raw_models):
                            import asyncio as _aio
                            try:
                                raw_models = _aio.get_event_loop().run_until_complete(raw_models)
                            except RuntimeError:
                                raw_models = _aio.new_event_loop().run_until_complete(raw_models)
                        models = list(raw_models) if raw_models else []
                    except Exception:
                        models = []
        except Exception:
            pass

        if models:
            models_html = "<ul class='model-list'>" + "".join(
                f"<li>{escape(str(m))}</li>" for m in models
            ) + "</ul>"
        else:
            models_html = "<p><em>No specific models listed</em></p>"

        # Build params list HTML
        if p["params"]:
            params_html = "<ul class='param-list'>" + "".join(
                f"<li>{escape(str(param))}</li>" for param in p["params"]
            ) + "</ul>"
        else:
            params_html = "<p><em>None</em></p>"

        # Build attributes table
        attrs_html = f"""
        <table class="attr-table">
            <tr><th>URL</th><td><a href="{escape(p['url'] or '')}" target="_blank">{escape(p['url'] or 'N/A')}</a></td></tr>
            <tr><th>Working</th><td>{'✅ Yes' if p['working'] else '❌ No'}</td></tr>
            <tr><th>Needs Auth</th><td>{'🔒 Yes' if p['needs_auth'] else '🔓 No'}</td></tr>
            <tr><th>Supports Stream</th><td>{'✅ Yes' if p['supports_stream'] else '❌ No'}</td></tr>
            <tr><th>Supports Message History</th><td>{'✅ Yes' if p['supports_message_history'] else '❌ No'}</td></tr>
            <tr><th>Supports System Message</th><td>{'✅ Yes' if p['supports_system_message'] else '❌ No'}</td></tr>
        </table>
        """

        # Screenshot / logo section
        screenshot_url = p.get("screenshot_url") or p.get("url") or ""
        create_url = f"/screenshot?url={quote_plus(str(screenshot_url))}"
        screenshot_url = screenshot_url.replace("https://", "").replace("http://", "").replace("www.", "")
        logo_url = "https://g4f.space/logo/" + p.get("name", "").replace(
            'MetaAIAccount', 'Facebook AI').replace(
            'MetaAI', 'Facebook AI').replace(
            'Aria', '').replace(
            'OpenAI', 'ChatGPT').replace(
            'Video', 'TV').replace(
            'Phi-4', 'Windows').replace(
            '(Text Generation)', '').replace(
            'Glhf', 'AI').replace(
            'GithubCopilot', 'GitHub Copilot').replace(
            'PerplexityApi', 'Perplexity API').replace(
            'Gemini', '').replace(
            'API', '').replace(
            '-2.5M', '').replace(
            'grok', 'xAI').replace(
            'Qwen_Qwen_3', 'Qwen').replace(
            'Yupp', 'with yupp').replace(
            'groq', 'Groq')
        screenshot_html = f"""
        <div class="screenshot-section">
            <img src="/screenshot/{quote_plus(screenshot_url)}.webp" data-src="{create_url}" alt="{escape(p['name'])} logo" class="provider-logo"
                 style="max-width:100%;border-radius:8px;border:1px solid var(--card-border)" />
            <img src="{logo_url}" alt="{escape(p['name'])} logo" class="provider-logo" style="max-width:100%;border-radius:8px;border:1px solid var(--card-border)" />
            <p class="screenshot-caption">Load screenshot from {escape(p['url'] or 'N/A')}</p>
        </div>
        <script>
            const img = document.querySelector('img[data-src="{create_url}"]');
            const input = document.createElement('input');
            const previewImg = document.querySelector('img[src="{logo_url}"]');
            img.parentElement.appendChild(previewImg);
            let n = 1;
            let previewRemoved = false;
            const createSrc = img.dataset.src;
            img.onload = () => {{
                input.placeholder = 'Ask {escape(p.get("label", p["name"]))}';
                if (!previewRemoved && previewImg.parentNode) {{
                    previewImg.parentNode.removeChild(previewImg);
                    previewRemoved = true;
                    document.querySelector('.screenshot-caption').style.display = 'none';
                }}
            }};
            img.onmouseenter = () => {{
                if (!previewRemoved || !img.complete) return;
                n = (n % 3) + 1;
                if (n <= 3) {{
                    const append = n == 1 ? ".webp" : `_${{n}}.webp`;
                    img.src = `/screenshot/{quote_plus(screenshot_url)}${{append}}`;
                }}
            }};
            img.onmouseleave = img.onmouseenter;
            img.ontouchstart = img.onmouseenter;
            img.onerror = () => {{
                input.placeholder = 'Ask {escape(p.get("label", p["name"]))}';
                if (img.src.includes(createSrc)) {{
                    img.parentNode.removeChild(img);
                    return;
                }}
                if (n === 1) {{
                    img.src = createSrc;
                    return;
                }}
                n = 3; // Stop carousel on error
                img.src = createSrc;
            }};
            input.type = 'text';
            input.placeholder = 'Ask {escape(p.get("label", p["name"]))}';
            input.className = 'provider-input';
            input.addEventListener('change', function(event) {{
                if (!event.target.value) {{
                    img.src = createSrc;
                    return;
                }}
                let newUrl = '';
                if ('{p["name"]}' == 'YouTube') {{
                    const createUrl = new URL('{create_url}', location.origin);
                    const queryUrl = new URL(createUrl.searchParams.get('url'));
                    queryUrl.pathname = "/results";
                    queryUrl.searchParams.set('search_query', event.target.value);
                    newUrl = "/screenshot?url=" + encodeURIComponent(queryUrl.toString());
                }} else if ('{create_url}'.includes('q=Hello')) {{
                    newUrl = "{create_url}".replace("q=Hello", "q=" + encodeURIComponent(event.target.value));
                }} else {{
                    newUrl = createSrc + encodeURIComponent('?q=' + event.target.value);
                }}
                if (img.src !== newUrl) {{
                    img.src = newUrl;
                }}
                input.value = '';
                input.placeholder = 'Is Loading...';
            }});
            const inputContainer = document.createElement('div');
            inputContainer.className = 'provider-input-container';
            inputContainer.appendChild(input);
            img.parentElement.appendChild(inputContainer);
        </script>
        """

        detail_html = f"""
        <div class="detail-header">
            <h1>{escape(p['name'])}</h1>
            <p class="detail-subtitle">Provider #{idx + 1} of {len(providers)}</p>
        </div>

        <div class="nav-prev-next">
            <a href="/providers/{escape(prev_p['name'])}" class="nav-btn nav-prev">
                ← {escape(prev_p['name'])}
            </a>
            <a href="/providers/" class="nav-btn nav-back">All Providers</a>
            <a href="/providers/{escape(next_p['name'])}" class="nav-btn nav-next">
                {escape(next_p['name'])} →
            </a>
        </div>

        <div class="detail-grid">
            <div class="detail-card">
                <h2>Attributes</h2>
                {attrs_html}
            </div>
            <div class="detail-card">
                <h2>Logo / Screenshot</h2>
                {screenshot_html}
            </div>
        </div>

        <div class="detail-card detail-models">
            <h2>Models ({len(p['models'])})</h2>
            {models_html}
        </div>

        <div class="detail-card">
            <h2>Parameters</h2>
            {params_html}
        </div>

        <div class="nav-prev-next" style="margin-top:2rem">
            <a href="/providers/{escape(prev_p['name'])}" class="nav-btn nav-prev">
                ← {escape(prev_p['name'])}
            </a>
            <a href="/providers/" class="nav-btn nav-back">All Providers</a>
            <a href="/providers/{escape(next_p['name'])}" class="nav-btn nav-next">
                {escape(next_p['name'])} →
            </a>
        </div>
        """

        # Read the template and inject detail content
        template_path = os.path.join(os.path.dirname(__file__), "providers.html")
        if not os.path.exists(template_path):
            return "Providers template not found"

        if template_path not in _template_cache:
            with open(template_path, "r", encoding="utf-8") as f:
                _template_cache[template_path] = f.read()
        html = _template_cache[template_path]

        html = _CONTENT_PATTERN.sub(f"<!-- CONTENT_START -->{detail_html}<!-- CONTENT_END -->", html)
        html = html.replace(
            "<title>Providers</title>",
            f"<title>{escape(p['name'])} – Provider Details</title>"
        )
        return html

    def _chat(self, filename=""):
        filename = f"chat/{filename}" if filename else "chat/index"
        return render(filename)

    def _private(self, filename=""):
        filename = f"private/{filename}" if filename else "private/index"
        return render(filename)

    def _dist(self, name: str):
        return render(f"dist/{name}")

    def _apps(self, filename: str = "index.html"):
        return render(f"apps/{filename}")

    def _playground(self, filename: str = "index.html"):
        PLAYGROUND_URL = (
            "https://raw.githubusercontent.com/gpt4free/playground/refs/heads/main/"
        )
        if not filename or filename.endswith("/"):
            filename = "index.html"
        filename += "" if "." in filename else ".html"
        # Serve from local ./playground directory if present
        local_dir = os.path.abspath("./playground")
        local_path = os.path.normpath(os.path.join(local_dir, filename))
        if local_path.startswith(local_dir + os.sep) and os.path.isfile(local_path):
            return send_from_directory(
                os.path.dirname(local_path),
                os.path.basename(local_path),
                max_age=31536000,
            )
        # Use cache dir
        cache_dir = os.path.join(get_cookies_dir(), ".playground_cache")
        safe_path = os.path.normpath(os.path.join(cache_dir, filename))
        if not safe_path.startswith(cache_dir + os.sep) and safe_path != cache_dir:
            return jsonify({"error": "Invalid filename"}), 400
        # Serve from cache if present
        if os.path.isfile(safe_path):
            return send_from_directory(
                os.path.dirname(safe_path),
                os.path.basename(safe_path),
                max_age=31536000,
            )
        # Download and cache from GitHub
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        try:
            response = _gui_session.get(f"{PLAYGROUND_URL}{filename}", timeout=10)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            try:
                response = _gui_session.get(
                    f"{PLAYGROUND_URL}{filename}", timeout=10, verify=False
                )
                response.raise_for_status()
            except requests.RequestException:
                pass
        except requests.RequestException:
            pass

        if "response" in locals() and response.status_code == 200:
            with open(safe_path, "wb") as f:
                f.write(response.content)
            return send_from_directory(
                os.path.dirname(safe_path),
                os.path.basename(safe_path),
                max_age=31536000,
            )
        # SPA fallback: serve index.html for unknown sub-paths
        index_path = os.path.join(cache_dir, "index.html")
        if os.path.isfile(index_path):
            return send_from_directory(cache_dir, "index.html")
        local_index = os.path.join(local_dir, "index.html")
        if os.path.isfile(local_index):
            return send_from_directory(local_dir, "index.html")
        return redirect("https://gpt4free.github.io/playground")
