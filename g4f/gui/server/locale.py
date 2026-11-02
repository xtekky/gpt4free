"""Locale-aware URLs, SEO metadata, and runtime translation bootstrapping."""

import json
import re
from html import escape
from urllib.parse import urljoin, urlsplit, urlunsplit

from flask import request


BASE_URL = "https://g4f.space"
LOCALES = ("en", "es", "fr", "de")
PAGES = ("home", "providers", "chat", "playground")

# Server-rendered metadata for the pages we can describe in every locale.
COPY = {
    "en": {
        "home": ("G4F - Framework & Tools Hub", "Explore free AI tools and models with G4F.", "G4F AI Tools & Dashboard"),
        "providers": ("AI Providers - GPT4Free", "Browse AI providers and free models supported by G4F.", "Available Providers"),
        "chat": ("G4F Chat - Free AI Chat", "Chat with free AI models using G4F.", "G4F Chat"),
        "playground": ("AI Playground - G4F", "Experiment with AI chat, roleplay, coding, and providers in the G4F playground.", "Playground"),
    },
    "es": {
        "home": ("G4F - Herramientas de IA", "Explora herramientas y modelos de IA gratis con G4F.", "Herramientas de IA de G4F"),
        "providers": ("Proveedores de IA - GPT4Free", "Explora proveedores de IA y modelos gratis de G4F.", "Proveedores disponibles"),
        "chat": ("G4F Chat - Chat de IA gratis", "Chatea con modelos de IA gratis usando G4F.", "Chat de G4F"),
        "playground": ("Playground de IA - G4F", "Experimenta con chat, rol, código y proveedores de IA en el playground de G4F.", "Playground"),
    },
    "fr": {
        "home": ("G4F - Outils d'IA", "Explorez les outils et modèles d'IA gratuits avec G4F.", "Outils d'IA G4F"),
        "providers": ("Fournisseurs d'IA - GPT4Free", "Découvrez les fournisseurs d'IA et modèles gratuits de G4F.", "Fournisseurs disponibles"),
        "chat": ("G4F Chat - Chat IA gratuit", "Discutez avec des modèles d'IA gratuits avec G4F.", "Chat G4F"),
        "playground": ("Playground IA - G4F", "Expérimentez le chat, le roleplay, le code et les fournisseurs d'IA dans le playground G4F.", "Playground"),
    },
    "de": {
        "home": ("G4F - KI-Werkzeuge", "Entdecke kostenlose KI-Werkzeuge und Modelle mit G4F.", "G4F KI-Werkzeuge"),
        "providers": ("KI-Anbieter - GPT4Free", "Entdecke KI-Anbieter und kostenlose Modelle von G4F.", "Verfügbare Anbieter"),
        "chat": ("G4F Chat - Kostenloser KI-Chat", "Chatte mit kostenlosen KI-Modellen über G4F.", "G4F Chat"),
        "playground": ("KI-Playground - G4F", "Experimentiere mit KI-Chat, Rollenspiel, Code und Anbietern im G4F-Playground.", "Playground"),
    },
}

# Static visible copy. Lookups are whitespace-normalised and preserve the
# surrounding whitespace, matching framework.translate() in framework.js.
COMMON_COPY = {
    "es": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Accede a herramientas de IA, búsquedas web, registros en tiempo real, estado de modelos y servicios de varios proveedores.",
        "Browse the list of AI providers supported by G4F": "Explora los proveedores de IA compatibles con G4F",
        "Loading providers...": "Cargando proveedores...",
        "No specific models": "Sin modelos específicos",
        "Backend Tools & Modules": "Herramientas y módulos",
        "AI Web Browser": "Navegador web de IA",
        "Backend Logs": "Registros del servidor",
        "Usage & Statistics": "Uso y estadísticas",
        "System & Provider Status": "Estado del sistema y proveedores",
        "Providers Catalog": "Catálogo de proveedores",
        "Home": "Inicio", "Providers": "Proveedores", "Stats": "Estadísticas",
        "Details": "Detalles", "Website": "Sitio web", "Models:": "Modelos:",
        "Auth:": "Acceso:", "Required": "Obligatorio", "Status:": "Estado:", "Working": "Disponible",
        "Language": "Idioma", "Translate UI": "Traducir interfaz",
    },
    "fr": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Accédez aux outils d'IA, à la recherche web, aux journaux en temps réel, à l'état des modèles et aux services de plusieurs fournisseurs.",
        "Browse the list of AI providers supported by G4F": "Découvrez les fournisseurs d'IA compatibles avec G4F",
        "Loading providers...": "Chargement des fournisseurs...",
        "No specific models": "Aucun modèle spécifique",
        "Backend Tools & Modules": "Outils et modules",
        "AI Web Browser": "Navigateur web d'IA",
        "Backend Logs": "Journaux du serveur",
        "Usage & Statistics": "Utilisation et statistiques",
        "System & Provider Status": "État du système et des fournisseurs",
        "Providers Catalog": "Catalogue des fournisseurs",
        "Home": "Accueil", "Providers": "Fournisseurs", "Stats": "Statistiques",
        "Details": "Détails", "Website": "Site web", "Models:": "Modèles :",
        "Auth:": "Accès :", "Required": "Obligatoire", "Status:": "État :", "Working": "Disponible",
        "Language": "Langue", "Translate UI": "Traduire l'interface",
    },
    "de": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Nutze KI-Werkzeuge, Websuche, Echtzeitprotokolle, Modellstatus und Dienste mehrerer Anbieter.",
        "Browse the list of AI providers supported by G4F": "Entdecke die von G4F unterstützten KI-Anbieter",
        "Loading providers...": "Anbieter werden geladen...",
        "No specific models": "Keine bestimmten Modelle",
        "Backend Tools & Modules": "Werkzeuge und Module",
        "AI Web Browser": "KI-Webbrowser",
        "Backend Logs": "Serverprotokolle",
        "Usage & Statistics": "Nutzung und Statistiken",
        "System & Provider Status": "System- und Anbieterstatus",
        "Providers Catalog": "Anbieterkatalog",
        "Home": "Startseite", "Providers": "Anbieter", "Stats": "Statistiken",
        "Details": "Details", "Website": "Webseite", "Models:": "Modelle:",
        "Auth:": "Zugang:", "Required": "Erforderlich", "Status:": "Status:", "Working": "Verfügbar",
        "Language": "Sprache", "Translate UI": "Oberfläche übersetzen",
    },
}

_LOCALE_PREFIX = re.compile(r"^/([A-Za-z]{2})(?=/|$)")
_PROTECTED_BLOCKS = re.compile(r"(<script\b.*?</script\s*>|<style\b.*?</style\s*>|<!--.*?-->)", re.I | re.S)
_TEXT_NODE = re.compile(r">([^<]+)<", re.S)
_TAG = re.compile(r"<[a-zA-Z][^>]*>", re.S)
_URL_ATTR = re.compile(r'(\s(?:href|src|action|poster)\s*=\s*)(["\'])(.*?)\2', re.I | re.S)
_SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")
_RUNTIME_MARKER = "data-g4f-locale-runtime"


def normalize_text(text):
    """Collapse whitespace like framework.translate() before a lookup."""
    return re.sub(r"\s+", " ", text or "").strip()


def page_path(page, locale="en"):
    path = {
        "home": "/",
        "chat": "/chat/",
        "playground": "/playground/",
    }.get(page, f"/{page}/")
    return f"/{locale}{path}" if locale != "en" else path


def page_for_path(path):
    """Map an unprefixed GUI path to the page metadata it belongs to."""
    clean = (path or "/").split("?", 1)[0].split("#", 1)[0].rstrip("/") or "/"
    if clean in ("/", "/index.html", "/home.html"):
        return "home"
    if clean in ("/providers", "/providers.html"):
        return "providers"
    if clean in ("/chat", "/chat/index.html"):
        return "chat"
    if clean in ("/playground", "/playground/index.html"):
        return "playground"
    return None


def translate_text(text, translations):
    """Trim-aware translation lookup that preserves surrounding whitespace."""
    if not text or not text.strip():
        return text
    key = normalize_text(text)
    if not key or key.startswith(("http://", "https://")):
        return text
    if not re.search(r"[A-Za-z]", key):
        return text
    translated = translations.get(key)
    if not translated:
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    return leading + translated + trailing


def translations_for(locale, page=None):
    translations = dict(COMMON_COPY.get(locale, {}))
    if locale != "en" and page in COPY.get(locale, {}):
        translations[normalize_text(COPY["en"][page][2])] = COPY[locale][page][2]
    return translations


def _translate_nodes(html, translations):
    if not translations:
        return html
    parts = _PROTECTED_BLOCKS.split(html)
    for index in range(0, len(parts), 2):
        parts[index] = _TEXT_NODE.sub(
            lambda match: ">" + translate_text(match.group(1), translations) + "<",
            parts[index],
        )
    return "".join(parts)


def _set_lang(html, locale):
    """Set the lang attribute without dropping other <html> attributes."""
    match = re.search(r"<html\b[^>]*>", html, re.I)
    if not match:
        return html
    tag = match.group(0)
    if re.search(r"\slang\s*=", tag, re.I):
        tag = re.sub(
            r'(\slang\s*=\s*)(?:"[^"]*"|\'[^\']*\'|[^\s>]+)',
            lambda attr: attr.group(1) + f'"{locale}"',
            tag,
            count=1,
            flags=re.I,
        )
    else:
        tag = tag[:-1] + f' lang="{locale}">'
    return html[: match.start()] + tag + html[match.end() :]


def _seo_document(html, page, locale):
    """Replace upstream SEO tags with canonical, locale-specific metadata."""
    if page not in COPY.get(locale, {}):
        return html
    title, description, _ = COPY[locale][page]
    canonical = BASE_URL + page_path(page, locale)
    html = re.sub(
        r"<title\b[^>]*>.*?</title>",
        f"<title>{escape(title)}</title>",
        html,
        count=1,
        flags=re.I | re.S,
    )
    html = re.sub(
        r'<(?:meta\s+(?=[^>]*(?:name=["\']description["\']|property=["\']og:(?:url|title|description|locale)["\']|name=["\']twitter:(?:title|description)["\']))[^>]*|link\s+(?=[^>]*(?:rel=["\']canonical["\']|hreflang=))[^>]*)>',
        "",
        html,
        flags=re.I,
    )
    tags = [
        f'<meta name="description" content="{escape(description, quote=True)}">',
        f'<meta property="og:title" content="{escape(title, quote=True)}">',
        f'<meta property="og:description" content="{escape(description, quote=True)}">',
        f'<meta property="og:url" content="{canonical}">',
        f'<meta property="og:locale" content="{locale}">',
        f'<meta name="twitter:title" content="{escape(title, quote=True)}">',
        f'<meta name="twitter:description" content="{escape(description, quote=True)}">',
        f'<link rel="canonical" href="{canonical}">',
    ]
    tags.extend(
        f'<link rel="alternate" hreflang="{lang}" href="{BASE_URL + page_path(page, lang)}">'
        for lang in LOCALES
    )
    tags.append(
        f'<link rel="alternate" hreflang="x-default" href="{BASE_URL + page_path(page)}">'
    )
    return re.sub(
        r"</head>",
        "\n    " + "\n    ".join(tags) + "\n</head>",
        html,
        count=1,
        flags=re.I,
    )


def _locale_url(value, path, locale):
    """Resolve an internal URL to a root-absolute, locale-prefixed path.

    Locale pages do not inject a ``<base>`` element any more: a base element
    changes how fragment-only links (``#login``) and
    ``history.pushState("#settings")`` resolve, which used to move
    ``/de/chat/`` to ``/de/``. Resolving internal links here keeps them inside
    the locale prefix without that side effect. Fragment-only and query-only
    URLs are returned unchanged so they stay on the current page.
    """
    if not value or value.startswith(("#", "?")):
        return value
    if _SCHEME.match(value) or value.startswith(("{{", "{%", "${")):
        return value
    resolved = urlsplit(urljoin(path, value))
    if resolved.scheme or resolved.netloc:
        return value
    path_only = resolved.path or "/"
    segments = path_only.split("/")
    # Avoid /es/es/... when the source already contains a locale prefix.
    if len(segments) > 1 and segments[1].lower() in LOCALES:
        path_only = "/" + "/".join(segments[2:])
    if locale != "en":
        path_only = f"/{locale}{path_only if path_only.startswith('/') else '/' + path_only}"
    return urlunsplit(("", "", path_only, resolved.query, resolved.fragment))


def _rewrite_urls(html, path, locale):
    def rewrite_tag(match):
        tag = match.group(0)
        if re.match(r"<base\b", tag, re.I):
            return tag

        def rewrite_attr(attr):
            prefix, quote, value = attr.group(1), attr.group(2), attr.group(3)
            return prefix + quote + _locale_url(value, path, locale) + quote

        return _URL_ATTR.sub(rewrite_attr, tag)

    return _TAG.sub(rewrite_tag, html)


def _runtime_script(locale):
    return f"""
<script {_RUNTIME_MARKER}="1">
(function () {{
    "use strict";
    const locale = {json.dumps(locale)};
    const supported = ["en", "es", "fr", "de"];
    // The opened URL prefix always wins; document/navigator language is the fallback.
    const match = location.pathname.match(/^\\/([a-z]{2})(?:\\/|$)/i);
    const opened = match && supported.includes(match[1].toLowerCase())
        ? match[1].toLowerCase()
        : String(document.documentElement.lang || navigator.language || locale || "en")
            .split(/[-_]/)[0].toLowerCase();
    const base = supported.includes(opened) ? opened : "en";
    const apply = () => {{
        if (!window.framework) return false;
        if (framework.setSelectedLanguage) framework.setSelectedLanguage(base);
        const cached = framework.globalTranslations && framework.globalTranslations[base];
        if (cached && cached.translations) {{
            framework.translations = Object.assign({{}}, framework.translations, cached.translations);
            if (framework.translateElements) framework.translateElements();
        }}
        if (base !== "en" && typeof framework.loadGlobalTranslations === "function") {{
            framework.loadGlobalTranslations(base).then((entry) => {{
                if (entry && entry.translations) {{
                    framework.translations = Object.assign(
                        {{}}, framework.translations, entry.translations
                    );
                }}
                if (framework.translateElements) framework.translateElements();
                if (typeof framework.translateAll === "function") framework.translateAll();
            }}).catch(() => {{}});
        }} else if (typeof framework.translateAll === "function") {{
            framework.translateAll();
        }}
        return true;
    }};
    if (!apply()) {{
        document.addEventListener("DOMContentLoaded", apply, {{ once: true }});
        window.addEventListener("load", apply, {{ once: true }});
    }}
}})();
</script>
"""


def localize_document(html, locale, page=None, path="/"):
    """Localize a proxied HTML document: metadata, links, and runtime i18n."""
    if locale not in LOCALES:
        return html
    html = _set_lang(html, locale)
    html = _seo_document(html, page, locale)
    html = _translate_nodes(html, translations_for(locale, page))
    html = _rewrite_urls(html, path or "/", locale)
    # Links are already locale-prefixed; drop any upstream <base> so relative
    # fragment links keep resolving against the current page instead of the
    # locale root (e.g. #login on /de/chat/ must not become /de/#login).
    html = re.sub(r"<base\b[^>]*>", "", html, flags=re.I)
    if _RUNTIME_MARKER not in html:
        script = _runtime_script(locale)
        html = re.sub(r"</body>", script + "</body>", html, count=1, flags=re.I)
        if _RUNTIME_MARKER not in html:
            html += script
    return html


def seo_document(html, page, locale="en"):
    """Metadata-only decoration for unprefixed pages (no base/URL rewrite)."""
    if locale not in LOCALES:
        return html
    html = _set_lang(html, locale)
    html = _seo_document(html, page, locale)
    return _translate_nodes(html, translations_for(locale, page))


class LocaleMiddleware:
    """Strip a leading /<locale> prefix before Flask routing.

    This lets every existing page, asset, and API route work under the locale
    prefix without duplicate route definitions.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        match = _LOCALE_PREFIX.match(path)
        if match and match.group(1).lower() in LOCALES:
            locale = match.group(1).lower()
            environ["g4f.locale"] = locale
            environ["g4f.locale_path"] = path[match.end() :] or "/"
            environ["PATH_INFO"] = environ["g4f.locale_path"]
        return self.wsgi_app(environ, start_response)


def _localize_after_request(response):
    locale = request.environ.get("g4f.locale")
    if not locale or request.method not in ("GET", "HEAD"):
        return response
    if response.status_code != 200 or response.mimetype != "text/html":
        return response
    path = request.environ.get("g4f.locale_path", "/")
    page = page_for_path(path)
    was_passthrough = response.direct_passthrough
    response.direct_passthrough = False
    try:
        html = response.get_data(as_text=True)
    except (RuntimeError, UnicodeDecodeError):
        return response
    finally:
        if was_passthrough:
            close = getattr(response.response, "close", None)
            if close:
                close()
    response.set_data(localize_document(html, locale, page, path))
    response.headers.pop("ETag", None)
    response.headers.pop("Last-Modified", None)
    response.headers["Content-Language"] = locale
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


def install_locale_support(app):
    """Install the locale prefix middleware and HTML decorator once."""
    if getattr(app, "_g4f_locale_support", False):
        return
    app.wsgi_app = LocaleMiddleware(app.wsgi_app)
    app.after_request(_localize_after_request)
    app._g4f_locale_support = True
