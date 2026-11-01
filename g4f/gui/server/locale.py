"""Server-rendered SEO metadata for the public GUI's translated landing pages."""

import re
from html import escape

from flask import make_response


BASE_URL = "https://g4f.space"
LOCALES = ("en", "es", "fr", "de")
PAGES = ("home", "providers")

# Only publish alternate URLs for pages for which we have server-side copy.
COPY = {
    "en": {
        "home": ("G4F - Framework & Tools Hub", "Explore free AI tools and models with G4F.", "G4F AI Tools & Dashboard"),
        "providers": ("AI Providers - GPT4Free", "Browse AI providers and free models supported by G4F.", "Available Providers"),
    },
    "es": {
        "home": ("G4F - Herramientas de IA", "Explora herramientas y modelos de IA gratis con G4F.", "Herramientas de IA de G4F"),
        "providers": ("Proveedores de IA - GPT4Free", "Explora proveedores de IA y modelos gratis de G4F.", "Proveedores disponibles"),
    },
    "fr": {
        "home": ("G4F - Outils d'IA", "Explorez les outils et modèles d'IA gratuits avec G4F.", "Outils d'IA G4F"),
        "providers": ("Fournisseurs d'IA - GPT4Free", "Découvrez les fournisseurs d'IA et modèles gratuits de G4F.", "Fournisseurs disponibles"),
    },
    "de": {
        "home": ("G4F - KI-Werkzeuge", "Entdecke kostenlose KI-Werkzeuge und Modelle mit G4F.", "G4F KI-Werkzeuge"),
        "providers": ("KI-Anbieter - GPT4Free", "Entdecke KI-Anbieter und kostenlose Modelle von G4F.", "Verfügbare Anbieter"),
    },
}

BODY_COPY = {
    "es": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Accede a herramientas de IA, búsquedas web, registros en tiempo real, estado de modelos y servicios de varios proveedores.",
        "Browse the list of AI providers supported by G4F": "Explora los proveedores de IA compatibles con G4F",
        "Loading providers...": "Cargando proveedores...",
        "No specific models": "Sin modelos específicos",
        "Details": "Detalles", "Website": "Sitio web", "Models:": "Modelos:",
        "Auth:": "Acceso:", "Required": "Obligatorio", "Status:": "Estado:", "Working": "Disponible",
    },
    "fr": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Accédez aux outils d'IA, à la recherche web, aux journaux en temps réel, à l'état des modèles et aux services de plusieurs fournisseurs.",
        "Browse the list of AI providers supported by G4F": "Découvrez les fournisseurs d'IA compatibles avec G4F",
        "Loading providers...": "Chargement des fournisseurs...",
        "No specific models": "Aucun modèle spécifique",
        "Details": "Détails", "Website": "Site web", "Models:": "Modèles :",
        "Auth:": "Accès :", "Required": "Obligatoire", "Status:": "État :", "Working": "Disponible",
    },
    "de": {
        "Access high-performance AI tools, web search, real-time backend logs, model status, and multi-provider endpoints.": "Nutze KI-Werkzeuge, Websuche, Echtzeitprotokolle, Modellstatus und Dienste mehrerer Anbieter.",
        "Browse the list of AI providers supported by G4F": "Entdecke die von G4F unterstützten KI-Anbieter",
        "Loading providers...": "Anbieter werden geladen...",
        "No specific models": "Keine bestimmten Modelle",
        "Details": "Details", "Website": "Webseite", "Models:": "Modelle:",
        "Auth:": "Zugang:", "Required": "Erforderlich", "Status:": "Status:", "Working": "Verfügbar",
    },
}


def page_path(page, locale="en"):
    path = "/" if page == "home" else f"/{page}/"
    return f"/{locale}{path}" if locale != "en" else path


def seo_html(html, page, locale):
    """Decorate a proxied HTML document without changing scripts or asset URLs."""
    title, description, heading = COPY[locale][page]
    canonical = BASE_URL + page_path(page, locale)
    html = re.sub(r'<html\b[^>]*>', f'<html lang="{locale}">', html, count=1, flags=re.I)
    html = re.sub(r"<title\b[^>]*>.*?</title>", f"<title>{escape(title)}</title>", html, count=1, flags=re.I | re.S)
    # Remove upstream SEO tags so the document has one unambiguous canonical set.
    html = re.sub(
        r'<(?:meta\s+(?=[^>]*(?:name="description"|property="og:(?:url|title|description|locale)"|name="twitter:(?:title|description)"))[^>]*|link\s+(?=[^>]*(?:rel="canonical"|hreflang=))[^>]*)>',
        "", html, flags=re.I,
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
    tags.append(f'<link rel="alternate" hreflang="x-default" href="{BASE_URL + page_path(page)}">')
    html = re.sub(r"</head>", "\n    " + "\n    ".join(tags) + "\n</head>", html, count=1, flags=re.I)
    # Translate the static hero and navigation, not arbitrary JS or model names.
    english_heading = COPY["en"][page][2]
    html = re.sub(r"(<h1\b[^>]*>)(?:" + re.escape(english_heading) + "|" + re.escape(escape(english_heading)) + r")(</h1>)",
                  lambda m: m.group(1) + escape(heading) + m.group(2), html, count=1)
    if locale != "en":
        for original, translated in BODY_COPY[locale].items():
            html = html.replace(f">{original}<", f">{escape(translated)}<")
        if page == "home":
            home_text = {
                "es": ("Herramientas y módulos", "Navegador web de IA", "Registros del servidor", "Uso y estadísticas", "Estado del sistema y proveedores", "Proveedores"),
                "fr": ("Outils et modules", "Navigateur web d'IA", "Journaux du serveur", "Utilisation et statistiques", "État du système et des fournisseurs", "Fournisseurs"),
                "de": ("Werkzeuge und Module", "KI-Webbrowser", "Serverprotokolle", "Nutzung und Statistiken", "System- und Anbieterstatus", "Anbieter"),
            }[locale]
            for original, translated in zip(("Backend Tools & Modules", "AI Web Browser", "Backend Logs", "Usage & Statistics", "System & Provider Status", "Providers Catalog"), home_text):
                html = html.replace(f">{original}<", f">{escape(translated)}<")
        for name, translated in zip(("Home", "Providers", "Stats"),
                                    ("Inicio", "Proveedores", "Estadísticas") if locale == "es" else
                                    ("Accueil", "Fournisseurs", "Statistiques") if locale == "fr" else
                                    ("Startseite", "Anbieter", "Statistiken")):
            html = html.replace(f">{name}</a>", f">{translated}</a>")
        html = re.sub(r'href="/(providers)/?"',
                      lambda m: f'href="/{locale}/{m.group(1)}/"', html)
        html = html.replace('href="/"', f'href="/{locale}/"')
    return html


def localized_response(result, page, locale):
    response = make_response(result)
    if response.status_code == 200 and response.mimetype == "text/html":
        response.direct_passthrough = False
        response.set_data(seo_html(response.get_data(as_text=True), page, locale))
        response.headers.pop("ETag", None)
        response.headers.pop("Last-Modified", None)
        response.headers["Content-Language"] = locale
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response
