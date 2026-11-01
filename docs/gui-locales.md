# GUI locale and diagnostics routes

The GUI serves the same pages under a locale prefix, for example `/es/`,
`/fr/chat/`, `/de/playground/`, and `/es/index.html`. Supported locales are
`en`, `es`, `fr`, and `de`. The prefix is stripped by WSGI middleware before
Flask routing, so every existing page, static asset, app, and API route keeps
working under the prefix without duplicate route definitions. Unprefixed URLs
keep their previous behaviour.

The homepage (`/`, `/home.html`, `/index.html`), providers catalog, chat, and
playground receive locale-aware titles, descriptions, canonical links, and
`hreflang` alternates. `/meta` (and `/es/meta`, `/fr/meta`, ...) returns the
route map as JSON, and `/sitemap.xml` lists the localized landing pages.

Locale HTML is post-processed after routing:

- `<html lang>` is set to the requested locale.
- A `<base href="/<locale>/">` tag is inserted.
- Every `href`, `src`, `action`, and `poster` that points inside the site is
  rewritten to a relative URL, resolved against the original page path. This
  keeps links and assets inside the locale prefix, including already-relative
  URLs such as `./manifest.webmanifest` or `js/store.js`.
- Static copy is translated server-side with the same whitespace rules as
  `framework.translate()`: text is collapsed and trimmed for lookup, and the
  original leading/trailing whitespace is preserved. Upstream SEO tags are
  replaced with one canonical set.
- A small runtime bootstrap script applies the language of an explicitly
  opened locale URL, stores it with `framework.setSelectedLanguage()`, loads
  the community package through `framework.loadGlobalTranslations()`, and
  applies it with `framework.translateElements()` / `framework.translateAll()`.
  This covers dynamic strings that cannot be translated at request time.
- Unprefixed pages never translate silently. `framework.detectLanguage()`
  checks `navigator.languages` first and falls back to the IP country from
  `/country-ip`; `framework.maybePromptTranslation()` then asks before loading
  the package for the detected language. The visitor can accept, decline for
  that language, or open the language selection page.
- Loading a package from the framed language selection page redirects the
  parent window to the matching locale URL (`/es/chat/`, `/fr/chat/`, ...) when
  the parent exposes locale routes. Other sites keep the postMessage + reload
  flow.

Server-side copy covers the homepage, providers catalog, and static labels.
Chat and playground content is mostly dynamic and relies on the runtime global
translations. Static translations live in `g4f/gui/server/locale.py`; add a page
to `PAGES` and `COPY` only when its visible copy can be translated.

`GET /headers` returns request headers as JSON with common authentication and
secret header names redacted. `GET /country-ip` returns
`{"ip": ..., "country": ...}`. IP is the web server's `remote_addr`, not a
client-supplied forwarded header; country is the validated two-letter
`CF-IPCountry` header, or `null` when missing. Only trust country data if the
edge proxy removes client-supplied copies and supplies its own header. Both
diagnostic responses use `no-store`.
