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
- Every `href`, `src`, `action`, and `poster` that points inside the site is
  rewritten to a root-absolute URL with the locale prefix, resolved against the
  original page path. This keeps links and assets inside the locale prefix,
  including already-relative URLs such as `./manifest.webmanifest` or
  `js/store.js`. No `<base>` element is injected: fragment-only links
  (`#login`) and `history.pushState("#settings")` must keep resolving against
  the current page, otherwise `/de/chat/` gets sent to `/de/`.
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

## Static GitHub Pages mirror

The public site at <https://g4f.dev> is statically hosted on GitHub Pages, so it
cannot run the Flask middleware. `g4f.dev/scripts/build-locales.mjs` generates
prefixed copies of every HTML page for each language in the community
translation catalog (38 at the time of writing: `ar`, `bg`, `cs`, `da`, `de`,
`el`, `es`, `et`, `fa`, `fi`, `fr`, `he`, `hi`, `hr`, `hu`, `id`, `it`, `ja`,
`kk`, `ko`, `nb`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sq`, `sr`, `sv`,
`th`, `tr`, `uk`, `ur`, `uz`, `vi`, `zh`). This is broader than the GUI, which
serves only `en`, `es`, `fr`, and `de`. Examples are `/fr/chat/`,
`/ja/chat/`, and `/pt/docs/`.

Each copy gets the locale `lang` attribute (`dir="rtl"` for Arabic, Persian,
Hebrew, and Urdu), root-absolute locale-prefixed links (relative assets and
manifests are rewritten at build time instead of relying on a `<base>`),
canonical and `hreflang` alternates for every generated language, and the same
runtime translation bootstrap used by the GUI. Before writing each page, the
generator fetches the community translation package for that language and
translates static text nodes plus visible attributes (`alt`, `title`,
`placeholder`, `aria-label`, SEO `content`, and button values) at build time.
That includes the document `<title>` and the SEO metadata (`description`,
`keywords`, `og:title`, `og:description`, `twitter:title`, and
`twitter:description`). The build also writes every English metadata string to
`dist/js/snippets/meta.json`, so the community catalog can translate metadata
the same way it translates UI copy. `og:url` and `og:locale` are rewritten to
the current locale page instead of the English source URL. The runtime
bootstrap then covers dynamic copy for all 38 API languages, not just the four
languages the Flask GUI serves. Unprefixed URLs stay canonical for English, and
`/en/...` is not generated; selecting English from a locale page returns to the
unprefixed path.

The build reads `https://beta.g4f.dev/challenge/translations?lang=<lang>` by
default. Set `G4F_TRANSLATIONS_URL`, `--translations-url`, or
`--translations-file` to use another source, or pass `--no-translations` for an
offline build. A language whose package cannot be fetched is still generated
and keeps the runtime bootstrap, so a temporary API outage never breaks the
deployment. The language selection page lists every API language and links to
the matching generated locale route.

The GitHub Pages workflows run the generator before uploading the artifact, and
the mirror sync commits the generated routes (they are gitignored in the source
repository) so the mirror's own deploy publishes them.

`GET /headers` returns request headers as JSON with common authentication and
secret header names redacted. `GET /country-ip` returns
`{"ip": ..., "country": ...}`. IP is the web server's `remote_addr`, not a
client-supplied forwarded header; country is the validated two-letter
`CF-IPCountry` header, or `null` when missing. Only trust country data if the
edge proxy removes client-supplied copies and supplies its own header. Both
diagnostic responses use `no-store`.
