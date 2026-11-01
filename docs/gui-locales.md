# GUI locale and diagnostics routes

The GUI server (`python -m g4f --gui`) serves the English homepage and
providers catalog at `/` and `/providers/`. Spanish, French, and German
versions are available at `/es/`, `/fr/`, `/de/` and their respective
`/providers/` paths. The `.html` forms (for example `/fr/home.html` and
`/fr/providers.html`) return the same HTML. Assets and the chat API retain
their existing unprefixed URLs.

The home HTML is still fetched from g4f.dev (or the local dist when present),
then its title, description, primary heading, selected visible text, and
navigation are localized. The providers catalog is generated locally. This is
not a full translation of dynamic data, scripts, or every section of upstream
HTML. Add new pages to `g4f/gui/server/locale.py` only when translations of
their visible content are available. Canonical and alternate links are emitted
in HTML; `/meta` and `/<locale>/meta` expose the same route map as JSON, and
`/sitemap.xml` lists the supported landing pages. These SEO URLs use the
public `https://g4f.space` origin.

`GET /headers` returns request headers as JSON with common authentication and
secret header names redacted. `GET /country-ip` returns `{"ip": ..., "country": ...}`.
IP is the web server's `remote_addr`, not a client-supplied forwarded header;
country is the validated two-letter `CF-IPCountry` header, or `null` when
missing. Only trust country data if the edge proxy removes client-supplied
copies and supplies its own header. Both diagnostic responses use `no-store`.
