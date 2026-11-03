# Browsing APIs, live feeds, triggers and notifications

g4f ships a small browsing layer (`g4f.browsing`) that collects free and paid
web APIs behind one catalog. It is exposed through the HTTP server, the MCP
tool set, and the chat playground.

No credential is bundled. APIs that need a key are reported as
`available: false` until the matching environment variable is set.

## API catalog

```http
GET /v1/browsing/apis
GET /v1/browsing/apis?capability=search&include_unavailable=false
```

| Id | Pricing | Key | Capabilities |
| --- | --- | --- | --- |
| `rss` | free | – | feed, fetch |
| `wikipedia` | free | – | search, fetch |
| `hackernews` | free | – | search, feed |
| `duckduckgo` | free | – | search |
| `searxng` | free | `G4F_SEARXNG_URL` | search |
| `reddit` | free | – | search, feed |
| `github` | free-tier | optional `GITHUB_TOKEN` | search |
| `arxiv` | free | – | search, feed |
| `jina-reader` | free-tier | optional `JINA_API_KEY` | fetch |
| `brave` | free-tier | `BRAVE_API_KEY` | search |
| `tavily` | free-tier | keyless, optional `TAVILY_API_KEY` | search, fetch |
| `serper` | free-tier | `SERPER_API_KEY` | search |
| `exa` | free-tier | keyless, optional `EXA_API_KEY` | search |
| `parallel` | free | keyless, optional `PARALLEL_API_KEY` | search |
| `firecrawl` | free-tier | keyless search, optional `FIRECRAWL_API_KEY` | search, fetch |
| `tinyfish` | free | keyless, optional `TINYFISH_API_KEY` | search |
| `keenable` | free-tier | keyless, optional `KEENABLE_API_KEY` | search, fetch |
| `newsapi` | free-tier | `NEWSAPI_KEY` | search, feed |

Search several sources at once (keyless defaults are used when `apis` is
omitted). The default set mirrors OpenCode's free search providers: Exa,
Parallel, Tavily, Firecrawl, TinyFish, Wikipedia, Hacker News and DuckDuckGo.
Override it with `G4F_SEARCH_APIS=exa,wikipedia`. The top-level endpoint accepts
GET or POST:

```bash
curl 'http://localhost:8080/v1/search?q=small+language+models&apis=exa,parallel&limit=5'

curl -X POST http://localhost:8080/v1/search \
  -H 'content-type: application/json' \
  -d '{"query":"small language models","apis":["wikipedia","exa"],"limit":5}'
```

Fast browsing fetches up to `concurrency` pages in parallel and extracts
readable text:

```bash
curl -X POST http://localhost:8080/v1/browsing/browse \
  -H 'content-type: application/json' \
  -d '{"urls":["https://example.com"],"max_words":400}'
```

Execute a catalog API operation explicitly:

```bash
curl -X POST http://localhost:8080/v1/browsing/execute \
  -H 'content-type: application/json' \
  -d '{"api":"rss","params":{"operation":"feed","url":"https://news.ycombinator.com/rss"}}'
```

## RSS/Atom as live conversations

```http
GET /v1/browsing/feeds?url=https://news.ycombinator.com/rss&limit=20
POST /v1/browsing/feeds/conversation
GET /v1/browsing/feeds/stream  # server-sent events, body: {url, interval}
```

The response is a normal g4f conversation:

```json
{
  "id": "…",
  "type": "feed",
  "title": "Hacker News",
  "feedUrl": "https://news.ycombinator.com/rss",
  "items": [
    {"id": "…", "role": "assistant", "content": "## Story title …", "ts": 0}
  ],
  "feedSeen": ["…"]
}
```

`feedSeen` lets a client append only new entries. In the chat playground,
**＋ Feed** adds a feed to the conversation list; it is refreshed every five
minutes and new entries raise a notification. The public site chat
(`g4f.dev/chat`) gets the same **Live Feed**, **Trigger from Prompt** and bell
controls from `chat/g4f-browsing.js`.

The parser uses the standard library only. `G4F_BROWSING_ALLOW_LOCAL=1`
permits loopback/private feed URLs (for local testing); private addresses are
rejected by default.

## Prompt-created triggers

```http
POST /v1/browsing/triggers/from-prompt   {"prompt":"Notify me about AI every 10 minutes","save":true}
GET  /v1/browsing/triggers
POST /v1/browsing/triggers/{id}/run
POST /v1/browsing/triggers/run-due
```

`from-prompt` parses a URL (RSS if the path looks like a feed), quoted
keywords, an interval and a summary/chat action. Set `"save": false` to review
the draft first. `run` fetches the source, emits one notification per unseen
matching item, and returns an `action_prompt` that can be sent to a chat model.
State is stored in `$G4F_CACHE_DIR/browsing/triggers.json`
(`G4F_BROWSING_STORE` overrides the path).

`run-due` is the hook for a future scheduler/worker. It only runs triggers
whose interval has elapsed.

## Notifications

```http
GET    /v1/browsing/notifications?unread_only=true
POST   /v1/browsing/notifications          # push a "message finished" event
POST   /v1/browsing/notifications/read     # {"id": "…"} or all
DELETE /v1/browsing/notifications
```

Triggers write here automatically. The chat client also uses the browser
Notification API: the 🔔 button in the chat toolbar asks for permission, and
finished assistant messages plus new feed entries raise a notification when
the tab is hidden. Unread chat messages are counted in the page title.

## MCP tools

Agents get six tools backed by the same layer:

- `browsing_apis` — list the catalog and availability
- `browsing_search` — search one or more APIs
- `fast_browse` — fetch several URLs concurrently
- `browsing_execute` — execute an allow-listed catalog operation
- `live_feed` — read an RSS/Atom feed (optionally as a conversation)
- `browsing_triggers` — create/list/run/delete prompt-created triggers

## Python

```python
import asyncio
from g4f.browsing import fetch_feed_conversation, search, parse_trigger_prompt

async def main():
    results = await search("gpt4free", apis=["wikipedia", "hackernews"])
    conversation = await fetch_feed_conversation("https://news.ycombinator.com/rss")
    draft = parse_trigger_prompt('Notify me about "AI" every 15 minutes')
    print(results["results"][0]["title"], conversation["title"], draft["interval"])

asyncio.run(main())
```
