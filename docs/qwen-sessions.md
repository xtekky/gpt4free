# Qwen Chat sessions

Qwen Chat access tokens last about 15 minutes. `QwenAuth` refreshes them when
less than three minutes remain, using the account's `refresh_token` cookie at
`https://auth.qwen.ai/api/v2/auths/refresh`. This is separate from Qwen Code OAuth.

## Automatic browser and HAR lookup

Automatic credential lookup first reuses the cached session or newest Qwen HAR.
When none exists, it captures the browser session through g4f's configured CDP
transport and saves `chat.qwen.ai-cdp.har` in the configured cookies directory.
Later messages and process restarts reuse that HAR instead of reopening the
browser. `use_browser=True` explicitly forces a new capture.

Browser capture waits for a visible chat composer, then reads the SDK fingerprint
and cookie records. It does not submit a seed message or subscribe to network
events. Account login is recovered from the `refresh_token` cookie, including
HttpOnly cookies; capture does not read access tokens from local storage. A HAR
without account credentials is treated as a guest session.

Automatic lookup defaults to `persist_auth=True`, saving browser captures and
refreshed credentials. Set `persist_auth=False` for memory-only capture.
Explicit credentials and `auth_session` remain memory-only unless persistence
is requested. The HAR contains session credentials and must be kept private.
Without a HAR or available CDP transport, the legacy automatic path can fall back
to generated guest cookies. An unready browser raises `MissingAuthError`.

HAR imports keep the newest credential-bearing request's token and cookies
together. Fingerprint headers can come from an earlier completion request.
Cookie expiry supports browser epoch seconds and HAR ISO timestamps. Loading a
HAR alone does not renew credentials or reset the server's guest quota.

## Explicit credentials

A refresh cookie alone is supported; an access token is obtained before the first
request. An access token without a refresh credential expires and requires login.
Use credentials from the same account:

```python
import os
from g4f.Provider.Qwen import Qwen, QwenAuth

auth = QwenAuth(
    token=os.environ.get("QWEN_ACCESS_TOKEN"),
    refresh_token=os.environ["QWEN_REFRESH_TOKEN"],
)

async def ask(prompt):
    async for chunk in Qwen.create_async_generator(
        model="qwen3.7-plus",
        messages=[{"role": "user", "content": prompt}],
        auth_session=auth,
    ):
        yield chunk
```

`cookies` accepts a dictionary or browser/HAR records with domain, path and
expiry. `QwenAuth(cookies=cookies)` automatically reads `refresh_token` from
those cookies. `auth.refresh_token` returns the current usable refresh cookie,
including replacements from `Set-Cookie`, or `None` when absent, expired or out
of scope. `headers` accepts the provider's supported browser fingerprint headers.

Reuse `auth_session` to retain rotating credentials across calls and separate
`asyncio.run()` invocations. Do not combine it with `token`, `api_key`,
`refresh_token`, `cookies`, or `use_browser`. Explicit credentials are never mixed
with an unrelated HAR account. `auth.get_cookies()` exports current records;
`auth.clear()` ends the session.

## Fingerprint and request behavior

`_read_browser_fingerprint` waits up to 30 seconds for initialized Baxia, then
reads `baxiaCommon.getUA(completion_url)` and `baxiaCommon.version`. It returns
`bx-ua` and optional `bx-v`, rejecting empty values and `default` placeholders.
It neither reads `bx-umidtoken` nor waits for the UID module. When no saved
`bx-umidtoken` header exists, request preparation uses the existing midtoken
service path. Neither token has a measured fixed lifetime.

The website SDK was inspected on 2026-10-04:

- [Qwen application 0.3.12](https://assets.alicdn.com/g/qwenweb/qwen-chat-fe/0.3.12/js/main.js)
- [Baxia 2.5.37](https://g.alicdn.com/sd/baxia/2.5.37/baxiaCommon.js)
- [AWSC](https://g.alicdn.com/AWSC/AWSC/awsc.js)
- [Fireye 1.234.37](https://g.alicdn.com/AWSC/fireyejs/1.234.37/fireyejs.js)

Baxia hooks Fetch/XHR and obtains `bx-ua` through the POST Fireye module. Its
obfuscated implementation has not been reproduced as a portable Python generator.

- Concurrent requests for one account share a single refresh, including requests
  across event loops. Cancelling a waiter does not cancel the shared refresh.
- A definite auth rejection is retried once after refresh, before any stream
  output. Refresh-cookie replacements and deletions are read before JSON errors.
  JSON endpoints reject malformed JSON and non-object responses.
- Known guest chat-limit errors retry once with a new request midtoken and chat;
  guest cookies and the shared image cache are preserved. A guest `quota_limit`
  response also retries once, retaining its conversation. These retries do not
  guarantee a renewed server quota. A second failure or error after output stops.
- Other rate limits and browser-validation challenges are raised directly.
  Baxia's HTTP-200 `ret` challenge response raises `CloudflareError` without
  including the session-bearing challenge URL.
- Expired or rejected account refresh credentials raise `MissingAuthError`; that
  account is not silently converted to a guest.
- Image uploads retain the existing shared cache keyed by image content hash.
  Bearer tokens and cookies are attached only to HTTPS Qwen chat/auth requests,
  not upload-storage or fingerprint-service hosts. Credential-bearing requests
  disable redirects by default.
- Stream parsing handles text, reasoning, image/tool results, citations, usage
  and finish events. A completed thinking or tool phase does not end the response.
- `get_models(auth_session=auth)` uses the available token without forcing a
  refresh. `get_quota` accepts generation's session options and uses chat creation.

For image editing, pass `chat_type="image_edit"` and `media`.
`thinking_mode="Auto"` enables planning and tool use; `qwen_image_model` can
select a supported image model. Earlier live text tests verified browser SDK
extraction and direct requests without seed messages. They do not establish that
image editing succeeds after a guest quota rejection. The final session and retry
contracts are covered by local tests; server acceptance still depends on the
session and available quota.
