from __future__ import annotations

import asyncio
import html
import json
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from ..requests.cdp import CDPSession
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from .openai.proofofwork import generate_proof_token
from .openai.new import get_requirements_token
from .openai.har_file import RequestConfig
from .openai.turnstile_vm import process_turnstile_new
from ..providers.response import JsonConversation, FinishReason
from ..cookies import get_cookies
from ..config import COOKIES_DIR, CUSTOM_COOKIES_DIR
from .. import debug


class Conversation(JsonConversation):
    """Tracks the anonymous (lightweight) chat session state."""

    def __init__(self, model: str):
        self.model = model
        self.oai_did = str(uuid.uuid4())
        self.conversation_id: str = None
        self.message_id: str = None


_RE_CONVERSATION_ID = re.compile(r'data-conversation-id="([\w-]+)"')
_RE_MESSAGE_ID = re.compile(r'data-message-id="([\w-]+)"')
_RE_ASSISTANT_BLOCK = re.compile(
    r'(<p\b[^>]*data-assistant-stream-block=""[^>]*>)(.*?)</p>',
    re.DOTALL,
)
_RE_BLOCK_INDEX = re.compile(r'data-assistant-stream-block-index="(\d+)"')
_RE_FAILURE_STATUS = re.compile(r'data-failure-status="(\d+)"')
_RE_GATE_KIND = re.compile(r'data-gate-kind="(\w+)"')
_RE_XML_MARKER = re.compile(r"<\?[^>]*>")

DEFAULT_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.8",
    "referer": "https://chatgpt.com/",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
}


class ChatGPTLightweight(AsyncGeneratorProvider, ProviderModelMixin):
    """ChatGPT anonymous guest chat via the "web-mobile" (unauth-mweb) surface.

    Implements the lightweight_authenticated=0 flow observed in the guest
    web UI: requirements token -> prepare -> sentinel/req -> proof-of-work ->
    finalize -> conversation/prepare -> conversation/updates. The reply is
    returned as a DPU HTML partial document containing assistant stream blocks.

    The upstream gate intermittently answers with a 403 "Chat verification
    required" (rate limiting / IP reputation). Requests are therefore retried
    with a fresh session a few times before giving up.

    A "cf_clearance" cookie from a real browser session marks the visitor
    as trusted and greatly improves the pass rate. Fresh cookies and matching
    client-hint headers are harvested through a CDP browser session on first
    use (see `_harvest_via_cdp`), falling back to cached HAR captures
    (`_load_auth_state`) when no browser is available.
    """

    label = "ChatGPT (Lightweight)"
    url = "https://chatgpt.com"
    working = False
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "auto"
    # The guest surface routes to ChatGPT's default model; it is advertised
    # for the models it can serve so model/provider validation passes.
    models = [default_model, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-5"]

    user_agent = DEFAULT_HEADERS["user-agent"]
    max_retries = 4

    # Cached (cookies, headers) harvested from HAR files / browser profile.
    _auth_state = None
    _auth_state_loaded_at = 0.0
    _AUTH_STATE_TTL = 600.0
    # Document script srcs captured during the last CDP harvest. The
    # turnstile VM probes them with regexes (op 11); without the real
    # sentinel sdk.js URL the resolved token is short and the gate rejects
    # the chat with 403.
    _script_srcs: list = []

    # Static fingerprint headers worth copying from a captured session.
    # Session-bound headers are deliberately NOT copied:
    # - "x-web-mobile-conversation-document-affinity" is signed client-side
    #   per session (its payload binds the worker version AND the
    #   oai-session-id) and cannot be reused; the worker-version headers
    #   ("x-web-mobile-document-worker-version",
    #   "cloudflare-workers-version-overrides") are validated against it.
    #   Replaying any of them fails with a hard 403 ("Invalid conversation
    #   document affinity" / gate rejection) — sending neither reaches the
    #   app gate.
    # - "oai-session-id" must match the id sent in the form field, so a
    #   fresh one is generated per request instead of replaying a captured
    #   one.
    _AUTH_HEADERS_WHITELIST = (
        "x-web-mobile-conversation-renderer",
        "x-web-mobile-prepare-state",
        "sec-ch-ua",
        "sec-ch-ua-arch",
        "sec-ch-ua-bitness",
        "sec-ch-ua-full-version",
        "sec-ch-ua-full-version-list",
        "sec-ch-ua-mobile",
        "sec-ch-ua-model",
        "sec-ch-ua-platform",
        "sec-ch-ua-platform-version",
    )

    # Endpoints of the unauth-mweb surface
    requirements_prepare_url = "https://chatgpt.com/unauth-mweb/sentinel/chat-requirements/prepare"
    requirements_finalize_url = "https://chatgpt.com/unauth-mweb/sentinel/chat-requirements/finalize"
    sentinel_req_url = "https://chatgpt.com/backend-api/sentinel/req"
    conversation_prepare_url = "https://chatgpt.com/unauth-mweb/conversation/prepare"
    conversation_updates_url = "https://chatgpt.com/unauth-mweb/conversation/updates"

    @classmethod
    def get_client_context(cls) -> str:
        # This exact shape is validated server-side; simplified versions are
        # rejected with 400 "Invalid clientContextualInfo".
        return json.dumps({
            "app_name": "chatgpt.com",
            "has_web_push_capabilities": True,
            "is_dark_mode": False,
            "web_push_notification_permission": "default",
            "page_height": random.randint(600, 900),
            "page_width": random.randint(600, 1400),
            "pixel_ratio": 1,
            "screen_height": random.choice([800, 900, 1080]),
            "screen_width": random.choice([1280, 1600, 1920]),
            "time_since_loaded": random.randint(1, 10),
        })

    @classmethod
    def _get_proof_config(cls, user_agent: str, kind: str = "proof") -> list:
        """Build the sentinel config in the shape the guest surface sends.

        Captured from a real guest session: the mweb surface uses the full
        sentinel config (25 elements) extended by seven trailing zeros. The
        browser mints the requirements token and the proof-of-work token
        from two different script contexts, so the configs are NOT shared:
        the requirements token carries the
        "declarative-partial-updates-<hash>.js" script URL while the
        proof-of-work token carries "assets/octane-home-client-<hash>.js"
        with its own element values. Sharing one config between both tokens
        gets the issued chat requirements token flagged and the updates
        request gated with "Chat verification required".
        """
        now = datetime.now(timezone(timedelta(hours=2)))
        parse_time = now.strftime("%a %b %d %Y %H:%M:%S") + \
            " GMT+0200 (Mitteleuropäische Sommerzeit)"
        if kind == "requirements":
            # Context: the partial-updates page script that rebuilds the
            # requirements token (captured). Prefer the fresh src from the
            # harvest over the stale hardcoded fallback.
            script_url = next(
                (s for s in cls._script_srcs
                 if re.search(r"declarative-partial-updates-[0-9a-f]+\.js$", s)),
                "https://chatgpt.com/unauth-mweb/scripts/declarative-partial-updates-1222007e7648.js",
            )
            pow_counter = 20
            nav_entry = 207
            feature_probe = "cookieEnabled−true"
            early_intent = "__webMobileConversationAnnouncements"
            event_probe = "onclick"
            perf_now = random.randint(800, 3000)
        else:
            # Context: the octane home bundle that mints the proof token
            # (captured).
            script_url = next(
                (s for s in cls._script_srcs
                 if re.search(r"octane-home-client-[0-9a-zA-Z]+\.js$", s)),
                "https://chatgpt.com/unauth-mweb/assets/octane-home-client-DUBzS6ZD.js",
            )
            pow_counter = 0  # PoW counter, overwritten while solving
            nav_entry = 9
            feature_probe = "product−Gecko"
            early_intent = "__octaneEarlyHydrationIntents"
            event_probe = "onbeforematch"
            perf_now = random.uniform(400, 2500)
        return [
            2500,
            parse_time,
            4395630592,
            pow_counter,
            user_agent,
            script_url,
            RequestConfig.data_build,
            "de-DE",
            "en-US,en",
            nav_entry,
            feature_probe,
            early_intent,
            event_probe,
            perf_now,
            str(uuid.uuid4()),
            "model",
            8,
            time.time() * 1000,
            0, 0, 0, 0, 0, 0, 0,
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        conversation: Conversation = None,
        return_conversation: bool = True,
        **kwargs,
    ) -> AsyncResult:
        model = cls.get_model(model)
        prompt = format_prompt(messages)
        if conversation is None:
            conversation = Conversation(model)

        last_error = None
        for attempt in range(cls.max_retries):
            try:
                # A fresh visitor id per attempt: a flagged oai-did stays
                # flagged, so reusing it would poison every retry.
                if attempt > 0:
                    conversation.oai_did = str(uuid.uuid4())
                reply = await cls._fetch_reply(prompt=prompt, conversation=conversation, proxy=proxy, timeout=timeout)
                if reply is not None:
                    yield reply
                    if return_conversation:
                        yield conversation
                    yield FinishReason("stop")
                    return
            except Exception as e:
                last_error = e
                debug.log(f"ChatGPTLightweight: attempt {attempt + 1} failed: {e}")
            if attempt + 1 < cls.max_retries:
                # The anonymous-chat gate is rate-limit based — back off
                # before retrying with a fresh session.
                await asyncio.sleep(3 * (attempt + 1) + random.uniform(0, 2))
        if last_error is not None:
            raise last_error
        raise RuntimeError("ChatGPTLightweight: no reply received (verification gate)")

    @classmethod
    def _har_paths(cls) -> list:
        paths = []
        seen = set()
        for dir_path in (CUSTOM_COOKIES_DIR, str(COOKIES_DIR), os.path.expanduser("~/.g4f/workspace")):
            try:
                if not os.path.isdir(dir_path):
                    continue
                for name in os.listdir(dir_path):
                    if name.endswith(".har") and name not in seen:
                        seen.add(name)
                        paths.append(os.path.join(dir_path, name))
            except OSError:
                pass
        return paths

    @classmethod
    def _load_auth_state(cls) -> tuple:
        """Collect chatgpt.com cookies and app headers from real sessions.

        Sources, in order: the g4f cookie cache (populated from HAR files by
        the API server) and browser_cookie3, then a direct scan of *.har
        files in the cookie directories. The "cf_clearance" cookie is the
        important one: without it the anonymous-chat gate rejects most
        requests with 403 "Chat verification required".
        """
        now = time.time()
        if cls._auth_state is not None and now - cls._auth_state_loaded_at < cls._AUTH_STATE_TTL:
            return cls._auth_state

        cookies = {}
        headers = {}
        try:
            cookies.update(get_cookies("chatgpt.com", raise_requirements_error=False))
        except Exception:
            pass

        for path in cls._har_paths():
            try:
                with open(path, "rb") as file:
                    har = json.load(file)
            except Exception:
                continue
            for entry in har.get("log", {}).get("entries", []):
                request = entry.get("request", {})
                entry_headers = {
                    h["name"].lower(): h["value"] for h in request.get("headers", [])
                }
                host = entry_headers.get(":authority") or entry_headers.get("host", "")
                if "chatgpt.com" not in host:
                    continue
                for cookie in request.get("cookies", []):
                    cookies.setdefault(cookie["name"], cookie["value"])
                if "x-web-mobile-conversation-document-affinity" in entry_headers:
                    for name, value in entry_headers.items():
                        # Session-bound headers must not be replayed: the
                        # affinity token and the worker-version headers are
                        # validated against each other and expire per session
                        # (replaying them fails with 403 "Invalid conversation
                        # document affinity"). Only stable fingerprint headers
                        # from the whitelist are safe to copy.
                        if name in cls._AUTH_HEADERS_WHITELIST or name in (
                            "user-agent", "accept-language",
                        ):
                            headers.setdefault(name, value)
        if cookies:
            debug.log(
                f"ChatGPTLightweight: loaded {len(cookies)} chatgpt.com cookies"
                f" (cf_clearance={'yes' if cookies.get('cf_clearance') else 'no'})"
            )
        cls._auth_state = (cookies, headers)
        cls._auth_state_loaded_at = now
        return cls._auth_state

    @classmethod
    async def _harvest_via_cdp(cls, proxy: str = None) -> tuple:
        """Harvest fresh chatgpt.com cookies and headers via a CDP browser.

        Opens a real browser, waits for the Cloudflare gate to clear and
        captures the cookies (notably "cf_clearance") together with the
        client-hint headers the browser actually sends, so later requests
        match the fingerprint the cookies were issued for.
        """
        async with CDPSession(proxy=proxy, headless=False) as session:
            await session.navigate(f"{cls.url}/#q=Hello")
            debug.log("ChatGPTLightweight: navigated to chat page")
            await session.bypass_turnstile()
            debug.log("ChatGPTLightweight: bypassing turnstile")
            await session.insert_text_and_submit()
            debug.log("ChatGPTLightweight: waiting for network idle")
            await session.wait_for_network_idle()
            debug.log("ChatGPTLightweight: network idle reached")
            # The "?q=" parameter makes the guest UI auto-submit a chat. Watch
            # the browser's own turn: a rendered reply proves this IP and
            # fingerprint pass the gate, a failure marker means they are
            # flagged. A reload then gives Cloudflare a second chance to
            # serve a challenge and mint a "cf_clearance" cookie.
            outcome = None
            for _ in range(20):
                await asyncio.sleep(1)
                outcome = await session.evaluate_js(
                    "(() => { const el = document.querySelector("
                    "'[data-failure-status],[data-assistant-stream-block]');"
                    " return el ? (el.getAttribute('data-failure-status') || 'reply') : ''; })()"
                )
                if outcome:
                    break
            if outcome and outcome != "reply":
                debug.log(f"ChatGPTLightweight: browser chat rejected (status={outcome})")
                try:
                    await session.navigate(f"{cls.url}/?q=Hello")
                    await session.wait_for_network_idle()
                except Exception as e:
                    debug.log(f"ChatGPTLightweight: CDP reload: {e}")
            cookies = await session.get_cookies([f"{cls.url}/"])
            if not cookies.get("cf_clearance"):
                # The clearance cookie is minted once a challenge resolves;
                # give it a moment to show up.
                for _ in range(15):
                    await asyncio.sleep(1)
                    cookies = await session.get_cookies([f"{cls.url}/"])
                    if cookies.get("cf_clearance"):
                        break
            if not cookies:
                # Fall back to whatever the current page holds.
                cookies = await session.get_cookies()
            # Copy headers from requests the browser itself sent to chatgpt.com.
            headers = {}
            for params in session.network_requests:
                request = params.get("request", {})
                if "https://chatgpt.com/unauth-mweb/conversation/" not in request.get("url", ""):
                    continue
                sent = {
                    name.lower(): value
                    for name, value in request.get("headers", {}).items()
                }
                for name in cls._AUTH_HEADERS_WHITELIST:
                    if name in sent:
                        headers.setdefault(name, sent[name])
                if "user-agent" in sent:
                    headers["user-agent"] = sent["user-agent"]
                    # Keep proof-of-work / turnstile on the same UA the
                    # cookies were issued for.
                    cls.user_agent = sent["user-agent"]
                if "accept-language" in sent:
                    headers.setdefault("accept-language", sent["accept-language"])
            # Report how the browser's own auto-submitted chat fared: its
            # response status is the ground truth for whether this IP and
            # fingerprint pass the app gate at all.
            for params in session.network_responses:
                response = params.get("response", {})
                url = params.get("request", {}).get("url", "") or response.get("url", "")
                if "unauth-mweb/conversation/updates" in url:
                    debug.log(
                        f"ChatGPTLightweight: browser chat status {response.get('status')}"
                        f" ({response.get('mimeType', '').split(';')[0]})"
                    )
                    break
            # Dump the browser's successful updates request (headers + form)
            # so the replayed request can be aligned with it field by field.
            for params in session.network_requests:
                request = params.get("request", {})
                if "unauth-mweb/conversation/updates" not in request.get("url", ""):
                    continue
                sent_headers = {
                    name.lower(): value
                    for name, value in request.get("headers", {}).items()
                }
                debug.log(
                    "ChatGPTLightweight: browser request headers: "
                    + json.dumps(sorted(sent_headers))
                )
                break
            # Capture the document script srcs: the turnstile challenge
            # probes them with regexes (e.g. the sentinel sdk.js URL) and
            # folds the matches into the token.
            try:
                srcs_json = await session.evaluate_js(
                    "JSON.stringify([...document.scripts].map(s => s.src).filter(Boolean))"
                )
                srcs = json.loads(srcs_json) if srcs_json else []
                if srcs:
                    cls._script_srcs = srcs
            except Exception as e:
                debug.log(f"ChatGPTLightweight: script src capture failed: {e}")
            debug.log(
                f"ChatGPTLightweight: CDP harvest: {len(cookies)} cookies"
                f" (cf_clearance={'yes' if cookies.get('cf_clearance') else 'no'}),"
                f" {len(headers)} headers, {len(cls._script_srcs)} script srcs"
            )
            return cookies, headers

    @classmethod
    async def _fetch_reply(
        cls,
        prompt: str,
        conversation: Conversation,
        proxy: str = None,
        timeout: int = 120,
    ) -> str:
        if cls._auth_state is None:
            # Harvest fresh cookies and headers from a real browser before
            # falling back to cached HAR captures: a cf_clearance minted for
            # the current IP passes the gate far more reliably.
            try:
                cls._auth_state = await cls._harvest_via_cdp(proxy=proxy)
                cls._auth_state_loaded_at = time.time()
            except Exception as e:
                debug.log(f"ChatGPTLightweight: CDP harvest failed: {e}")
        auth_cookies, auth_headers = cls._load_auth_state()
        # Reuse the captured visitor id when available so the session stays
        # consistent with the cf_clearance cookie it was issued with.
        oai_did = conversation.oai_did = auth_cookies.get("oai-did") or str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        # The guest UI does not send an "oai-did" header on the mweb calls
        # (captured); the device id travels in the cookie and request bodies.
        headers = {
            **DEFAULT_HEADERS,
            "oai-session-id": session_id,
            **auth_headers,
        }
        cookies = {**auth_cookies, "oai-did": oai_did}
        # The browser mints the requirements token and the proof-of-work
        # token from two different script contexts, so each token gets its
        # own config (captured); sharing one gets the chat token flagged.
        config = cls._get_proof_config(cls.user_agent, kind="requirements")
        proof_config = cls._get_proof_config(cls.user_agent, kind="proof")
        requirements_token = get_requirements_token(config)

        async with StreamSession(
            impersonate="chrome", timeout=timeout, proxy=proxy,
            cookies=cookies,
        ) as session:
            json_headers = {
                **headers,
                # No "oai-did" header here: the guest UI does not send one on
                # the mweb calls (captured); the device id travels in the
                # cookie and request bodies only.
                "accept": "application/json",
                "content-type": "application/json",
            }
            # 1. Prepare: register the requirements attempt
            async with session.post(
                cls.requirements_prepare_url,
                json={"p": requirements_token},
                headers=json_headers,
            ) as response:
                await raise_for_status(response)
                prepare_token = (await response.json())["prepare_token"]

            # 2. Ask for the proof-of-work challenge
            async with session.post(
                cls.sentinel_req_url,
                json={"p": requirements_token, "id": oai_did, "flow": "conversation"},
                headers=json_headers,
            ) as response:
                await raise_for_status(response)
                requirements = await response.json()
            pow_config = requirements.get("proofofwork") or {}

            # 3. Solve the hashcash challenge
            proof_token = generate_proof_token(
                True, pow_config.get("seed", ""), pow_config.get("difficulty", ""),
                cls.user_agent, proof_token=proof_config
            )

            # 3b. Solve the invisible turnstile challenge. The "dx" payload is
            # a VM op list XOR-encoded with the requirements token ("p") that
            # was sent to sentinel/req; the browser runs it through the
            # sentinel script and submits the result.
            turnstile_config = requirements.get("turnstile") or {}
            turnstile_token = ""
            if turnstile_config.get("required"):
                turnstile_token = process_turnstile_new(
                    turnstile_config.get("dx", ""), requirements_token, cls.user_agent,
                    script_srcs=cls._script_srcs,
                )
                debug.log(f"ChatGPTLightweight: turnstile token len {len(turnstile_token)}")

            # 4. Finalize: exchange prepare token + proof for a chat requirements
            # token. The turnstile token MUST be included here: the server
            # binds it to the issued requirements token (captured finalize
            # body: {"prepare_token", "proofofwork", "turnstile"}). Without
            # it the updates request is rejected with 403 "Chat verification
            # required" even though the token itself is valid.
            finalize_body = {"prepare_token": prepare_token, "proofofwork": proof_token}
            if turnstile_token:
                finalize_body["turnstile"] = turnstile_token
            async with session.post(
                cls.requirements_finalize_url,
                json=finalize_body,
                headers=json_headers,
            ) as response:
                await raise_for_status(response)
                chat_requirements_token = (await response.json())["token"]

            # The session observer token is minted by the sentinel script in
            # the browser and bound to the requirements token (captured: a
            # ~640 char blob). Without a browser we cannot compute one; the
            # gate accepts an empty value (the browser sends it only when a
            # sentinel observer ran).
            session_observer_token = ""

            client_context = cls.get_client_context()
            retry_owner = json.dumps({"mode": "anonymous", "sessionEpoch": None})
            form_headers = dict(headers)

            # 5. Register the conversation document worker
            async with session.post(
                f"{cls.conversation_prepare_url}?lightweight_authenticated=0",
                data={
                    "conversationRetryOwner": retry_owner,
                    "conversationState": json.dumps({
                        "messages": [],
                        "parentMessageId": "client-created-root",
                        "userMessageCount": 0,
                    }),
                    "clientContextualInfo": client_context,
                    "timezone": "Europe/Berlin",
                    "timezoneOffsetMinutes": -120,
                },
                headers=form_headers,
            ) as response:
                await raise_for_status(response)
                prepare_data = await response.json()
            conduit_token = prepare_data.get("conduit_token")

            # 6. Send the message and read the DPU partial document
            operation_id = str(uuid.uuid4())
            # Captured sessions use two distinct uuids here: the operation id
            # in the URL and a separate turn trace id in the headers.
            trace_id = str(uuid.uuid4())
            form = {
                "conversationState": json.dumps({
                    "messages": [],
                    "parentMessageId": "client-created-root",
                    "safety": {"dismissedInterventionIds": []},
                    "userMessageCount": 0,
                }),
                "messageMetadata": "{}",
                "oai-session-id": session_id,
                "imageAttachments": "[]",
                "pendingImageUploads": "[]",
                "prompt": prompt,
                "chatRequirementsToken": chat_requirements_token,
                "proofToken": proof_token,
                "turnstileToken": turnstile_token,
                # Captured: the browser sends an EMPTY telemetry token.
                "telemetryToken": "",
                # The real guest UI sends a constant "[1,null]" here (captured),
                # so a fabricated timing array stands out more than it helps.
                "timingToken": "[1,null]",
                # Captured: set to "1" when the chat was started from the
                # instant-query URL parameter.
                "__web_mobile_instant_query": "1",
                "clientContextualInfo": client_context,
                "imageSaveData": "off",
                "imageEffectiveType": "4g",
                "timezone": "Europe/Berlin",
                "timezoneOffsetMinutes": -120,
                # Captured: the prepare token from step 1 is echoed in the
                # form alongside the finalized requirements token.
                "chatRequirementsPrepareToken": prepare_token,
                "sessionObserverToken": session_observer_token,
                "conversationRetryOwner": retry_owner,
                "assistantMessageId": f"pending-{str(uuid.uuid4())}",
                "userMessageId": str(uuid.uuid4()),
            }
            update_headers = {
                **form_headers,
                "accept": "text/vnd.openai.web-mobile-partial+html",
                "x-conduit-token": conduit_token,
                "x-oai-turn-trace-id": trace_id,
                "x-web-mobile-conversation-renderer": "octane",
                "x-web-mobile-conversation-stream-protocol": "1",
                "x-web-mobile-prepare-state": "success",
            }
            async with session.post(
                f"{cls.conversation_updates_url}?lightweight_authenticated=0&operationId={operation_id}",
                data=form,
                headers=update_headers,
            ) as response:
                await raise_for_status(response)
                text = await response.text()

        return cls._parse_dpu(text, conversation)

    @classmethod
    def _parse_dpu(cls, text: str, conversation: Conversation) -> str:
        """Extract the assistant reply from the DPU partial document.

        Returns None when the upstream verification gate rejected the chat
        (403 "Chat verification required") so the caller can retry.
        """
        failure = _RE_FAILURE_STATUS.search(text)
        if failure is not None:
            gate = _RE_GATE_KIND.search(text)
            debug.log(
                f"ChatGPTLightweight: gate rejected chat "
                f"(status={failure.group(1)}, kind={gate.group(1) if gate else 'unknown'})"
            )
            return None
        conversation_id = _RE_CONVERSATION_ID.search(text)
        if conversation_id:
            conversation.conversation_id = conversation_id.group(1)
        message_id = _RE_MESSAGE_ID.search(text)
        if message_id:
            conversation.message_id = message_id.group(1)
        # Later blocks with the same index are streaming updates that
        # supersede earlier (partial) ones — keep only the last per index.
        blocks = {}
        for tag, block in _RE_ASSISTANT_BLOCK.findall(text):
            index_match = _RE_BLOCK_INDEX.search(tag)
            index = int(index_match.group(1)) if index_match else 0
            blocks[index] = _RE_XML_MARKER.sub("", block)
        if not blocks:
            debug.log(f"ChatGPTLightweight: no assistant block in response: {text[:300]!r}")
            return None
        return html.unescape("".join(blocks[index] for index in sorted(blocks)))

if __name__ == "__main__":
    debug.logging = True
    import asyncio

    async def main():
        # Exercise the full retry path (fresh sessions + re-harvests), not
        # just a single _fetch_reply attempt.
        async for chunk in ChatGPTLightweight.create_async_generator(
            "auto", [{"role": "user", "content": "Guten Tag"}]
        ):
            if isinstance(chunk, (FinishReason, JsonConversation)):
                continue
            print(chunk)

    asyncio.run(main())