from __future__ import annotations

import asyncio
import html
import json
import os
import random
import re
import time
import uuid

from ...typing import AsyncResult, Messages
from ...requests import StreamSession
from ...requests.cdp import CDPSession
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ..openai.proofofwork import generate_proof_token
from ..openai.new import get_requirements_token, get_config
from ..openai.turnstile_vm import process_turnstile_new
from ...providers.response import JsonConversation, FinishReason
from ...cookies import get_cookies
from ...config import COOKIES_DIR, CUSTOM_COOKIES_DIR
from ... import debug


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
    working = True
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

    # Static app-version headers worth copying from a captured session.
    # Note: the worker-version headers ("cloudflare-workers-version-overrides",
    # "x-web-mobile-document-worker-version") are deliberately NOT copied:
    # they are validated against the signed
    # "x-web-mobile-conversation-document-affinity" token, which is minted
    # client-side per session and cannot be reused. Sending the version
    # headers without a matching affinity fails with a hard 403 "Invalid
    # conversation document affinity"; sending neither reaches the app gate.
    _AUTH_HEADERS_WHITELIST = (
        "x-web-mobile-document-renderer",
        "x-web-mobile-connectivity-effective-type",
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
                if attempt >= 3:
                    cls._auth_state = None
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
                for name in cls._AUTH_HEADERS_WHITELIST:
                    if name in entry_headers:
                        headers.setdefault(name, entry_headers[name])

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
        async with CDPSession(proxy=proxy) as session:
            try:
                await session.navigate(f"{cls.url}/?q=Hello")
            except Exception as e:
                # A load-event race must not discard an otherwise good session.
                debug.log(f"ChatGPTLightweight: CDP navigate: {e}")
            # Wait for a pending Cloudflare challenge to resolve.
            for _ in range(30):
                title = await session.evaluate_js("document.title") or ""
                if title and "Just a moment" not in title and "Attention Required" not in title:
                    break
                await asyncio.sleep(1)
            debug.log("ChatGPTLightweight: waiting for network idle")
            await session.wait_for_network_idle()
            debug.log("ChatGPTLightweight: network idle reached")
            cookies = await session.get_cookies([f"{cls.url}/"])
            if not cookies.get("cf_clearance"):
                # The clearance cookie is minted once the challenge resolves;
                # give it a moment to show up.
                for _ in range(10):
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
                if "chatgpt.com" not in request.get("url", ""):
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
            debug.log(
                f"ChatGPTLightweight: CDP harvest: {len(cookies)} cookies"
                f" (cf_clearance={'yes' if cookies.get('cf_clearance') else 'no'}),"
                f" {len(headers)} headers"
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
        headers = {
            **DEFAULT_HEADERS,
            "oai-did": oai_did,
            "oai-session-id": session_id,
            **auth_headers,
        }
        cookies = {**auth_cookies, "oai-did": oai_did}
        config = get_config(cls.user_agent)
        requirements_token = get_requirements_token(config)

        async with StreamSession(
            impersonate="chrome", timeout=timeout, proxy=proxy,
            cookies=cookies,
        ) as session:
            json_headers = {
                **headers,
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
                True, pow_config.get("seed", ""), pow_config.get("difficulty", ""), cls.user_agent
            )

            # 3b. Solve the invisible turnstile challenge. The "dx" payload is
            # a VM op list XOR-encoded with the requirements token ("p") that
            # was sent to sentinel/req; the browser runs it through the
            # sentinel script and submits the result.
            turnstile_config = requirements.get("turnstile") or {}
            turnstile_token = ""
            if turnstile_config.get("required"):
                turnstile_token = process_turnstile_new(
                    turnstile_config.get("dx", ""), requirements_token, cls.user_agent
                )
                debug.log(f"ChatGPTLightweight: turnstile token len {len(turnstile_token)}")

            # 4. Finalize: exchange prepare token + proof for a chat requirements token
            async with session.post(
                cls.requirements_finalize_url,
                json={"prepare_token": prepare_token, "proofofwork": proof_token},
                headers=json_headers,
            ) as response:
                await raise_for_status(response)
                chat_requirements_token = (await response.json())["token"]

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
                    "timezoneOffsetMinutes": "-120",
                },
                headers=form_headers,
            ) as response:
                await raise_for_status(response)
                prepare_data = await response.json()
            conduit_token = prepare_data.get("conduit_token")

            # 6. Send the message and read the DPU partial document
            operation_id = str(uuid.uuid4())
            form = {
                "conversationState": json.dumps({
                    "messages": [],
                    "parentMessageId": "client-created-root",
                    "safety": {"dismissedInterventionIds": []},
                    "userMessageCount": 1,
                }),
                "messageMetadata": "{}",
                "oai-session-id": session_id,
                "imageAttachments": "[]",
                "pendingImageUploads": "[]",
                "prompt": prompt,
                "chatRequirementsToken": chat_requirements_token,
                "proofToken": proof_token,
                "turnstileToken": turnstile_token,
                "telemetryToken": "",
                # Real browsers send measured turn timings; "[1,null]" is a
                # bot signal that feeds the "Chat verification required" gate.
                "timingToken": "[1,%.1f,70,0,21,2,0,88]" % random.uniform(40.0, 150.0),
                "clientContextualInfo": client_context,
                "imageSaveData": "off",
                "imageEffectiveType": "4g",
                "timezone": "Europe/Berlin",
                "timezoneOffsetMinutes": "-120",
                "conversationRetryOwner": retry_owner,
                "assistantMessageId": f"pending-{str(uuid.uuid4())}",
                "userMessageId": str(uuid.uuid4()),
            }
            update_headers = {
                **form_headers,
                "accept": "text/vnd.openai.web-mobile-partial+html",
                "x-conduit-token": conduit_token,
                "x-oai-turn-trace-id": operation_id,
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
