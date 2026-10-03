"""
g4f — CDP relay: expose browser-extension agents as a CDP endpoint.

The g4f API server can use browsers provided by the g4f browser extension
(projects/browser-extension). The extension connects OUT to this relay over
a WebSocket and executes CDP commands locally via chrome.debugger on
dedicated automation tabs. This module presents the classic CDP HTTP surface
(/json, /json/new, /json/close) plus per-target WebSocket pass-through, so
g4f's own CDPSession (g4f/requests/cdp.py) can connect to it unchanged.

Wire protocol between relay and extension agent (JSON):
    -> {type:"cdp", id, tabId, method, params}   execute CDP command
    <- {type:"cdp", id, result?, error?}         command result
    -> {type:"new_tab", url?}  <- {type:"tab", tabId, url, title}
    -> {type:"close_tab", tabId} <- {type:"closed", tabId}
    -> {type:"tabs"}           <- {type:"tabs", tabs:[{tabId,title,url}]}

Enable from the extension's options page ("Browser as CDP provider") or by
setting G4F_BROWSER_MODE=extension before starting the server.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import secrets
import time
from typing import Dict, List, Optional
from urllib.parse import quote, urlparse

from fastapi import Body, Depends, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse

try:  # FastAPI / Starlette
    from starlette.websockets import WebSocketState
except ImportError:  # pragma: no cover
    WebSocketState = None  # type: ignore

debug = logging.getLogger("g4f.cdp_relay")


# Forwards clicks/inputs/selects in the served HTML copy to the live target.
_COPY_SCRIPT = r"""
let busy = false, timer;
const send = async (body, reload) => {
  if (busy) return;
  busy = true;
  clearTimeout(timer);
  try {
    await fetch(location.origin + '/browser/' + encodeURIComponent(T) + '/action', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  } catch {}
  busy = false;
  if (reload) location.reload();
};
const idx = el => { const t = el.closest('[data-index]'); return t ? [t, +t.dataset.index] : []; };
const plain = el => el.matches('input[type=checkbox],input[type=radio],input[type=file]');
document.addEventListener('click', e => {
  const [el, i] = idx(e.target);
  if (el === undefined || el.isContentEditable || el.matches('select,textarea,input:not([type=checkbox]):not([type=radio]):not([type=submit]):not([type=button])')) return;
  e.preventDefault();
  send({type: 'click', index: i}, true);
}, true);
document.addEventListener('submit', e => e.preventDefault(), true);
document.addEventListener('change', e => {
  const [el, i] = idx(e.target);
  if (el && el.matches('select')) send({type: 'select', index: i, value: el.value}, true);
}, true);
document.addEventListener('input', e => {
  const [el, i] = idx(e.target);
  if (busy || !el || el.matches('select') || plain(el)) return;
  clearTimeout(timer);
  timer = setTimeout(() => send({type: 'type', index: i,
    value: el.isContentEditable ? el.textContent : el.value}, true), 600);
}, true);
document.addEventListener('keydown', e => {
  const [el, i] = idx(e.target);
  if (e.key !== 'Enter' || !el || !el.matches('input')) return;
  e.preventDefault();
  send({type: 'type', index: i, value: el.value, submit: true}, true);
}, true);
"""
AGENT_PATH = "/v1/cdp/agent"  # extension agents connect here


class AgentConnection:
    """One connected extension agent (one browser)."""

    def __init__(self, ws: WebSocket, client_id: str):
        self.ws = ws
        self.client_id = client_id
        self.pending: Dict[int, asyncio.Future] = {}
        self._id = 0
        self.tabs: List[dict] = []
        self.last_seen: float = time.time()

    async def send(self, obj: dict) -> None:
        await self.ws.send_text(json.dumps(obj))

    async def roundtrip(self, obj: dict, timeout: float = 30.0) -> dict:
        """Send a request expecting a correlated response."""
        self._id += 1
        msg_id = self._id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[msg_id] = fut
        try:
            await self.send({**obj, "id": msg_id})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.pending.pop(msg_id, None)

    def resolve(self, msg: dict) -> None:
        """Resolve a pending roundtrip by response id."""
        fut = self.pending.pop(msg.get("id"), None)
        if fut and not fut.done():
            if "error" in msg and msg["error"] is not None:
                fut.set_exception(RuntimeError(str(msg["error"])))
            else:
                fut.set_result(msg)


class CdpRelay:
    """
    Registry of connected agents + the CDP-over-HTTP facade.

    Presents the subset of the Chrome DevTools HTTP API that
    g4f.requests.cdp.CDPSession uses:
        GET  /json            -> list targets
        PUT  /json/new        -> create target (returns webSocketDebuggerUrl)
        GET  /json/close/{id} -> close target
    plus WebSocket pass-through at /v1/cdp/ws/{target_id}.
    """

    def __init__(self):
        self.agents: Dict[str, AgentConnection] = {}
        # target_id -> (client_id, tab_id)
        self.targets: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()
        # target_id -> {title,url,html?} for tabs seen open; closed ones move to `closed`.
        self.known: Dict[str, dict] = {}
        self.closed: Dict[str, dict] = {}
        # Zombie reaper: MV3 service workers die after ~30s idle, which can
        # leave sockets that no longer respond. Track liveness via pings.
        self._reaper_task: Optional[asyncio.Task] = None

    def _ensure_reaper(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reaper())

    async def _reaper(self) -> None:
        """Drop agents that stop answering pings (dead service workers)."""
        while self.agents:
            await asyncio.sleep(10)
            for agent in list(self.agents.values()):
                if time.time() - agent.last_seen > 45:
                    debug.warning(
                        f"CDP relay: reaping zombie agent '{agent.client_id}'"
                    )
                    try:
                        await agent.ws.close()
                    except Exception:
                        pass
                    await self.unregister(agent)

    # ---------------- agent management ----------------

    async def register(self, ws: WebSocket, client_id: str) -> AgentConnection:
        agent = AgentConnection(ws, client_id)
        async with self._lock:
            # One agent per extension id: newest wins.
            old = self.agents.get(client_id)
            self.agents[client_id] = agent
        self._ensure_reaper()
        if old and old is not agent:
            try:
                await old.ws.close()
            except Exception:
                pass
        # An agent is connected: route CDPSession traffic through the relay
        # instead of launching a local Chrome instance.
        self._set_extension_mode(True)
        debug.info(f"CDP relay: agent '{client_id}' connected")
        return agent

    async def unregister(self, agent: AgentConnection) -> None:
        async with self._lock:
            if self.agents.get(agent.client_id) is agent:
                del self.agents[agent.client_id]
            dead = [
                tid for tid, (cid, _) in self.targets.items()
                if cid == agent.client_id
            ]
            for tid in dead:
                del self.targets[tid]
            for tid in [t for t in self.known if t.rpartition(":")[0] == agent.client_id]:
                self._mark_closed(tid)
        if not self.agents:
            # Last agent gone: fall back to local browser handling.
            self._set_extension_mode(False)
        debug.info(f"CDP relay: agent '{agent.client_id}' disconnected")

    @staticmethod
    def _set_extension_mode(enabled: bool) -> None:
        """Enable/disable extension routing in g4f's CDP core."""
        try:
            from ..cookies import BrowserConfig

            current = getattr(BrowserConfig, "browser_mode", None)
            if enabled and current != "extension":
                BrowserConfig.browser_mode = "extension"
                debug.info("CDP relay: extension mode enabled (agent registered)")
            elif not enabled and current == "extension":
                BrowserConfig.browser_mode = None
                debug.info("CDP relay: extension mode disabled (no agents)")
        except Exception as e:
            debug.warning(f"CDP relay: failed to set browser_mode: {e}")

    def get_agent(self, client_id: Optional[str] = None) -> AgentConnection:
        if client_id and client_id in self.agents:
            return self.agents[client_id]
        if self.agents:
            return next(iter(self.agents.values()))
        raise RuntimeError(
            "No browser extension agent connected. "
            "Enable 'Browser as CDP provider' in the g4f extension."
        )

    # ---------------- CDP facade (used by routes below) ----------------

    def _mark_closed(self, target_id: str) -> None:
        info = self.known.pop(target_id, None)
        if info is None:
            return
        self.closed[target_id] = {**info, "closed_at": time.time()}
        while len(self.closed) > 50:
            del self.closed[next(iter(self.closed))]

    async def evaluate(self, target_id: str, expression: str, timeout: float = 30.0):
        """Run JS in a target tab via the agent and return the JSON value."""
        client_id, tab_id = self.split_target(target_id)
        agent = self.get_agent(client_id)
        res = await agent.roundtrip(
            {"type": "cdp", "tabId": tab_id, "method": "Runtime.evaluate",
             "params": {"expression": expression, "returnByValue": True,
                        "awaitPromise": True}},
            timeout=timeout,
        )
        out = res.get("result", {})
        if out.get("exceptionDetails"):
            raise RuntimeError(str(out["exceptionDetails"].get("text", "evaluation failed")))
        return out.get("result", {}).get("value")

    async def snapshot(self, target_id: str) -> Optional[str]:
        """Standalone HTML copy of an open tab (cached); cached copy for closed tabs."""
        from ..mcp.browser_dom import SNAPSHOT_JS

        try:
            html = await self.evaluate(target_id, SNAPSHOT_JS)
        except Exception as e:
            debug.warning(f"CDP relay: snapshot failed for {target_id}: {e}")
            html = None
        if html:
            info = self.known.get(target_id)
            if info is not None:
                info["html"] = html
            return html
        return (self.known.get(target_id) or self.closed.get(target_id) or {}).get("html")

    async def list_targets(self) -> List[dict]:
        targets = []
        seen = set()
        for client_id, agent in list(self.agents.items()):
            try:
                res = await agent.roundtrip({"type": "tabs"}, timeout=5)
                agent.tabs = res.get("tabs", [])
            except Exception as e:
                debug.warning(f"CDP relay: tabs query failed for {client_id}: {e}")
                agent.tabs = []
            else:
                # Only a successful query proves a tab is gone.
                prefix = f"{client_id}:"
                live = {f"{client_id}:{t['tabId']}" for t in agent.tabs}
                for tid in [k for k in self.known if k.startswith(prefix) and k not in live]:
                    self._mark_closed(tid)
            for tab in agent.tabs:
                tid = f"{client_id}:{tab['tabId']}"
                seen.add(tid)
                info = self.known.setdefault(tid, {})
                info.update(title=tab.get("title", ""), url=tab.get("url", ""))
                targets.append(
                    {
                        "id": tid,
                        "type": "page",
                        "title": tab.get("title", ""),
                        "url": tab.get("url", ""),
                        "attached": False,
                    }
                )
        return targets

    async def new_target(self, url: str = "about:blank") -> dict:
        agent = self.get_agent()
        res = await agent.roundtrip({"type": "new_tab", "url": url}, timeout=15)
        tab_id = res.get("tabId")
        target_id = f"{agent.client_id}:{tab_id}"
        async with self._lock:
            self.targets[target_id] = (agent.client_id, tab_id)
        return {
            "id": target_id,
            "type": "page",
            "title": res.get("title", ""),
            "url": res.get("url", url),
            "webSocketDebuggerUrl": f"ws://__relay__/{target_id}",
        }

    async def close_target(self, target_id: str) -> bool:
        client_id, tab_id = self.targets.get(target_id, (None, None))
        if client_id is None:
            try:
                client_id, tab_id = self.split_target(target_id)
            except RuntimeError:
                return False
        agent = self.agents.get(client_id)
        if not agent:
            return False
        if target_id in self.known:
            await self.snapshot(target_id)
        try:
            await agent.roundtrip({"type": "close_tab", "tabId": tab_id}, timeout=5)
        except Exception:
            pass
        async with self._lock:
            self.targets.pop(target_id, None)
            self._mark_closed(target_id)
        return True

    def split_target(self, target_id: str):
        """target_id 'client:tab' -> (client_id, tab_id:int)."""
        client_id, _, tab = target_id.rpartition(":")
        if not client_id or not tab.isdigit():
            raise RuntimeError(f"Invalid target id: {target_id}")
        return client_id, int(tab)


# Single relay instance for the app.
relay = CdpRelay()


async def agent_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint: extension agents connect here."""
    client_id = ws.query_params.get("client", "extension")
    await ws.accept()
    agent = await relay.register(ws, client_id)
    try:
        while True:
            raw = await ws.receive_text()
            agent.last_seen = time.time()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            mtype = msg.get("type")
            if mtype == "ping":
                # Agent keepalive — reply so it knows the link is healthy.
                agent.last_seen = time.time()
                try:
                    await agent.send({"type": "pong", "id": msg.get("id")})
                except Exception:
                    pass
                continue
            if mtype == "cdp" and "id" in msg:
                agent.resolve(msg)
            elif mtype in ("tab", "tabs", "closed", "pong", "attached", "detached"):
                # responses to new_tab / tabs / close_tab / ping roundtrips
                agent.resolve(msg)
            elif mtype == "error":
                agent.resolve(msg)
            # "attached"/"detached"/"pong" are informational for now
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover
        debug.warning(f"CDP relay: agent error: {e}")
    finally:
        await relay.unregister(agent)


async def cdp_ws_endpoint(ws: WebSocket, target_id: str) -> None:
    """
    WebSocket endpoint: CDPSession connects here per target.
    Passes CDP commands through to the extension agent.
    """
    await ws.accept()
    try:
        client_id, tab_id = relay.split_target(target_id)
        agent = relay.get_agent(client_id)
    except RuntimeError as e:
        await ws.close(code=4404, reason=str(e))
        return

    async def reader():
        """g4f -> extension"""
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    cmd = json.loads(raw)
                except ValueError:
                    continue
                req_id = cmd.get("id")
                try:
                    result = await agent.roundtrip(
                        {"type": "cdp", "tabId": tab_id,
                         "method": cmd.get("method"),
                         "params": cmd.get("params", {})},
                        timeout=60,
                    )
                    await ws.send_text(json.dumps(
                        {"id": req_id, "result": result.get("result", {})}
                    ))
                except Exception as e:
                    await ws.send_text(json.dumps(
                        {"id": req_id, "error": str(e)}
                    ))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    async def writer():
        """extension events -> g4f (event fan-out is not needed by CDPSession,
        which polls; keep the loop alive and ignore)."""
        try:
            while True:
                await asyncio.sleep(1)
        except Exception:
            pass

    try:
        await asyncio.gather(reader(), writer())
    except Exception:
        pass


def register_cdp_relay(app) -> None:
    """Attach relay routes to the FastAPI app."""

    def _require_debug() -> None:
        # Checked per request so a later enable_logging() takes effect.
        from .. import debug as g4f_debug

        if not g4f_debug.logging:
            raise HTTPException(status_code=404, detail="Not Found")

    @app.websocket(AGENT_PATH)
    async def _agent(ws: WebSocket) -> None:
        await agent_endpoint(ws)

    @app.get("/json")
    async def _json_list() -> List[dict]:
        return await relay.list_targets()

    @app.get("/json/list")
    async def _json_list2() -> List[dict]:
        return await relay.list_targets()

    @app.put("/json/new")
    async def _json_new(url: str = "about:blank") -> dict:
        return await relay.new_target(url)

    @app.get("/json/new")
    async def _json_new_get(url: str = "about:blank") -> dict:
        return await relay.new_target(url)

    @app.get("/json/close/{target_id}")
    async def _json_close(target_id: str) -> dict:
        ok = await relay.close_target(target_id)
        return {"ok": ok}

    @app.websocket("/v1/cdp/ws/{target_id}")
    async def _cdp(ws: WebSocket, target_id: str) -> None:
        await cdp_ws_endpoint(ws, target_id)

    @app.get("/browser", response_class=HTMLResponse, dependencies=[Depends(_require_debug)])
    async def _browser_list() -> HTMLResponse:
        await relay.list_targets()
        from ..requests.cdp import list_session_targets

        local = await list_session_targets()
        opened = {t["id"]: t for t in local["open"]}
        closed = {t["id"]: t for t in local["closed"]}
        # Relay entries win: extension-mode CDPSessions appear in both.
        opened.update(relay.known)
        closed.update(relay.closed)
        for tid in opened:
            closed.pop(tid, None)

        def rows(items: dict, label: str) -> str:
            out = []
            for tid, info in reversed(list(items.items())):
                link = f"/browser/{quote(tid, safe='')}/html"
                has_copy = label == "open" or info.get("html")
                cell = f'<a href="{link}">html copy</a>' if has_copy else "-"
                if label == "open":
                    cell += (
                        f' <form method=post action="/browser/{quote(tid, safe="")}/close" '
                        'style=display:inline><button>close</button></form>'
                    )
                out.append(
                    f"<tr><td>{label}</td><td>{html.escape(info.get('title') or '')}</td>"
                    f"<td>{html.escape(info.get('url') or '')}</td><td>{cell}</td></tr>"
                )
            return "".join(out)

        body = (
            "<!DOCTYPE html><meta charset=utf-8><title>Browser targets</title>"
            "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 12px;"
            "text-align:left}</style><h1>Browser targets</h1>"
            "<form method=post action=/browser/new><input name=url type=url required "
            "placeholder='https://...' size=60> <button>open new target</button></form>"
            "<table><tr><th>State</th><th>Title</th><th>URL</th><th>Copy</th></tr>"
            f"{rows(opened, 'open')}{rows(closed, 'closed')}</table>"
        )
        return HTMLResponse(body)

    @app.post("/browser/new", dependencies=[Depends(_require_debug)])
    async def _browser_new(url: str = Form(...)) -> RedirectResponse:
        if urlparse(url).scheme not in ("http", "https"):
            raise HTTPException(status_code=400, detail="Only http(s) URLs are supported")
        from ..image import is_safe_url

        if not await asyncio.get_running_loop().run_in_executor(None, is_safe_url, url):
            raise HTTPException(status_code=400, detail="Local and network URLs are not allowed")
        if relay.agents:
            target = await relay.new_target(url)
            relay.known.setdefault(target["id"], {}).update(title="", url=url)
        else:
            from ..requests.cdp import CDPSession

            session = CDPSession()
            await session.start()
            await session.navigate(url)
        return RedirectResponse("/browser", status_code=303)

    @app.post("/browser/{target_id}/close", dependencies=[Depends(_require_debug)])
    async def _browser_close(target_id: str) -> RedirectResponse:
        from ..requests.cdp import _open_sessions

        session = _open_sessions.get(target_id)
        if session is not None and target_id not in relay.known:
            await session.close()
        else:
            await relay.close_target(target_id)
        return RedirectResponse("/browser", status_code=303)

    @app.get("/browser/{target_id}/html", response_class=HTMLResponse, dependencies=[Depends(_require_debug)])
    async def _browser_html(target_id: str) -> HTMLResponse:
        if target_id in relay.known or target_id in relay.closed:
            page = await relay.snapshot(target_id)
        else:
            from ..requests.cdp import snapshot_session_target

            page = await snapshot_session_target(target_id)
        if not page:
            raise HTTPException(status_code=404, detail="No HTML copy available")
        nonce = secrets.token_urlsafe(16)
        script = (
            f'<script nonce="{nonce}">const T={json.dumps(target_id)};{_COPY_SCRIPT}</script>'
        )
        page = page.replace("</body>", script + "</body>", 1) if "</body>" in page else page + script
        return HTMLResponse(
            page,
            headers={"Content-Security-Policy": "default-src * data: blob: 'unsafe-inline'; "
                     f"script-src 'nonce-{nonce}'; object-src 'none'; connect-src 'self'; "
                     "form-action 'none'"},
        )

    @app.post("/browser/{target_id}/action", dependencies=[Depends(_require_debug)])
    async def _browser_action(target_id: str, payload: dict = Body(...)) -> dict:
        from ..mcp.browser_dom import click_js, select_js, type_js

        try:
            index = int(payload["index"])
            kind = payload.get("type")
            if kind == "click":
                js = click_js(index)
            elif kind == "type":
                js = type_js(index, str(payload.get("value", "")), True, bool(payload.get("submit")))
            elif kind == "select":
                js = select_js(index, str(payload.get("value", "")))
            else:
                raise HTTPException(status_code=400, detail="Unknown action type")
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid action")
        if target_id in relay.known:
            result = await relay.evaluate(target_id, js)
        else:
            from ..requests.cdp import _open_sessions

            session = _open_sessions.get(target_id)
            if session is None or not session.is_alive:
                raise HTTPException(status_code=409, detail="Target is not open")
            result = await session.evaluate_js(js)
        await asyncio.sleep(0.3 if kind == "type" and not payload.get("submit") else 1)  # let the page settle before the copy reloads
        return {"result": result}
