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
import json
import logging
import time
from typing import Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

try:  # FastAPI / Starlette
    from starlette.websockets import WebSocketState
except ImportError:  # pragma: no cover
    WebSocketState = None  # type: ignore

debug = logging.getLogger("g4f.cdp_relay")

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

    async def list_targets(self) -> List[dict]:
        targets = []
        for client_id, agent in list(self.agents.items()):
            try:
                res = await agent.roundtrip({"type": "tabs"}, timeout=5)
                agent.tabs = res.get("tabs", [])
            except Exception as e:
                debug.warning(f"CDP relay: tabs query failed for {client_id}: {e}")
                agent.tabs = []
            for tab in agent.tabs:
                targets.append(
                    {
                        "id": f"{client_id}:{tab['tabId']}",
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
            return False
        agent = self.agents.get(client_id)
        if not agent:
            return False
        try:
            await agent.roundtrip({"type": "close_tab", "tabId": tab_id}, timeout=5)
        except Exception:
            pass
        async with self._lock:
            self.targets.pop(target_id, None)
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
