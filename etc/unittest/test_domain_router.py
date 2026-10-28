from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient

import g4f.api
from g4f.api.domain_router import (
    DEFAULT_SUFFIX,
    DomainRouterMiddleware,
    domain_router_enabled,
    domain_suffix,
    hosts_entries,
    register_domain_router,
    routes,
)
from g4f.api.local_gateway_proxy import websocket_url


class _RecordingApp:
    """Minimal ASGI app that records the scope it was called with."""

    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        if scope["type"] == "websocket":
            await send({"type": "websocket.accept"})
            await send({"type": "websocket.close"})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def _send(message):
    return None

def _ws_receive():
    """Receive callable that completes the WebSocket handshake."""
    messages = [
        {"type": "websocket.connect"},
        {"type": "websocket.disconnect", "code": 1000},
    ]

    async def receive():
        return messages.pop(0) if messages else {"type": "websocket.disconnect", "code": 1000}

    return receive


def _http_scope(host: str, path: str = "/v1/models"):
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", host.encode())],
    }


def _free_port() -> int:
    """Return a localhost port that is currently free."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def _closed_port() -> int:
    """Return a localhost port that nothing is listening on."""
    return _free_port()


class TestDomainRouterMatching(unittest.TestCase):
    def setUp(self):
        self.middleware = DomainRouterMiddleware(_RecordingApp())

    def test_match_by_name(self):
        route = self.middleware._match("omniroute.g4f.local")
        self.assertIsNotNone(route)
        self.assertEqual(route.name, "omniroute")
        self.assertEqual(route.port, 20128)

    def test_match_by_port(self):
        route = self.middleware._match("20128.g4f.local")
        self.assertIsNotNone(route)
        self.assertEqual(route.name, "omniroute")

    def test_match_prefix_route(self):
        route = self.middleware._match("remote-desktop.g4f.local")
        self.assertIsNotNone(route)
        self.assertEqual(route.prefix, "/remote_desktop")

    def test_match_ignores_port_suffix(self):
        route = self.middleware._match("omniroute.g4f.local:8080")
        self.assertIsNotNone(route)
        self.assertEqual(route.name, "omniroute")

    def test_match_is_case_insensitive(self):
        route = self.middleware._match("OmniRoute.G4F.Local")
        self.assertIsNotNone(route)
        self.assertEqual(route.name, "omniroute")

    def test_match_rejects_unknown_name(self):
        self.assertIsNone(self.middleware._match("evil.g4f.local"))

    def test_match_rejects_unknown_port(self):
        self.assertIsNone(self.middleware._match("9999.g4f.local"))

    def test_match_rejects_nested_label(self):
        self.assertIsNone(self.middleware._match("a.omniroute.g4f.local"))

    def test_match_rejects_other_suffix(self):
        self.assertIsNone(self.middleware._match("omniroute.example.com"))

    def test_match_rejects_bare_suffix(self):
        self.assertIsNone(self.middleware._match("g4f.local"))

    def test_match_rejects_short_port_label(self):
        self.assertIsNone(self.middleware._match("808.g4f.local"))


class TestDomainRouterDispatch(unittest.IsolatedAsyncioTestCase):
    async def _call(self, host, path="/v1/models", scope_type="http"):
        """Run the middleware and return the scope the inner app received."""
        inner = _RecordingApp()
        middleware = DomainRouterMiddleware(inner)
        scope = _http_scope(host, path)
        scope["type"] = scope_type
        if scope_type == "websocket":
            scope["scheme"] = "ws"
        await middleware(scope, _receive, _send)
        self.assertTrue(inner.scopes, "inner app was not called")
        return inner.scopes[0]

    async def test_rewrites_path_for_prefix_route(self):
        scope = await self._call("remote-desktop.g4f.local")
        self.assertEqual(scope["path"], "/remote_desktop/v1/models")
        self.assertEqual(scope["raw_path"], b"/remote_desktop/v1/models")

    async def test_rewrites_path_for_desktop_route(self):
        scope = await self._call("desktop.g4f.local", "/")
        self.assertEqual(scope["path"], "/desktop/")

    async def test_rewrites_websocket_path(self):
        scope = await self._call(
            "remote-desktop.g4f.local", "/ws", scope_type="websocket"
        )
        self.assertEqual(scope["path"], "/remote_desktop/ws")

    async def test_g4f_host_is_untouched(self):
        scope = await self._call("8080.g4f.local")
        self.assertEqual(scope["path"], "/v1/models")

    async def test_unmatched_host_is_untouched(self):
        scope = await self._call("evil.g4f.local")
        self.assertEqual(scope["path"], "/v1/models")

    async def test_original_scope_is_not_mutated(self):
        inner = _RecordingApp()
        middleware = DomainRouterMiddleware(inner)
        scope = _http_scope("remote-desktop.g4f.local")
        await middleware(scope, _receive, _send)
        self.assertEqual(scope["path"], "/v1/models")

    async def test_root_mounted_route_forwards_to_target(self):
        # A route without a path proxy is forwarded by a root-mounted
        # LocalGatewayProxy; the target is unreachable here, so the proxy
        # answers with its own 502 payload instead of reaching the inner app.
        # Autostart is off, so nothing tries to install and start the app.
        inner = _RecordingApp()
        middleware = DomainRouterMiddleware(inner)
        scope = _http_scope("litellm.g4f.local")
        sent = []

        async def send(message):
            sent.append(message)

        with patch.dict(os.environ, {"G4F_DOMAIN_AUTOSTART": "0"}):
            await middleware(scope, _receive, send)
        self.assertEqual(inner.scopes, [])
        self.assertEqual(sent[0]["status"], 502)

    async def test_root_mounted_route_forwards_websocket(self):
        # A route without a path proxy forwards WebSockets through its
        # root-mounted LocalGatewayProxy; the target is unreachable here, so
        # the proxy closes the socket instead of reaching the inner app.
        inner = _RecordingApp()
        middleware = DomainRouterMiddleware(inner)
        scope = _http_scope("litellm.g4f.local", "/ws")
        scope["type"] = "websocket"
        scope["scheme"] = "ws"
        sent = []

        async def send(message):
            sent.append(message)

        await middleware(scope, _ws_receive(), send)
        self.assertEqual(inner.scopes, [])
        self.assertEqual(sent[0]["type"], "websocket.accept")
        self.assertEqual(sent[-1]["type"], "websocket.close")
        self.assertEqual(sent[-1]["code"], 1011)


class TestWebsocketUrl(unittest.TestCase):
    def test_http_becomes_ws(self):
        self.assertEqual(
            websocket_url("http://127.0.0.1:4000", "ws/echo"),
            "ws://127.0.0.1:4000/ws/echo",
        )

    def test_https_becomes_wss(self):
        self.assertEqual(
            websocket_url("https://example.test", "/ws"),
            "wss://example.test/ws",
        )

    def test_keeps_base_path(self):
        self.assertEqual(
            websocket_url("http://127.0.0.1:4000/api", "ws"),
            "ws://127.0.0.1:4000/api/ws",
        )

    def test_keeps_query(self):
        self.assertEqual(
            websocket_url("http://127.0.0.1:4000", "ws", "a=1&b=2"),
            "ws://127.0.0.1:4000/ws?a=1&b=2",
        )

    def test_empty_path_uses_root(self):
        self.assertEqual(
            websocket_url("http://127.0.0.1:4000", ""),
            "ws://127.0.0.1:4000/",
        )


class TestDomainRouterConfig(unittest.TestCase):
    def test_default_suffix(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertEqual(domain_suffix(), DEFAULT_SUFFIX)

    def test_suffix_from_env(self):
        with patch.dict("os.environ", {"G4F_DOMAIN_SUFFIX": ".g4f.test."}):
            self.assertEqual(domain_suffix(), "g4f.test")

    def test_enabled_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertTrue(domain_router_enabled())

    def test_disabled_by_env(self):
        with patch.dict("os.environ", {"G4F_DOMAIN_ROUTER": "0"}):
            self.assertFalse(domain_router_enabled())

    def test_register_is_noop_when_disabled(self):
        class _App:
            def __init__(self):
                self.middleware = []

            def add_middleware(self, *args, **kwargs):
                self.middleware.append((args, kwargs))

        app = _App()
        with patch.dict("os.environ", {"G4F_DOMAIN_ROUTER": "0"}):
            register_domain_router(app)
        self.assertEqual(app.middleware, [])

    def test_register_adds_middleware(self):
        class _App:
            def __init__(self):
                self.middleware = []

            def add_middleware(self, *args, **kwargs):
                self.middleware.append((args, kwargs))

        app = _App()
        with patch.dict("os.environ", {"G4F_DOMAIN_SUFFIX": "g4f.test"}):
            register_domain_router(app)
        self.assertEqual(len(app.middleware), 1)
        self.assertEqual(app.middleware[0][1]["suffix"], "g4f.test")


class TestHostsEntries(unittest.TestCase):
    def test_contains_names_and_ports(self):
        entries = hosts_entries()
        self.assertIn("127.0.0.1 omniroute.g4f.local", entries)
        self.assertIn("127.0.0.1 20128.g4f.local", entries)
        self.assertIn("127.0.0.1 desktop.g4f.local", entries)
        self.assertIn("127.0.0.1 8080.g4f.local", entries)

    def test_every_route_is_covered(self):
        entries = set(hosts_entries())
        for route in routes().values():
            self.assertIn(f"127.0.0.1 {route.name}.g4f.local", entries)
            if route.port:
                self.assertIn(f"127.0.0.1 {route.port}.g4f.local", entries)

    def test_custom_suffix(self):
        entries = hosts_entries("g4f.test")
        self.assertIn("127.0.0.1 omniroute.g4f.test", entries)


class TestDomainRouterIntegration(unittest.TestCase):
    """End-to-end checks through the real app, using the Host header."""

    def setUp(self):
        self.client = TestClient(g4f.api.create_app())

    def _unreachable(self, name: str):
        """Point one route at a closed port so the proxy cannot connect."""
        registry = routes()
        registry[name] = replace(
            registry[name],
            root_url=lambda: f"http://127.0.0.1:{_closed_port()}",
        )
        return patch("g4f.api.domain_router.routes", return_value=registry)

    def test_g4f_host_serves_api(self):
        response = self.client.get("/v1/models", headers={"host": "8080.g4f.local"})
        self.assertEqual(response.status_code, 200)

    def test_unmatched_host_serves_api(self):
        response = self.client.get("/v1/models", headers={"host": "evil.g4f.local"})
        self.assertEqual(response.status_code, 200)

    def test_unknown_port_host_serves_api(self):
        response = self.client.get("/v1/models", headers={"host": "9999.g4f.local"})
        self.assertEqual(response.status_code, 200)

    def test_gateway_host_reports_unavailable(self):
        with self._unreachable("omniroute"):
            response = self.client.get(
                "/v1/models", headers={"host": "omniroute.g4f.local"}
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "omniroute_unavailable")

    def test_port_host_reaches_same_gateway(self):
        with self._unreachable("omniroute"):
            response = self.client.get(
                "/v1/models", headers={"host": "20128.g4f.local"}
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "omniroute_unavailable")

    def test_ollama_host_reports_unavailable(self):
        with self._unreachable("ollama"):
            response = self.client.get(
                "/api/tags", headers={"host": "ollama.g4f.local"}
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "ollama_unavailable")

    def test_remote_desktop_host_rewrites_to_prefix(self):
        with self._unreachable("remote-desktop"):
            response = self.client.get(
                "/api/status", headers={"host": "remote-desktop.g4f.local"}
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["error"]["type"], "remote desktop_unavailable"
        )

    def test_gateway_host_relays_websocket(self):
        """A root-mounted route proxies WebSocket frames to the target app."""
        upstream = FastAPI()

        @upstream.websocket("/ws/echo")
        async def echo(websocket: WebSocket):
            await websocket.accept(
                subprotocol=websocket.headers.get("sec-websocket-protocol")
            )
            try:
                while True:
                    await websocket.send_text(f"echo:{await websocket.receive_text()}")
            except WebSocketDisconnect:
                pass

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(upstream, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            self.assertTrue(server.started, "upstream server did not start")

            registry = routes()
            registry["litellm"] = replace(
                registry["litellm"], root_url=lambda: f"http://127.0.0.1:{port}"
            )
            with patch("g4f.api.domain_router.routes", return_value=registry):
                client = TestClient(g4f.api.create_app())
                with client.websocket_connect(
                    "/ws/echo",
                    headers={
                        "host": "litellm.g4f.local",
                        "sec-websocket-protocol": "chat",
                    },
                ) as websocket:
                    self.assertEqual(websocket.accepted_subprotocol, "chat")
                    websocket.send_text("hello")
                    self.assertEqual(websocket.receive_text(), "echo:hello")
                    websocket.send_text("again")
                    self.assertEqual(websocket.receive_text(), "echo:again")
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    def test_gateway_host_forwards_websocket_credentials(self):
        """Cookies and Authorization survive the WebSocket handshake."""
        upstream = FastAPI()
        seen = {}

        @upstream.websocket("/ws/auth")
        async def auth(websocket: WebSocket):
            seen.update(websocket.headers)
            await websocket.accept()
            await websocket.close()

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(upstream, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            self.assertTrue(server.started, "upstream server did not start")

            registry = routes()
            registry["litellm"] = replace(
                registry["litellm"], root_url=lambda: f"http://127.0.0.1:{port}"
            )
            with patch("g4f.api.domain_router.routes", return_value=registry):
                client = TestClient(g4f.api.create_app())
                with client.websocket_connect(
                    "/ws/auth",
                    headers={
                        "host": "litellm.g4f.local",
                        "cookie": "session=abc",
                        "authorization": "Bearer token",
                    },
                ):
                    pass
        finally:
            server.should_exit = True
            thread.join(timeout=5)

        self.assertEqual(seen.get("cookie"), "session=abc")
        self.assertEqual(seen.get("authorization"), "Bearer token")
        self.assertEqual(seen.get("x-forwarded-host"), "litellm.g4f.local")
        self.assertEqual(seen.get("x-forwarded-proto"), "ws")
        # aiohttp must generate its own handshake key, not reuse the browser's.
        self.assertTrue(seen.get("sec-websocket-key"))

    def test_gateway_host_closes_websocket_when_unreachable(self):
        with self._unreachable("litellm"):
            client = TestClient(g4f.api.create_app())
            with client.websocket_connect(
                "/ws", headers={"host": "litellm.g4f.local"}
            ) as websocket:
                message = websocket.receive()
        self.assertEqual(message["type"], "websocket.close")
        self.assertEqual(message["code"], 1011)

    def test_desktop_host_serves_novnc_client(self):
        with tempfile.TemporaryDirectory() as novnc_dir:
            with open(os.path.join(novnc_dir, "index.html"), "w") as fh:
                fh.write("<html>novnc</html>")
            with patch.dict(
                "os.environ",
                {"G4F_VNC_PORT": "5900", "G4F_NOVNC_DIR": novnc_dir},
            ):
                client = TestClient(g4f.api.create_app())
                response = client.get(
                    "/", headers={"host": "desktop.g4f.local"}, follow_redirects=False
                )
        # Without the rewrite this would be the g4f API's own "/" redirect.
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
