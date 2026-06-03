from __future__ import annotations

import os
import re
import time
import base64
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .template import OpenaiTemplate
from ..typing import AsyncResult, Messages
from .. import debug

# ---------------------------------------------------------------------------
# Seed list & Cache — known open Ollama servers
# 
# [!] PRO TIP: If these seed servers ever become slow or dead (since they are 
# public/home nodes), you can easily find 100+ fresh ones using our built-in tool:
# Run: python etc/tool/fofa_ollama_parser.py --help
# ---------------------------------------------------------------------------
import json
from pathlib import Path
from ..cookies import get_cookies_dir

_CACHE_FILE = Path(get_cookies_dir()) / "ollama_servers.json"

_DEFAULT_SEED_SERVERS = [
    "http://220.249.186.40:11434",
    "http://90.149.239.71:11434",
    "http://125.227.28.166:11434",
    "http://160.16.60.183:11434",
    "http://211.73.161.201:11434",
    "http://223.85.216.230:11434",
    "http://202.141.161.50:11434",
    "http://150.230.164.69:11434",
    "http://57.128.64.100:11434",
    "http://35.221.126.180:11434",
    "http://136.116.54.121:11434",
    "http://38.180.104.127:11434",
    "http://87.98.145.87:11434",
    "http://79.43.23.226:11434",
    "http://180.114.6.82:11434",
    "http://64.225.38.49:11434",
    "http://114.34.180.200:11434",
    "http://116.234.35.242:11434",
    "http://76.93.107.161:11434",
    "http://13.140.25.193:11434",
    "http://51.254.134.96:11434",
    "http://193.237.153.60:11434",
    "http://178.104.205.2:11434",
    "http://37.59.98.74:11434",
    "http://81.131.169.17:11434",
    "http://193.237.205.200:11434",
    "http://1.255.85.149:11434",
    "http://31.70.78.250:11434",
    "http://203.176.113.216:11434",
    "http://71.251.218.102:11434",
    "http://83.86.59.188:11434",
    "http://199.204.135.71:11434",
    "http://101.111.228.63:11434",
    "http://168.235.74.31:11434",
    "http://217.182.133.168:11434",
    "http://223.113.66.126:11434",
    "http://220.134.52.221:11434",
    "http://47.79.39.175:11434",
    "http://158.69.27.163:11434",
    "http://152.67.134.205:11434",
    "http://213.136.76.182:11434",
    "http://178.254.28.95:11434",
    "http://77.68.10.64:11434",
    "http://45.87.137.100:11434",
    "http://57.128.123.135:11434",
    "http://18.136.206.156:11434",
    "http://88.168.52.207:11434",
    "http://145.239.207.5:11434",
    "http://167.86.113.188:11434",
    "http://58.127.230.165:11434",
    "http://223.113.254.84:11434",
    "http://51.254.130.116:11434",
    "http://64.176.229.210:11434",
    "http://150.136.60.84:11434",
    "http://209.97.173.219:11434",
    "http://109.86.166.86:11434",
    "http://103.66.120.232:11434",
    "http://118.163.0.89:11434",
    "http://64.176.39.95:11434",
    "http://178.105.145.53:11434",
    "http://59.125.184.40:11434",
    "http://92.29.91.135:11434",
    "http://210.59.176.82:11434",
    "http://167.71.147.184:11434",
    "http://108.160.206.30:11434",
    "http://163.13.128.47:11434",
    "http://31.70.86.211:11434",
    "http://207.148.68.227:11434",
    "http://139.129.25.182:11434",
    "http://5.78.200.46:11434",
    "http://94.141.160.99:11434",
    "http://223.166.234.219:11434",
    "http://24.236.158.179:11434",
    "http://217.174.245.24:11434",
    "http://201.137.77.153:11434",
    "http://79.157.228.102:11434",
    "http://86.220.0.198:11434",
    "http://188.166.254.32:11434",
    "http://46.224.83.114:11434",
    "http://178.151.36.115:11434",
    "http://217.182.67.5:11434",
    "http://133.4.188.2:11434",
    "http://1.243.43.248:11434",
    "http://211.23.87.144:11434",
    "http://5.101.168.158:11434",
    "http://167.114.192.243:11434",
    "http://27.92.231.18:11434",
    "http://220.135.48.55:11434",
    "http://59.10.172.168:11434",
    "http://178.105.202.139:11434",
    "http://121.190.96.209:11434",
    "http://49.13.102.77:11434",
    "http://132.226.20.20:11434",
    "http://125.138.77.111:11434",
    "http://51.77.188.225:11434",
    "http://180.110.147.114:11434",
    "http://60.185.196.91:11434",
    "http://178.63.104.147:11434",
    "http://51.178.49.219:11434",
    "http://151.80.21.134:11434"
]

def _get_cached_servers() -> list[str]:
    servers = set(_DEFAULT_SEED_SERVERS)
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    servers.update(data)
    except Exception as e:
        debug.error(f"OllamaSwarm: failed to read cache: {e}")
    return list(servers)

def _save_servers_to_cache(servers: list[str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(servers, f)
    except Exception as e:
        debug.error(f"OllamaSwarm: failed to save cache: {e}")

# ---------------------------------------------------------------------------
# FOFA discovery — free public API
# ---------------------------------------------------------------------------
_FOFA_API = "https://fofa.info/api/v1/search/all"
_FOFA_QUERY = 'port="11434" && body="Ollama"'
_FOFA_FIELDS = "ip,port"


def _fofa_discover(max_results: int = 50) -> list[str]:
    """Fetch Ollama endpoints from FOFA public API (requires FOFA_EMAIL + FOFA_KEY).

    Returns list of 'http://ip:port' strings. Returns empty list on failure.
    """
    email = os.environ.get("FOFA_EMAIL", "")
    key = os.environ.get("FOFA_KEY", "")
    if not email or not key:
        return []

    try:
        qbase64 = base64.b64encode(_FOFA_QUERY.encode()).decode()
        params = {
            "email": email,
            "key": key,
            "qbase64": qbase64,
            "page": 1,
            "size": min(max_results, 100),
            "fields": _FOFA_FIELDS,
        }
        resp = requests.get(_FOFA_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("error"):
            debug.error(f"OllamaSwarm FOFA: {data.get('errmsg', 'unknown error')}")
            return []

        results = data.get("results", [])
        endpoints = []
        for item in results:
            if isinstance(item, list) and len(item) >= 2:
                ip, port = item[0], item[1]
                endpoints.append(f"http://{ip}:{port}")
            elif isinstance(item, dict):
                ip = item.get("ip", "")
                port = item.get("port", 11434)
                if ip:
                    endpoints.append(f"http://{ip}:{port}")

        debug.log(f"OllamaSwarm: FOFA returned {len(endpoints)} endpoints")
        return endpoints

    except Exception as exc:
        debug.error(f"OllamaSwarm FOFA discovery failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Server validation
# ---------------------------------------------------------------------------
_PROBE_TIMEOUT = 5
_PROBE_WORKERS = 20


def _probe_server(url: str) -> tuple[str, list[str]] | None:
    """Probe a single Ollama server. Returns (url, [model_names]) or None."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=_PROBE_TIMEOUT)
        resp.raise_for_status()
        models = [
            m.get("name", "")
            for m in resp.json().get("models", [])
            if m.get("name") and "-cloud" not in m.get("name", "")
        ]
        if models:
            return url, models
    except Exception:
        pass
    return None


def _discover_servers() -> dict[str, list[str]]:
    """Discover alive Ollama servers and their models.

    Returns dict: {server_url: [model1, model2, ...]}
    """
    # Collect candidate URLs
    candidates = _get_cached_servers()

    # Try FOFA discovery
    fofa_results = _fofa_discover()
    for url in fofa_results:
        if url not in candidates:
            candidates.append(url)

    debug.log(f"OllamaSwarm: probing {len(candidates)} candidate servers ...")

    # Validate concurrently
    alive: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
        futures = {pool.submit(_probe_server, url): url for url in candidates}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                url, models = result
                alive[url] = models

    debug.log(f"OllamaSwarm: {len(alive)} servers alive with models")
    
    # Save newly discovered alive servers to cache
    if alive:
        _save_servers_to_cache(list(alive.keys()))
        
    return alive


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
_CACHE_TTL = 3600  # re-discover every hour


class OllamaSwarm(OpenaiTemplate):
    label = "Ollama Swarm 🐝"
    url = "https://ollama.com"
    needs_auth = False
    working = True
    active_by_default = True
    default_model = "qwen3:14b"
    sort_models = True

    # Maps model name -> list of server base URLs
    model_to_servers: dict[str, list[str]] = {}
    _cache_time: float = 0

    @classmethod
    def get_models(cls, api_key: str = None, base_url: str = None, **kwargs) -> list[str]:
        now = time.time()

        # Return cached models if still valid
        if cls.models and (now - cls._cache_time) < _CACHE_TTL:
            return cls.models

        # Discover servers
        alive = _discover_servers()

        if not alive:
            # If nothing found and we had a cache, keep using it
            if cls.models:
                debug.log("OllamaSwarm: no servers found, using cached models")
                return cls.models
            return []

        # Build model -> servers mapping
        cls.models = []
        cls.model_to_servers = {}
        seen = set()

        for server_url, models in alive.items():
            for name in models:
                if name not in seen:
                    seen.add(name)
                    cls.models.append(name)
                
                if name not in cls.model_to_servers:
                    cls.model_to_servers[name] = []
                cls.model_to_servers[name].append(server_url)

        if cls.models:
            cls.live += 1

        if cls.default_model not in seen and cls.models:
            cls.default_model = cls.models[0]

        if cls.sort_models:
            cls.models.sort()

        cls._cache_time = now
        debug.log(f"OllamaSwarm: {len(cls.models)} models from {len(alive)} servers")
        return cls.models

    @classmethod
    async def get_quota(cls, api_key: str = None, **kwargs) -> dict:
        return {}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ) -> AsyncResult:
        if not cls.models:
            cls.get_models()

        server_urls = cls.model_to_servers.get(model)
        if server_urls is None:
            resolved = cls.get_model(model)
            server_urls = cls.model_to_servers.get(resolved)
            if server_urls is not None:
                model = resolved

        if server_urls is None:
            raise ValueError(
                f"OllamaSwarm: model '{model}' not found on any server. "
                f"Available: {list(cls.model_to_servers.keys())[:10]}"
            )

        last_error = None
        for server_url in server_urls:
            base_url = f"{server_url}/v1"
            debug.log(f"OllamaSwarm: trying server {server_url} for model {model}")
            try:
                gen = super().create_async_generator(
                    model,
                    messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
                
                # TTFT Timeout: Wait max 10 seconds for the first token chunk
                try:
                    first_chunk = await asyncio.wait_for(gen.__anext__(), timeout=10.0)
                    yield first_chunk
                except StopAsyncIteration:
                    pass
                except asyncio.TimeoutError:
                    raise Exception("TTFT Timeout: Model took too long to start generating (>10s)")

                # If first chunk succeeded, stream the rest without strict chunk timeout
                async for chunk in gen:
                    yield chunk
                
                # If we get here, generation succeeded on this server
                return
            except Exception as e:
                debug.error(f"OllamaSwarm: server {server_url} failed with error: {e}")
                last_error = e
                # If it failed AFTER yielding some chunks, we can't seamlessly switch
                # to a new server, because the output would be broken/duplicated.
                # However, with our explicit TTFT check, any failure in first chunk is safely caught!
                # Wait, how to know if we yielded? We yielded `first_chunk`.
                # If we threw an error DURING the first chunk fetch, we never yielded, so we can retry!
                # If it threw inside the subsequent `async for`, we already yielded, so we must raise.
                if 'first_chunk' in locals():
                    raise e
                # Otherwise, it failed before yielding anything, so try the next server!
                continue
        
        # If all servers failed before yielding any chunks
        if last_error:
            raise last_error
