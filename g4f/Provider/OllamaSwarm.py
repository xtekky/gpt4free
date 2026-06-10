from __future__ import annotations

import os
import time
import base64
import requests
import asyncio
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from .template import OpenaiTemplate
from ..typing import AsyncResult, Messages
from ..errors import BadRequestError
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


#     setInterval(()=>{
#         window.urls = window.urls || JSON.parse(localStorage.getItem('urls')) || [];
#         document.querySelectorAll('.hsxa-host a').forEach(e=>{
#             if (!window.urls.includes(e.href)) {
#                 window.urls.push(e.href);
#             }
#         })
#         localStorage.setItem('urls', JSON.stringify(window.urls));
#         console.log(window.urls.length);
#     }, 1000);

_DEFAULT_SEED_SERVERS = [
  "http://116.202.111.94:11434",
  "http://130.89.48.109:11434",
  "http://85.214.43.150:11434",
  "http://130.89.48.109:11434",
  "http://213.132.219.17:11434",
  "http://155.133.208.195:11434",
  "http://46.17.99.157:11434",
  "http://193.237.205.200:11434",
  "http://51.17.50.106:2052",
  "http://46.224.147.115:11434",
  "http://54.215.114.112:7547",
  "http://43.210.64.106:7547",
  "http://87.208.240.33:11434",
  "http://64.176.39.95:11434",
  "http://46.224.156.158:11434",
  "http://185.100.232.224:11434",
  "http://38.76.189.45:11434",
  "http://38.76.189.41:11434",
  "http://116.203.177.162:11434",
  "http://204.168.244.123:11434",
  "http://195.201.234.76:11434",
  "http://194.62.157.184:11434",
  "http://185.45.193.80:11434",
  "http://199.79.202.22:11434",
  "http://78.46.41.183:11434",
  "http://1.243.43.248:11434",
  "http://43.210.64.106:7547",
  "http://38.76.189.74:11434",
  "http://213.136.76.182:11434",
  "http://89.58.3.79:11434",
  "http://116.203.112.201:11434",
  "http://100.30.6.43:11434",
  "http://146.0.72.136:11434",
  "http://204.168.244.123:11434",
  "http://44.244.46.70:7547",
  "http://193.237.153.60:11434",
  "http://79.137.197.6:11434",
  "http://220.249.186.40:11434",
  "http://103.235.75.117:11434",
  "http://202.141.161.50:11434",
  "http://66.94.124.143:11434",
  "http://94.141.160.99:11434",
  "http://38.180.104.127:11434",
  "http://147.93.183.134:11434",
  "http://82.135.28.45:11434",
  "http://152.53.93.215:11434",
  "http://45.145.42.104:11434",
  "http://203.176.113.216:11434",
  "http://85.214.43.150:11434",
  "http://133.4.188.2:11434",
  "http://116.202.197.155:11434",
  "http://57.128.123.135:11434",
  "http://82.135.28.45:11434",
  "http://63.177.73.22:7547",
  "http://211.23.87.144:11434",
  "http://204.168.139.0:11434",
  "http://16.26.230.113:7547",
  "http://46.224.83.114:11434",
  "http://20.246.91.177:11434",
  "http://116.203.112.201:11434",
  "http://168.235.74.31:11434",
  "http://84.22.103.64:11434",
  "http://223.113.66.126:11434",
  "http://38.76.189.19:11434",
  "http://194.62.157.184:11434",
  "http://81.4.125.240:11434",
  "http://209.97.173.219:11434",
  "http://163.172.212.132:11434",
  "http://116.202.111.94:11434",
  "http://18.223.75.148:11434",
  "http://213.132.219.17:11434",
  "http://139.129.25.182:11434",
  "http://62.45.168.106:11434",
  "http://46.4.216.118:11434",
  "http://117.55.199.23:11434",
  "http://31.172.78.56:11434",
  "http://62.171.155.8:11434",
  "http://49.13.48.26:11434",
  "http://82.165.174.61:11434",
  "http://116.203.53.120:11434",
  "http://31.172.78.56:11434",
  "http://108.160.206.30:11434",
  "http://116.203.219.128:11434",
  "http://161.153.32.111:11434",
  "http://217.174.245.24:11434",
  "http://77.239.123.2:11434",
  "http://18.223.75.148:11434",
  "http://89.58.3.79:11434",
  "http://5.75.180.13:11434",
  "http://34.31.140.94:11434",
  "http://204.168.198.89:11434",
  "http://45.87.137.100:11434",
  "http://107.175.125.166:11434",
  "http://20.246.91.177:11434",
  "http://185.45.193.80:11434",
  "http://204.168.139.0:11434",
  "http://178.105.145.53:11434",
  "http://64.156.70.180:11434",
  "http://165.1.76.13:11434",
  "http://62.45.168.106:11434",
  "http://158.101.214.195:11434",
  "http://5.9.1.80:11434",
  "http://82.165.174.61:11434",
  "http://63.177.73.22:7547",
  "http://77.239.123.2:11434",
  "http://51.178.49.219:11434",
  "http://116.203.198.188:11434",
  "http://38.76.189.18:11434",
  "http://216.70.69.75:11434",
  "http://88.168.52.207:11434",
  "http://62.238.14.177:11434",
  "http://142.132.252.21:11434",
  "http://38.76.189.9:11434",
  "http://155.133.208.195:11434",
  "http://167.71.147.184:11434",
  "http://35.221.126.180:11434",
  "http://63.179.110.87:2082",
  "http://3.67.10.231:8080",
  "http://199.79.202.22:11434",
  "http://178.104.205.2:11434",
  "http://116.203.177.162:11434",
  "http://165.1.78.194:11434",
  "http://3.67.10.231:8080",
  "http://5.129.226.192:11434",
  "http://198.206.133.250:11434",
  "http://178.254.28.95:11434",
  "http://146.0.72.136:11434",
  "http://16.63.120.43:7547",
  "http://204.168.196.150:11434",
  "http://46.224.186.78:11434",
  "http://136.243.60.49:11434",
  "http://116.202.9.89:11434",
  "http://46.224.186.78:11434",
  "http://116.203.219.128:11434",
  "http://38.76.189.31:11434",
  "http://75.128.229.121:11434",
  "http://16.26.230.113:7547",
  "http://52.201.213.145:11434",
  "http://5.78.200.46:11434",
  "http://2.59.170.202:11434",
  "http://185.191.127.178:11434",
  "http://81.131.169.17:11434",
  "http://178.105.145.53:11434",
  "http://116.203.212.217:11434",
  "http://54.215.114.112:7547",
  "http://178.104.197.254:11434",
  "http://116.203.212.217:11434",
  "http://5.9.1.80:11434",
  "http://18.61.29.191:7547",
  "http://63.179.110.87:2082",
  "http://54.36.111.107:11434",
  "http://103.66.120.232:11434",
  "http://27.92.231.18:11434",
  "http://79.137.197.6:11434",
  "http://51.17.50.106:2052",
  "http://178.105.62.143:11434",
  "http://101.111.228.63:11434",
  "http://150.230.164.69:11434",
  "http://85.214.44.11:11434",
  "http://178.104.163.52:11434",
  "http://45.145.42.104:11434",
  "http://34.16.62.196:11434",
  "http://49.13.102.77:11434",
  "http://195.201.234.76:11434",
  "http://38.76.189.21:11434",
  "http://145.239.207.5:11434",
  "http://100.30.6.43:11434",
  "http://38.76.189.97:11434",
  "http://45.139.77.246:11434",
  "http://45.154.87.43:11434",
  "http://49.13.48.26:11434",
  "http://64.188.91.237:11434",
  "http://38.76.189.18:11434",
  "http://178.105.147.204:11434",
  "http://77.68.10.64:11434",
  "http://52.201.213.145:11434",
  "http://125.138.77.111:11434",
  "http://223.85.216.230:11434",
  "http://57.128.64.100:11434",
  "http://178.219.166.81:2082",
  "http://37.59.98.74:11434",
  "http://87.208.240.33:11434",
  "http://71.251.218.102:11434",
  "http://165.1.78.194:11434",
  "http://178.105.147.204:11434",
  "http://210.59.176.82:11434",
  "http://34.31.140.94:11434",
  "http://84.22.103.64:11434",
  "http://223.113.254.84:11434",
  "http://113.44.194.208:11434",
  "http://3.237.237.228:2077",
  "http://44.244.46.70:7547",
  "http://13.140.143.210:11434",
  "http://64.188.91.237:11434",
  "http://38.76.189.97:11434",
  "http://62.171.155.8:11434",
  "http://204.168.149.141:11434",
  "http://51.254.134.96:11434",
  "http://109.86.166.86:11434",
  "http://178.105.66.185:11434",
  "http://104.54.238.64:11434",
  "http://116.202.66.86:11434",
  "http://129.80.194.194:11434",
  "http://45.140.140.26:11434",
  "http://45.140.140.26:11434",
  "http://199.204.135.71:11434",
  "http://135.237.98.245:11434",
  "http://23.95.148.22:11434",
  "http://5.75.180.13:11434",
  "http://78.13.53.95:2077",
  "http://31.70.86.211:11434",
  "http://158.101.214.195:11434",
  "http://13.140.143.210:11434",
  "http://136.243.60.49:11434",
  "http://83.86.59.188:11434",
  "http://198.206.133.250:11434",
  "http://69.243.159.16:11434",
  "http://51.158.152.190:11434",
  "http://178.105.62.143:11434",
  "http://34.16.62.196:11434",
  "http://204.168.198.89:11434",
  "http://116.202.66.86:11434",
  "http://78.13.53.95:2077",
  "http://167.86.113.188:11434",
  "http://51.77.188.225:11434",
  "http://90.149.239.71:11434",
  "http://51.254.130.116:11434",
  "http://64.156.70.180:11434",
  "http://180.110.147.114:11434",
  "http://47.79.39.175:11434",
  "http://20.107.59.198:11434",
  "http://16.63.120.43:7547",
  "http://114.34.180.200:11434",
  "http://84.86.220.240:11434",
  "http://38.76.189.45:11434",
  "http://204.168.175.197:11434",
  "http://129.80.194.194:11434",
  "http://2.59.170.202:11434",
  "http://116.202.197.155:11434",
  "http://78.46.41.183:11434",
  "http://18.61.29.191:7547",
  "http://46.224.156.158:11434",
  "http://204.168.196.150:11434",
  "http://62.68.75.4:11434",
  "http://161.153.32.111:11434",
  "http://152.53.251.120:11434",
  "http://204.168.149.141:11434",
  "http://129.80.43.33:11434",
  "http://207.148.68.227:11434",
  "http://117.50.171.144:11434",
  "http://46.243.3.122:11434",
  "http://220.135.48.55:11434",
  "http://18.136.206.156:11434",
  "http://58.127.230.165:11434",
  "http://45.139.77.246:11434",
  "http://188.166.254.32:11434",
  "http://178.104.197.254:11434",
  "http://201.137.77.153:11434",
  "http://135.237.98.245:11434",
  "http://107.175.125.166:11434",
  "http://38.76.189.41:11434",
  "http://23.95.148.22:11434",
  "http://103.137.250.43:11434",
  "http://204.168.254.120:11434",
  "http://185.100.232.224:11434",
  "http://150.136.60.84:11434",
  "http://220.134.52.221:11434",
  "http://66.94.124.143:11434",
  "http://204.168.254.120:11434",
  "http://163.172.212.132:11434",
  "http://38.76.189.74:11434",
  "http://45.154.87.43:11434",
  "http://54.93.197.63:7547",
  "http://217.182.133.168:11434",
  "http://35.208.40.227:11434",
  "http://129.80.43.33:11434",
  "http://84.86.220.240:11434",
  "http://165.1.76.13:11434",
  "http://38.76.189.19:11434",
  "http://116.203.198.188:11434",
  "http://116.202.9.89:11434",
  "http://46.4.216.118:11434",
  "http://152.53.251.120:11434",
  "http://178.105.66.185:11434",
  "http://142.132.252.21:11434",
  "http://103.235.75.117:11434",
  "http://85.214.44.11:11434",
  "http://46.243.3.122:11434",
  "http://185.191.127.178:11434",
  "http://147.93.139.24:11434",
  "http://113.44.194.208:11434",
  "http://217.182.67.5:11434",
  "http://81.4.125.240:11434",
  "http://62.68.75.4:11434",
  "http://5.101.168.158:11434",
  "http://117.50.171.144:11434",
  "http://147.93.139.24:11434",
  "http://147.93.183.23:11434",
  "http://54.36.111.107:11434",
  "http://46.224.203.89:11434",
  "http://103.137.250.43:11434",
  "http://116.203.53.120:11434",
  "http://87.98.145.87:11434",
  "http://24.236.158.179:11434",
  "http://64.225.38.49:11434",
  "http://62.238.14.177:11434",
  "http://216.70.69.75:11434",
  "http://79.157.228.102:11434",
  "http://185.237.206.241:11434",
  "http://13.140.25.193:11434",
  "http://46.224.147.115:11434",
  "http://152.53.93.215:11434",
  "http://178.219.166.81:2082",
  "http://69.243.159.16:11434",
  "http://75.128.229.121:11434",
  "http://38.76.189.31:11434",
  "http://5.129.226.192:11434",
  "http://117.55.199.23:11434",
  "http://51.158.152.190:11434",
  "http://20.107.59.198:11434",
  "http://158.69.27.163:11434",
  "http://185.237.206.241:11434",
  "http://46.17.99.157:11434",
  "http://125.227.28.166:11434",
  "http://147.93.183.23:11434",
  "http://54.93.197.63:7547",
  "http://3.237.237.228:2077",
  "http://151.80.21.134:11434",
  "http://147.93.183.134:11434",
  "http://178.104.163.52:11434",
  "http://38.76.189.21:11434",
  "http://46.224.203.89:11434",
  "http://1.255.85.149:11434",
  "http://204.168.175.197:11434",
  "http://31.70.78.250:11434",
  "http://35.208.40.227:11434",
  "http://104.54.238.64:11434",
  "http://38.76.189.9:11434",
  "http://160.16.60.183:11434",
  "http://64.176.229.210:11434"
]

def get_cache_file() -> Path:
    return Path(get_cookies_dir()) / "ollama_servers.json"

def _get_cached_servers() -> list[str]:
    servers = set(_DEFAULT_SEED_SERVERS)
    try:
        cache_file = get_cache_file()
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    servers.update(data)
    except Exception as e:
        debug.error(f"OllamaSwarm: failed to read cache: {e}")
    return list(servers)

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
            if "/attacker/" not in m.get("name", "")
            and not m.get("name", "").startswith("model-b")
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
    models_file =  Path(get_cookies_dir()) / "models" / date.today().strftime("%Y-%m-%d") / "ollama-swarm.json"
    if models_file.exists():
        try:
            with open(models_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    debug.log(f"OllamaSwarm: loaded {len(data)} servers from today's cache")
                    return data
        except Exception as e:
            debug.error(f"OllamaSwarm: failed to read today's models cache: {e}")
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
        try:
            models_file.parent.mkdir(parents=True, exist_ok=True)
            with open(models_file, "w") as f:
                json.dump(alive, f)
        except Exception as e:
            debug.error(f"OllamaSwarm: failed to save today's models cache: {e}")
        
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
        cls.model_to_servers = {}
        for server_url, models in alive.items():
            for name in models:
                if name not in cls.model_to_servers:
                    cls.model_to_servers[name] = []
                cls.model_to_servers[name].append(server_url)
                cls.models_count[name] = cls.models_count.get(name, 0) + 1
        cls.models_count = dict(sorted(cls.models_count.items(), key=lambda item: item[1], reverse=True))
        cls.models = list(cls.models_count.keys())

        if cls.models:
            cls.live += 1

        if cls.default_model not in cls.models and cls.models:
            cls.default_model = cls.models[0]

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
        first_chunk = None
        for server_url in server_urls:
            base_url = f"{server_url}/v1"
            debug.log(f"OllamaSwarm: trying server {server_url} for model {model}")
            try:
                gen = super().create_async_generator(
                    model,
                    messages,
                    api_key=api_key,
                    base_url=base_url,
                    yield_request=False,
                    **kwargs,
                )
                
                # TTFT Timeout: Wait max 10 seconds for the first token chunk
                try:
                    first_chunk = await asyncio.wait_for(gen.__anext__(), timeout=10.0)
                    if first_chunk:
                        yield first_chunk
                except StopAsyncIteration:
                    pass
                except asyncio.TimeoutError:
                    raise Exception("TTFT Timeout: Model took too long to start generating (>10s)")

                # If first chunk succeeded, stream the rest without strict chunk timeout
                async for chunk in gen:
                    yield chunk

                cls.model_to_servers[model] = [server_url] + [s for s in cls.model_to_servers[model] if s != server_url]
                # If we get here, generation succeeded on this server
                return
            except BadRequestError as e:
                raise e  # BadRequestError means the server is alive but model is missing/invalid, no need to try other servers
            except Exception as e:
                debug.error(f"OllamaSwarm: server {server_url} failed with error: {e}")
                last_error = e
                # If it failed AFTER yielding some chunks, we can't seamlessly switch
                # to a new server, because the output would be broken/duplicated.
                # However, with our explicit TTFT check, any failure in first chunk is safely caught!
                # Wait, how to know if we yielded? We yielded `first_chunk`.
                # If we threw an error DURING the first chunk fetch, we never yielded, so we can retry!
                # If it threw inside the subsequent `async for`, we already yielded, so we must raise.
                if first_chunk:
                    raise e
                # Otherwise, it failed before yielding anything, so try the next server!
                continue
        
        # If all servers failed before yielding any chunks
        if last_error:
            raise last_error
