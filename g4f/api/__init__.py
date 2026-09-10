from __future__ import annotations

import inspect
import logging
import json
import asyncio
import uvicorn
import secrets
import os
import re
import shutil
import time
from collections import deque
from email.utils import formatdate
import os.path
import hashlib
import base64
from contextlib import asynccontextmanager
from urllib.parse import quote_plus, unquote_plus
from fastapi import FastAPI, Response, Request, UploadFile, Form, Depends, Header
from fastapi.responses import (
    StreamingResponse,
    RedirectResponse,
    HTMLResponse,
    JSONResponse,
    FileResponse,
)
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_502_BAD_GATEWAY,
)

try:
    from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
except ImportError:
    HTTP_422_UNPROCESSABLE_CONTENT = 422
from starlette.staticfiles import NotModifiedResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, HTTPBasic
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.background import BackgroundTask

try:
    from a2wsgi import WSGIMiddleware

    has_a2wsgi = True
except ImportError:
    has_a2wsgi = False
try:
    from PIL import Image

    has_pillow = True
except ImportError:
    has_pillow = False
from types import SimpleNamespace
from typing import Union, Optional, List

try:
    from typing import Annotated
except ImportError:

    class Annotated:
        pass


from g4f.requests import has_cdp

import g4f
import g4f.debug
from g4f.client import AsyncClient, ChatCompletion, ImagesResponse
from g4f.providers.response import BaseConversation, JsonConversation
from g4f.client.helper import filter_none
from g4f.config import DEFAULT_PORT, DEFAULT_TIMEOUT, DEFAULT_STREAM_TIMEOUT
from g4f.image import EXTENSIONS_MAP, is_data_an_media, process_image, is_safe_url
from g4f.image.copy_images import get_media_dir, copy_media, get_source_url, secure_filename
from g4f.errors import (
    ProviderNotFoundError,
    ModelNotFoundError,
    MissingAuthError,
    NoValidHarFileError,
    MissingRequirementsError,
    RateLimitError,
)
from g4f.cookies import read_cookie_files, get_cookies_dir
from g4f.providers.types import ProviderType
from g4f.providers.response import AudioResponse
from g4f.providers.any_provider import AnyProvider
from g4f.providers.any_model_map import (
    model_map,
    vision_models,
    image_models,
    audio_models,
    video_models,
)
from g4f.config import AppConfig
from g4f.client.factory import AbstractClientFactory
from g4f import Provider
from g4f.Provider import ProviderUtils

from g4f.gui import get_gui_app
from .stubs import (
    ChatCompletionsConfig,
    ImageGenerationConfig,
    ResponsesConfig,
    MessagesConfig,
    ProviderResponseModel,
    ModelResponseModel,
    ErrorResponseModel,
    ProviderResponseDetailModel,
    FileResponseModel,
    TranscriptionResponseModel,
    AudioSpeechConfig,
)
from g4f import debug

try:
    from g4f.gui.server.crypto import create_or_read_keys, decrypt_data, get_session_key

    has_crypto = True
except ImportError:
    has_crypto = False

logger = logging.getLogger(__name__)

V1_LANDING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#101411"><title>g4f /v1 API</title>
<style>
:root{--bg:#101411;--panel:#171d18;--ink:#f5f4ed;--muted:#a7afa4;--line:#344138;--lime:#c4f06c;--coral:#ff816d;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.55 Georgia,serif}body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.2;background-image:linear-gradient(#344138 1px,transparent 1px),linear-gradient(90deg,#344138 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(#000,transparent 72%)}a{color:var(--lime)}.wrap{width:min(1120px,calc(100% - 40px));margin:auto;position:relative}
header{display:flex;justify-content:space-between;align-items:center;padding:28px 0;border-bottom:1px solid var(--line)}.brand{color:var(--ink);text-decoration:none;font:700 18px/1 var(--mono);letter-spacing:-.04em}.brand b{color:var(--lime)}nav{display:flex;gap:22px;font:13px var(--mono)}nav a{color:var(--muted);text-decoration:none}nav a:hover{color:var(--lime)}
.hero{padding:96px 0 88px;display:grid;grid-template-columns:1.2fr .8fr;gap:70px;align-items:end}.eyebrow{color:var(--coral);font:12px var(--mono);letter-spacing:.12em;text-transform:uppercase}h1{max-width:720px;margin:20px 0;font-size:clamp(48px,8vw,94px);line-height:.94;letter-spacing:-.065em;font-weight:400}.lede{max-width:590px;color:var(--muted);font-size:20px}.hero-aside{border-left:2px solid var(--lime);padding-left:22px;color:var(--muted)}.hero-aside strong{display:block;color:var(--ink);font:700 14px var(--mono);margin-bottom:12px}.actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:30px}.button{display:inline-block;padding:11px 16px;border:1px solid var(--lime);color:var(--bg);background:var(--lime);font:700 13px var(--mono);text-decoration:none}.button.alt{color:var(--lime);background:transparent}
section{padding:28px 0 92px}.section-head{display:flex;justify-content:space-between;align-items:baseline;gap:20px;margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:14px}h2{margin:0;font-size:28px;font-weight:400;letter-spacing:-.04em}.section-head span{color:var(--muted);font:12px var(--mono)}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{display:block;padding:23px;border:1px solid var(--line);background:rgba(23,29,24,.88);min-height:174px;text-decoration:none}.card:hover{border-color:var(--lime)}.card h3{margin:0 0 8px;color:var(--ink);font-size:21px;font-weight:400}.card p{margin:0 0 18px;color:var(--muted)}code,pre{font-family:var(--mono)}.endpoint{color:var(--lime);font-size:12px}
.code-box{position:relative;border:1px solid var(--line);background:#0b0e0c;padding:24px;overflow:auto}pre{margin:0;color:#d9e5d2;font-size:13px;line-height:1.7}.token{color:var(--coral)}.copy{position:absolute;top:12px;right:12px;border:1px solid var(--line);color:var(--lime);background:transparent;padding:7px 10px;cursor:pointer;font:11px var(--mono)}footer{padding:22px 0 42px;color:var(--muted);border-top:1px solid var(--line);font:12px var(--mono)}
@media(max-width:720px){.wrap{width:min(100% - 28px,560px)}header{align-items:flex-start;gap:18px}nav{gap:10px;flex-wrap:wrap;justify-content:flex-end}.hero{padding:68px 0 58px;display:block}h1{font-size:clamp(50px,16vw,78px)}.hero-aside{margin-top:42px}.grid{grid-template-columns:1fr}section{padding-bottom:60px}}
</style></head>
<body><div class="wrap">
<header><a class="brand" href="/"><b>g4f</b> / api</a><nav><a href="/v1/models">models</a><a href="/docs">swagger</a><a href="/redoc">redoc</a></nav></header>
<main><div class="hero"><div><div class="eyebrow">Unified inference gateway · v1</div><h1>One endpoint.<br>Many minds.</h1><p class="lede">Build with chat, reasoning, vision, image, audio, and Anthropic-compatible interfaces through a single g4f API.</p><div class="actions"><a class="button" href="/docs">Open API reference</a><a class="button alt" href="/v1/models">Browse models</a></div></div><aside class="hero-aside"><strong>BASE URL</strong><code>/v1</code><p>Drop it into an OpenAI client, keep your existing request shape, and switch models without rewriting your application.</p></aside></div>
<section><div class="section-head"><h2>Interfaces</h2><span>three ways in</span></div><div class="grid">
<a class="card" href="/docs"><h3>Chat completions</h3><p>OpenAI-compatible messages with streaming, tools, vision, and model selection.</p><span class="endpoint">POST /v1/chat/completions</span></a>
<a class="card" href="/docs"><h3>Responses</h3><p>A modern response interface for multi-turn reasoning and richer output types.</p><span class="endpoint">POST /v1/responses</span></a>
<a class="card" href="/docs"><h3>Messages</h3><p>Anthropic-style requests for teams moving between compatible providers.</p><span class="endpoint">POST /v1/messages</span></a>
<a class="card" href="/docs"><h3>Media generation</h3><p>Generate images and other supported media from the same API surface.</p><span class="endpoint">POST /v1/media/generate</span></a>
<a class="card" href="/v1/models"><h3>Model catalog</h3><p>Inspect the models and provider options available on this server right now.</p><span class="endpoint">GET /v1/models</span></a>
<a class="card" href="/docs"><h3>Live schema</h3><p>Try requests in your browser and inspect the generated OpenAPI contract.</p><span class="endpoint">GET /docs</span></a>
</div></section><section><div class="section-head"><h2>Start here</h2><span>curl · JSON · streamable</span></div><div class="code-box"><button class="copy" onclick="copyExample(this)">copy</button><pre id="example">curl -X POST /v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'</pre></div></section></main>
<footer>g4f API · <a href="/openapi.json">openapi.json</a> · Compatible clients welcome.</footer></div>
<script>function copyExample(button){navigator.clipboard?.writeText(document.getElementById("example").textContent).then(()=>{button.textContent="copied";setTimeout(()=>button.textContent="copy",1400)})}</script></body></html>"""

# ---------------------------------------------------------------------------
# Request / response log store
# ---------------------------------------------------------------------------

_MAX_LOG_ENTRIES = 250
_MAX_BODY_LOG_SIZE = 1024 * 1024  # 1 MB
_SENSITIVE_HEADERS = {
    "authorization",
    "g4f-api-key",
    "cookie",
    "set-cookie",
    "x-api-key",
}

_request_log: deque = deque(maxlen=_MAX_LOG_ENTRIES)
_log_id_counter: int = 0


def _sanitize_headers(headers: dict) -> dict:
    return {
        k: ("***" if k.lower() in _SENSITIVE_HEADERS else v) for k, v in headers.items()
    }


def _try_parse_body(body_bytes: bytes, content_type: str):
    if not body_bytes:
        return None
    if len(body_bytes) > _MAX_BODY_LOG_SIZE:
        return f"<{len(body_bytes)} bytes – truncated>"
    if "application/json" in content_type:
        try:
            return json.loads(body_bytes)
        except Exception:
            pass
    try:
        return body_bytes.decode("utf-8", errors="replace")
    except Exception:
        return f"<binary {len(body_bytes)} bytes>"


_LOGS_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>g4f – Request Log</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh;font-size:14px}
a{color:#58a6ff}
.header{padding:14px 20px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:14px}
.header h1{font-size:15px;font-weight:600;color:#f0f6fc}
.header .sub{font-size:12px;color:#8b949e}
.toolbar{padding:10px 20px;border-bottom:1px solid #21262d;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.toolbar input[type=text]{flex:1;min-width:180px;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:5px 10px;border-radius:6px;font-size:13px;outline:none}
.toolbar input[type=text]:focus{border-color:#58a6ff}
.toolbar label{display:flex;align-items:center;gap:5px;font-size:13px;color:#8b949e;cursor:pointer;user-select:none}
.btn{padding:5px 14px;border-radius:6px;border:1px solid #30363d;cursor:pointer;font-size:13px;background:#21262d;color:#c9d1d9}
.btn:hover{background:#30363d}
.btn-danger{border-color:#6e3435;background:#1c1214;color:#ffa198}
.btn-danger:hover{background:#6e3435}
.meta{margin-left:auto;font-size:12px;color:#6e7681}
.table-wrap{overflow-x:auto;padding:0 20px 40px}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px}
th{padding:6px 8px;text-align:left;color:#8b949e;font-weight:500;border-bottom:1px solid #21262d;white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid #161b22;white-space:nowrap;max-width:320px;overflow:hidden;text-overflow:ellipsis;vertical-align:middle}
tbody tr{cursor:pointer}
tbody tr:hover td{background:#161b22}
.GET{color:#3fb950}.POST{color:#58a6ff}.PUT{color:#e3b341}.DELETE{color:#f85149}.PATCH{color:#d2a8ff}
.s2{color:#3fb950}.s3{color:#58a6ff}.s4{color:#e3b341}.s5{color:#f85149}
.tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:10px;font-weight:500}
.tag-sse{background:#0d2044;color:#79c0ff}
.tag-body{background:#0d2820;color:#56d364}
.tag-empty{background:#1c2128;color:#6e7681}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:200;padding:24px;align-items:flex-start;justify-content:center;overflow-y:auto}
.overlay.active{display:flex}
.modal{background:#161b22;border:1px solid #30363d;border-radius:10px;width:100%;max-width:1040px;display:flex;flex-direction:column}
.modal-head{padding:14px 18px;border-bottom:1px solid #21262d;display:flex;justify-content:space-between;align-items:center;gap:10px}
.modal-head h2{font-size:13px;font-weight:600;color:#f0f6fc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,monospace}
.modal-close{background:none;border:none;color:#6e7681;font-size:18px;cursor:pointer;line-height:1;padding:2px 6px;flex-shrink:0}
.modal-close:hover{color:#c9d1d9}
.modal-grid{display:grid;grid-template-columns:1fr 1fr 1fr}
.panel{padding:16px 18px;display:flex;flex-direction:column;gap:8px}
.panel:first-child{border-right:1px solid #21262d}
.panel:not(:last-child){border-right:1px solid #21262d}
.panel-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#6e7681}
.panel pre{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:12px;font-size:12px;line-height:1.5;overflow:auto;max-height:440px;white-space:pre-wrap;word-break:break-all;color:#c9d1d9;margin:0;font-family:ui-monospace,monospace}
@media(max-width:640px){.modal-grid{grid-template-columns:1fr}.panel:first-child{border-right:none;border-bottom:1px solid #21262d}.panel:not(:last-child){border-right:none;border-bottom:1px solid #21262d}}
</style>
</head>
<body>
<div class="header">
  <h1>g4f Request Log</h1>
  <span class="sub">last 500 entries &middot; <a href="/v1">/v1 API</a></span>
</div>
<div class="toolbar">
  <input type="text" id="q" placeholder="Filter by path, method, status, user&hellip;" oninput="render()">
  <label><input type="checkbox" id="auto" checked onchange="toggleAuto()"> Auto&#8209;refresh&nbsp;(3s)</label>
  <button class="btn btn-danger" onclick="clearLogs()">Clear</button>
  <span class="meta" id="meta"></span>
</div>
<div class="table-wrap"><table>
  <thead><tr>
    <th>#</th><th>Time (UTC)</th><th>Method</th><th>Path</th>
    <th>Status</th><th>ms</th><th>User</th><th>Body</th>
  </tr></thead>
  <tbody id="tb"></tbody>
</table></div>
<div class="overlay" id="ov" onclick="overlayClick(event)">
  <div class="modal" id="mod">
    <div class="modal-head">
      <h2 id="mtitle">&ndash;</h2>
      <button class="modal-close" onclick="closeModal()">&#x2715;</button>
    </div>
    <div class="modal-grid">
      <div class="panel"><div class="panel-title">Request</div><pre id="preq"></pre></div>
      <div class="panel"><div class="panel-title">Response</div><pre id="pres"></pre></div>
      <div class="panel"><div class="panel-title">Response Headers</div><pre id="presh"></pre></div>
    </div>
  </div>
</div>
<script>
'use strict';
var all=[], timer=null;
function sc(s){return s>=500?'s5':s>=400?'s4':s>=300?'s3':'s2';}
function btag(e){if(e.streaming&&e.response_body==null)return\'<span class="tag tag-sse">SSE&hellip;</span>\';if(e.response_body!=null)return\'<span class="tag tag-body">\'+(e.streaming?\'SSE\':\'body\')+\'</span>\';return\'<span class="tag tag-empty">&ndash;</span>\';}
function esc(s){return String(s??\'\'). replace(/&/g,\'&amp;\').replace(/</g,\'&lt;\').replace(/>/g,\'&gt;\');}
function fmt(v){if(v==null)return\'(empty)\';if(typeof v===\'object\')return JSON.stringify(v,null,2);return String(v);}
async function load(){
  try{var r=await fetch(\'/api/logs?limit=500\', { credentials: 'include' });if(!r.ok)return;var d=await r.json();all=d.entries||[];render();}catch(e){}
  clearTimeout(timer);timer = setTimeout(load,3000);
}
function render(){
  var q=document.getElementById(\'q\').value.trim().toLowerCase();
  var rows=q?all.filter(function(e){return(e.method+\' \'+e.path+\' \'+e.status+\' \'+(e.user||\'\')).toLowerCase().includes(q);}):all;
  document.getElementById(\'meta\').textContent=rows.length+\' / \'+all.length+\' entries\';
  document.getElementById(\'tb\').innerHTML=rows.map(function(e){
    var t=(e.timestamp||\'\').replace(\'T\',\' \').replace(\'Z\',\'\');
    var p=esc(e.path+(e.query?\'?\'+e.query:\'\'));
    return\'<tr onclick="detail(\'+e.id+\')">\'+
      \'<td style="color:#484f58">\'+e.id+\'</td>\'+
      \'<td style="color:#6e7681;font-size:12px">\'+esc(t)+\'</td>\'+
      \'<td class="\'+esc(e.method)+\'">\'+esc(e.method)+\'</td>\'+
      \'<td title="\'+p+\'">\'+p+\'</td>\'+
      \'<td class="\'+sc(e.status)+\'">\'+e.status+\'</td>\'+
      \'<td style="color:#8b949e">\'+e.duration_ms+\'</td>\'+
      \'<td style="color:#6e7681">\'+esc(e.user||\'\')+\'</td>\'+
      \'<td>\'+btag(e)+\'</td>\'+
      \'</tr>\';
  }).join(\'\');
}
function detail(id){
  var e=all.find(function(x){return x.id===id;});
  if(!e)return;
  document.getElementById(\'mtitle\').textContent=\'#\'+e.id+\'  \'+e.method+\' \'+e.path+(e.query?\'?\'+e.query:\'\')+\' \u2192 \'+e.status+\'  (\'+e.duration_ms+\'ms)\';
  var req=\'\';
  if(e.request_headers){req+=\'Headers:\\n\';for(var k in e.request_headers)req+=\'  \'+k+\': \'+e.request_headers[k]+\'\\n\';}
  if(e.request_body!=null)req+=\'\\nBody:\\n\'+fmt(e.request_body);
  document.getElementById(\'preq\').textContent=req||\'(none)\';
  document.getElementById(\'pres\').textContent=e.response_body!=null?fmt(e.response_body):(e.streaming?\'(streaming – collecting…)\':\'(empty)\');
  var resh=\'\';
  if(e.response_headers){resh+=\'Headers:\\n\';for(var k in e.response_headers)resh+=\'  \'+k+\': \'+e.response_headers[k]+\'\\n\';}
  document.getElementById(\'presh\').textContent=resh||\'(none)\';
  document.getElementById(\'ov\').classList.add(\'active\');
}
function closeModal(){document.getElementById(\'ov\').classList.remove(\'active\');}
function overlayClick(ev){if(ev.target===document.getElementById(\'ov\'))closeModal();}
document.addEventListener(\'keydown\',function(e){if(e.key===\'Escape\')closeModal();});
async function clearLogs(){await fetch(\'/api/logs\',{method:\'DELETE\'});all=[];render();}
function toggleAuto(){clearTimeout(timer);timer=null;if(document.getElementById(\'auto\').checked)timer=setTimeout(load,3000);}
load();if(document.getElementById(\'auto\').checked)timer=setTimeout(load,3000);
</script>
</body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read cookie files if not ignored
    if not AppConfig.ignore_cookie_files:
        read_cookie_files()
    else:
        AppConfig.load_from_env()
    yield
    if has_cdp:
        from g4f.requests.cdp import _terminate_shared_browser
        _terminate_shared_browser()


_LOG_SKIP_PREFIXES = ("/images/", "/media/", "/thumbnail/", "/dist/", "/.well-known/")
_LOG_SKIP_EXACT = {"/api/logs", "/logs", "/favicon.ico"}


def create_app():
    app = FastAPI(lifespan=lifespan)

    env_origins = [
        o.strip()
        for o in os.environ.get("G4F_CORS_ORIGINS", "").split(",")
        if o.strip()
    ]
    if env_origins:
        cors_origins = env_origins
        cors_regex = None
    else:
        cors_origins = [
            "https://g4f.dev",
            "https://g4f.space",
        ]
        cors_regex = r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:[0-9]+)?$"

    # When allow_credentials=True, the "*" wildcard is not permitted by the
    # CORS spec for allow_headers / expose_headers — browsers reject the
    # preflight and block the actual request.  List the headers explicitly.
    _cors_allow_headers = [
        "Accept",
        "Accept-Language",
        "Authorization",
        "Content-Type",
        "Content-Length",
        "Origin",
        "User-Agent",
        "X-Requested-With",
        "x-user-id",
        "x-user",
        "x-api-key",
        "x-session-id",
        "x-requested-with",
    ]
    _cors_expose_headers = [
        "Content-Length",
        "Content-Type",
        "Content-Disposition",
        "X-Request-ID",
    ]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=_cors_allow_headers,
        expose_headers=_cors_expose_headers,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        global _log_id_counter
        path = request.url.path
        if (
            any(path.startswith(p) for p in _LOG_SKIP_PREFIXES)
            or path in _LOG_SKIP_EXACT
        ):
            return await call_next(request)

        qs = f"?{request.url.query}" if request.url.query else ""
        user = request.headers.get("x-user", "")
        user_info = f" user={user}" if user else ""
        logger.debug("→ %s %s%s%s", request.method, path, qs, user_info)

        # Capture request body (Starlette caches after first read)
        req_body_bytes = await request.body()
        req_body = _try_parse_body(
            req_body_bytes, request.headers.get("content-type", "")
        )

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000)

        resp_content_type = response.headers.get("content-type", "")
        is_streaming = "text/event-stream" in resp_content_type
        log_entry: dict = {}

        resp_headers_log = _sanitize_headers(dict(response.headers))
        if not is_streaming:
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            resp_body_bytes = b"".join(chunks)
            resp_body = _try_parse_body(resp_body_bytes, resp_content_type)
            # Reconstruct response so it can still be sent to the client
            resp_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() != "content-length"
            }
            response = Response(
                content=resp_body_bytes,
                status_code=response.status_code,
                headers=resp_headers,
                media_type=response.media_type,
            )
        else:
            # Tee the streaming iterator: forward chunks to client AND accumulate for log
            sse_chunks: list[bytes] = []
            resp_body = None
            orig_iterator = response.body_iterator

            async def tee_iterator():
                async for chunk in orig_iterator:
                    if isinstance(chunk, bytes):
                        sse_chunks.append(chunk)
                    else:
                        sse_chunks.append(chunk.encode("utf-8", errors="replace"))
                    yield chunk
                # After iteration completes, parse and store the full SSE body
                raw = b"".join(sse_chunks)
                parsed = _try_parse_body(raw, "text/plain")
                log_entry["response_body"] = parsed

            response.body_iterator = tee_iterator()

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "%s %s%s → %d (%dms)%s",
            request.method,
            path,
            qs,
            response.status_code,
            duration_ms,
            user_info,
        )

        _log_id_counter += 1
        log_entry.update(
            {
                "id": _log_id_counter,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": request.method,
                "path": path,
                "query": request.url.query or None,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "user": user or None,
                "streaming": is_streaming,
                "request_headers": _sanitize_headers(dict(request.headers)),
                "request_body": req_body,
                "response_headers": resp_headers_log,
                "response_body": resp_body,  # None for SSE until iterator completes
            }
        )
        _request_log.append(log_entry)

        return response

    api = Api(app)

    api.register_routes()
    api.register_authorization()
    api.register_validation_exception_handler()

    if AppConfig.gui:
        if not has_a2wsgi:
            raise MissingRequirementsError(
                "a2wsgi is required for GUI. Install it with: pip install a2wsgi"
            )
        gui_app = WSGIMiddleware(
            get_gui_app(AppConfig.demo, AppConfig.timeout, AppConfig.stream_timeout)
        )
        app.mount("/", gui_app)

    if AppConfig.ignored_providers:
        for provider in AppConfig.ignored_providers:
            if provider in ProviderUtils.convert:
                ProviderUtils.convert[provider].working = False

    return app


def create_app_debug():
    g4f.debug.logging = True
    return create_app()


def create_app_with_gui_and_debug():
    g4f.debug.logging = True
    AppConfig.gui = True
    return create_app()


def create_app_with_demo_and_debug():
    g4f.debug.logging = True
    AppConfig.gui = True
    AppConfig.demo = True
    return create_app()


class ErrorResponse(Response):
    media_type = "application/json"

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        config: Union[ChatCompletionsConfig, ImageGenerationConfig] = None,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        logger.exception(exception)
        if isinstance(exception, ModelNotFoundError):
            safe_message = "ModelNotFoundError: Model not found"
        elif isinstance(exception, ProviderNotFoundError):
            safe_message = "ProviderNotFoundError: Provider not found"
        elif isinstance(exception, MissingAuthError):
            safe_message = "MissingAuthError: Authentication required"
        else:
            safe_message = "Request execution failed"
        return cls(format_exception(safe_message, config), status_code)

    @classmethod
    def from_message(
        cls,
        message: str,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        headers: dict = None,
    ):
        if not isinstance(message, str):
            message = "An error occurred"
        return cls(format_exception(message), status_code, headers=headers)

    def render(self, content) -> bytes:
        return str(content).encode(errors="ignore")


def update_headers(
    request: Request, user: str = None
) -> Request:
    new_headers = request.headers.mutablecopy()
    if user:
        new_headers["x-user"] = user
    request.scope["headers"] = new_headers.raw
    delattr(request, "_headers")
    return request


class Api:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.client = AsyncClient()
        self.get_g4f_api_key = APIKeyHeader(name="g4f-api-key")
        self.conversations: dict[str, dict[str, BaseConversation]] = {}
        self._models_cache: dict | None = None
        self._models_cache_time: float = 0.0

    security = HTTPBearer(auto_error=False)
    basic_security = HTTPBasic()

    async def get_username(self, request: Request) -> str:
        credentials = await self.basic_security(request)
        current_password_bytes = credentials.password.encode()
        is_correct_password = secrets.compare_digest(
            current_password_bytes, AppConfig.g4f_api_key.encode()
        )
        if not is_correct_password:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username

    def register_authorization(self):
        if AppConfig.g4f_api_key:
            print(
                "Register authentication key:",
                "".join(["*" for _ in range(len(AppConfig.g4f_api_key))]),
            )
        if has_crypto:
            private_key, _ = create_or_read_keys()
            session_key = get_session_key()

        @self.app.middleware("http")
        async def authorization(request: Request, call_next):
            user = None
            if (
                request.method != "OPTIONS"
                and AppConfig.g4f_api_key is not None
            ):
                try:
                    user_g4f_api_key = await self.get_g4f_api_key(request)
                except HTTPException:
                    user_g4f_api_key = getattr(
                        await self.security(request), "credentials", None
                    )
                if user_g4f_api_key:
                    user_g4f_api_key = user_g4f_api_key.split()
                if (
                    AppConfig.g4f_api_key is None
                    or not user_g4f_api_key
                    or not secrets.compare_digest(
                        AppConfig.g4f_api_key, user_g4f_api_key[0]
                    )
                ):
                    try:
                        new_user = await self.get_username(request)
                        if user is None:
                            user = new_user
                    except HTTPException as e:
                        return ErrorResponse.from_message(
                            e.detail, e.status_code, e.headers
                        )
                request = update_headers(request, user)
            response = await call_next(request)
            return response

    def register_validation_exception_handler(self):
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            details = exc.errors()
            modified_details = []
            for error in details:
                debug.log(
                    f"Validation error: {error['loc']} - {error['msg']} ({error['type']})"
                )
                modified_details.append(
                    {
                        "loc": error["loc"],
                        "message": error["msg"],
                        "type": error["type"],
                    }
                )
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_CONTENT,
                content=jsonable_encoder({"detail": modified_details}),
            )

    def register_routes(self):
        if not AppConfig.gui:

            @self.app.get("/")
            async def read_root():
                return RedirectResponse("/v1", 302)

        @self.app.get("/v1")
        async def read_root_v1():
            return HTMLResponse(V1_LANDING_PAGE)

        @self.app.get(
            "/v1/models",
            responses={
                HTTP_200_OK: {"model": List[ModelResponseModel]},
            },
        )
        async def models():
            now = time.time()
            if self._models_cache is not None and (now - self._models_cache_time) < 300:
                return self._models_cache

            result = {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "",
                        "image": isinstance(model, g4f.models.ImageModel),
                        "vision": isinstance(model, g4f.models.VisionModel),
                        "provider": False,
                    }
                    for model in AnyProvider.get_models()
                ]
                + [
                    {
                        "id": provider_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": getattr(provider, "label", ""),
                        "image": bool(getattr(provider, "image_models", False)),
                        "vision": bool(getattr(provider, "vision_models", False)),
                        "provider": True,
                    }
                    for provider_name, provider in ProviderUtils.convert.items()
                    if provider.working
                ],
            }
            self._models_cache = result
            self._models_cache_time = now
            return result

        @self.app.get(
            "/api/{provider:path}/models",
            responses={
                HTTP_200_OK: {"model": List[ModelResponseModel]},
            },
        )
        async def models(
            provider: str,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)
            if not hasattr(provider, "get_models"):
                models = getattr(provider, "models", [])
            elif credentials is not None and credentials.credentials != "secret":
                models = provider.get_models(api_key=credentials.credentials)
            else:
                models = provider.get_models()
            if inspect.isawaitable(models):
                models = await models
            return {
                "object": "list",
                "data": [
                    {
                        "id": model.get("id") if isinstance(model, dict) else model,
                        "object": "model",
                        "created": 0,
                        "owned_by": getattr(provider, "label", provider.__name__),
                        "image": (model.get("id") if isinstance(model, dict) else model)
                        in getattr(provider, "image_models", []),
                        "vision": (
                            model.get("id") if isinstance(model, dict) else model
                        )
                        in getattr(provider, "vision_models", []),
                        "audio": (model.get("id") if isinstance(model, dict) else model)
                        in getattr(provider, "audio_models", []),
                        "video": (model.get("id") if isinstance(model, dict) else model)
                        in getattr(provider, "video_models", []),
                        "type": "image"
                        if (model.get("id") if isinstance(model, dict) else model)
                        in getattr(provider, "image_models", [])
                        else "chat",
                        "count": getattr(provider, "models_count", {}).get(
                            model.get("id") if isinstance(model, dict) else model, 0
                        ),
                        **(model if isinstance(model, dict) else {}),
                    }
                    for model in (
                        models.values() if isinstance(models, dict) else models
                    )
                ],
            }

        # quota endpoint mimics backend-api/v2/quota but exposed on public API
        @self.app.get("/api/{provider:path}/quota")
        async def provider_quota(
            provider: str,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)
            if not hasattr(provider, "get_quota"):
                return ErrorResponse.from_message(
                    "Provider doesn't support get_quota", HTTP_500_INTERNAL_SERVER_ERROR
                )
            try:
                if credentials is not None and credentials.credentials != "secret":
                    usage = await provider.get_quota(api_key=credentials.credentials)
                else:
                    usage = await provider.get_quota()
                return usage
            except MissingAuthError:
                return ErrorResponse.from_message(
                    "MissingAuthError: Authentication required", HTTP_401_UNAUTHORIZED
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_message(
                    "Failed to retrieve provider quota", HTTP_500_INTERNAL_SERVER_ERROR
                )

        @self.app.get(
            "/v1/models/{model_name}",
            responses={
                HTTP_200_OK: {"model": ModelResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        @self.app.post(
            "/v1/models/{model_name}",
            responses={
                HTTP_200_OK: {"model": ModelResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def model_info(model_name: str) -> ModelResponseModel:
            if model_name in g4f.models.ModelUtils.convert:
                model_info = g4f.models.ModelUtils.convert[model_name]
                return JSONResponse(
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": model_info.base_provider,
                    }
                )
            return ErrorResponse.from_message(
                "The model does not exist.", HTTP_404_NOT_FOUND
            )

        responses = {
            HTTP_200_OK: {"model": ChatCompletion},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponseModel},
            HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }

        @self.app.post("/v1/chat/completions", responses=responses)
        @self.app.post("/{mode}/{provider:path}/chat/completions",  responses=responses)
        @self.app.post(
            "/api/{provider}/{conversation_id}/chat/completions", responses=responses
        )
        async def chat_completions(
            config: ChatCompletionsConfig,
            request: Request = None,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
            mode: str | None = None,
            provider: str | None = None,
            conversation_id: str | None = None,
            x_user: Annotated[str | None, Header()] | None = None,
        ):
            if mode == "raw":
                config.raw = True
            elif mode == "custom":
                provider = f"{mode}:{provider}"
            if provider is not None:
                config.provider = provider
            if config.provider is None:
                config.provider = AppConfig.provider
            try:
                provider = AbstractClientFactory.create_provider(None, config.provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {config.provider}", 404)
            try:
                if config.conversation_id is None:
                    config.conversation_id = conversation_id
                if config.timeout is None:
                    config.timeout = AppConfig.timeout
                if config.stream_timeout is None and config.stream:
                    config.stream_timeout = AppConfig.stream_timeout
                if credentials is not None and credentials.credentials != "secret":
                    config.api_key = credentials.credentials

                conversation = config.conversation
                if conversation:
                    conversation = JsonConversation(**conversation)
                elif config.conversation_id is not None and config.provider is not None:
                    if config.conversation_id in self.conversations:
                        if (
                            config.provider
                            in self.conversations[config.conversation_id]
                        ):
                            conversation = self.conversations[config.conversation_id][
                                config.provider
                            ]

                if config.image is not None:
                    try:
                        is_data_an_media(config.image)
                    except ValueError as e:
                        return ErrorResponse.from_message(
                            f"The image you send must be a data URI. Example: data:image/jpeg;base64,...",
                            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
                        )
                if config.media is None:
                    config.media = config.images
                if config.media is not None:
                    for image in config.media:
                        try:
                            is_data_an_media(image[0], image[1])
                        except ValueError as e:
                            example = json.dumps(
                                {
                                    "media": [
                                        ["data:image/jpeg;base64,...", "filename.jpg"]
                                    ]
                                }
                            )
                            return ErrorResponse.from_message(
                                f"The media you send must be a data URIs. Example: {example}",
                                status_code=HTTP_422_UNPROCESSABLE_CONTENT,
                            )

                # Create the completion response
                response = self.client.chat.completions.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "provider": AppConfig.provider,
                            "proxy": AppConfig.proxy,
                            **(
                                config.model_dump(exclude_none=True)
                                if hasattr(config, "model_dump")
                                else config.dict(exclude_none=True)
                            ),
                            **{
                                "provider": provider,
                                "conversation_id": None,
                                "conversation": conversation,
                                "user": x_user,
                            },
                        },
                        ignored=AppConfig.ignored_providers,
                    ),
                )

                if not config.stream:
                    result = await response
                    return Response(
                        content=result.model_dump_json()
                        if hasattr(result, "model_dump_json")
                        else result.json(),
                        media_type="application/json",
                        headers=getattr(result, "_headers").get_dict()
                        if hasattr(result, "_headers")
                        else None,
                    )

                first_chunk = await response.__anext__()

                async def streaming():
                    yield f"data: {first_chunk.model_dump_json() if hasattr(first_chunk, 'model_dump_json') else first_chunk.json()}\n\n"
                    try:
                        async for chunk in response:
                            if request is not None and await request.is_disconnected():
                                debug.log("Client disconnected, aborting streaming response.")
                                return
                            if isinstance(chunk, BaseConversation):
                                if (
                                    config.conversation_id is not None
                                    and config.provider is not None
                                ):
                                    if config.conversation_id not in self.conversations:
                                        self.conversations[config.conversation_id] = {}
                                    self.conversations[config.conversation_id][
                                        config.provider
                                    ] = chunk
                            else:
                                yield f"data: {chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else chunk.json()}\n\n"
                    except GeneratorExit:
                        pass
                    except RateLimitError as e:
                        debug.error(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    except Exception as e:
                        logger.exception(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    finally:
                        if hasattr(response, "aclose"):
                            try:
                                await response.aclose()
                            except Exception:
                                pass
                    yield "data: [DONE]\n\n"

                headers = (
                    getattr(first_chunk, "_headers").get_dict()
                    if hasattr(first_chunk, "_headers")
                    else {}
                )
                headers = {
                    k.encode("latin-1", "ignore")
                    .decode("latin-1"): v.encode("latin-1", "ignore")
                    .decode("latin-1")
                    for k, v in headers.items()
                }
                return StreamingResponse(
                    streaming(), media_type="text/event-stream", headers=headers
                )
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except (MissingAuthError, NoValidHarFileError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except RateLimitError as e:
                return ErrorResponse.from_exception(
                    e, config, HTTP_429_TOO_MANY_REQUESTS
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, config, HTTP_500_INTERNAL_SERVER_ERROR
                )

        # ------------------------------------------------------------------ #
        # OpenAI Responses API  (/v1/responses)                               #
        # https://platform.openai.com/docs/api-reference/responses            #
        # ------------------------------------------------------------------ #
        @self.app.post("/v1/responses", responses=responses)
        @self.app.post("/api/{provider:path}/responses", responses=responses)
        async def create_response(
            config: ResponsesConfig,
            request: Request = None,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] | None = None,
            provider: str | None = None,
            x_user: Annotated[str | None, Header(alias="X-User")] | None = None,
        ):
            if provider is not None:
                config.provider = provider
            if config.provider is None:
                config.provider = AppConfig.provider
            try:
                provider = AbstractClientFactory.create_provider(None, config.provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {config.provider}", 404)
            try:
                if config.timeout is None:
                    config.timeout = AppConfig.timeout
                if config.stream_timeout is None and config.stream:
                    config.stream_timeout = AppConfig.stream_timeout
                if credentials is not None and credentials.credentials != "secret":
                    config.api_key = credentials.credentials

                # Normalize `input` into a messages list.
                messages = config.input
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                if config.instructions:
                    messages = [
                        {"role": "system", "content": config.instructions},
                        *messages,
                    ]

                response = self.client.chat.completions.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "provider": AppConfig.provider,
                            "proxy": AppConfig.proxy,
                            **(
                                config.model_dump(exclude_none=True)
                                if hasattr(config, "model_dump")
                                else config.dict(exclude_none=True)
                            ),
                            **{
                                "provider": provider,
                                "messages": messages,
                                "user": x_user,
                            },
                        },
                        ignored=AppConfig.ignored_providers,
                    ),
                )

                if not config.stream:
                    result = await response
                    text = result.choices[0].message.content if result.choices else ""
                    usage = getattr(result, "usage", None)
                    if usage is not None and hasattr(usage, "model_dump"):
                        usage = usage.model_dump()
                    elif usage is not None and hasattr(usage, "dict"):
                        usage = usage.dict()
                    return JSONResponse(
                        {
                            "id": getattr(
                                result, "id", f"resp_{secrets.token_hex(12)}"
                            ),
                            "object": "response",
                            "created_at": getattr(result, "created", int(time.time())),
                            "model": getattr(result, "model", config.model),
                            "provider": getattr(provider, "__name__", config.provider),
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": text}],
                                }
                            ],
                            "output_text": text,
                            "usage": usage,
                        }
                    )

                first_chunk = await response.__anext__()

                async def responses_streaming():
                    yield f"data: {first_chunk.model_dump_json() if hasattr(first_chunk, 'model_dump_json') else first_chunk.json()}\n\n"
                    try:
                        async for chunk in response:
                            if request is not None and await request.is_disconnected():
                                debug.log("Client disconnected, aborting responses stream.")
                                return
                            if isinstance(chunk, BaseConversation):
                                pass
                            else:
                                yield f"data: {chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else chunk.json()}\n\n"
                    except GeneratorExit:
                        pass
                    except RateLimitError as e:
                        debug.error(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    except Exception as e:
                        logger.exception(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    finally:
                        if hasattr(response, "aclose"):
                            try:
                                await response.aclose()
                            except Exception:
                                pass
                    yield "data: [DONE]\n\n"

                headers = (
                    getattr(first_chunk, "_headers").get_dict()
                    if hasattr(first_chunk, "_headers")
                    else {}
                )
                headers = {
                    k.encode("latin-1", "ignore")
                    .decode("latin-1"): v.encode("latin-1", "ignore")
                    .decode("latin-1")
                    for k, v in headers.items()
                }
                return StreamingResponse(
                    responses_streaming(),
                    media_type="text/event-stream",
                    headers=headers,
                )
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except (MissingAuthError, NoValidHarFileError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except RateLimitError as e:
                return ErrorResponse.from_exception(
                    e, config, HTTP_429_TOO_MANY_REQUESTS
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, config, HTTP_500_INTERNAL_SERVER_ERROR
                )

        # ------------------------------------------------------------------ #
        # Anthropic Messages API  (/v1/messages)                              #
        # https://docs.anthropic.com/en/api/messages                          #
        # ------------------------------------------------------------------ #
        @self.app.post("/v1/messages", responses=responses)
        @self.app.post("/api/{provider:path}/messages", responses=responses)
        async def create_message(
            config: MessagesConfig,
            request: Request = None,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
            provider: str = None,
            x_user: Annotated[str | None, Header()] = None,
        ):
            if provider is not None:
                config.provider = provider
            if config.provider is None:
                config.provider = AppConfig.provider
            try:
                provider = AbstractClientFactory.create_provider(None, config.provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {config.provider}", 404)
            try:
                if config.timeout is None:
                    config.timeout = AppConfig.timeout
                if config.stream_timeout is None and config.stream:
                    config.stream_timeout = AppConfig.stream_timeout
                if credentials is not None and credentials.credentials != "secret":
                    config.api_key = credentials.credentials

                # Anthropic uses a top-level `system` field; fold it into messages.
                messages = config.messages
                if config.system:
                    system_content = config.system
                    if isinstance(system_content, list):
                        system_content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in system_content
                        )
                    messages = [
                        {"role": "system", "content": system_content},
                        *messages,
                    ]

                response = self.client.chat.completions.create(
                    **filter_none(
                        **{
                            "model": AppConfig.model,
                            "provider": AppConfig.provider,
                            "proxy": AppConfig.proxy,
                            **(
                                config.model_dump(exclude_none=True)
                                if hasattr(config, "model_dump")
                                else config.dict(exclude_none=True)
                            ),
                            **{
                                "provider": provider,
                                "messages": messages,
                                "user": x_user,
                            },
                        },
                        ignored=AppConfig.ignored_providers,
                    ),
                )

                if not config.stream:
                    result = await response
                    text = result.choices[0].message.content if result.choices else ""
                    usage = getattr(result, "usage", None)
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                    return JSONResponse(
                        {
                            "id": getattr(result, "id", f"msg_{secrets.token_hex(12)}"),
                            "type": "message",
                            "role": "assistant",
                            "model": getattr(result, "model", config.model),
                            "provider": getattr(provider, "__name__", config.provider),
                            "content": [{"type": "text", "text": text}],
                            "stop_reason": getattr(
                                result.choices[0], "finish_reason", None
                            )
                            if result.choices
                            else None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            },
                        }
                    )

                first_chunk = await response.__anext__()

                async def messages_streaming():
                    yield f"data: {first_chunk.model_dump_json() if hasattr(first_chunk, 'model_dump_json') else first_chunk.json()}\n\n"
                    try:
                        async for chunk in response:
                            if request is not None and await request.is_disconnected():
                                debug.log("Client disconnected, aborting messages stream.")
                                return
                            if isinstance(chunk, BaseConversation):
                                pass
                            else:
                                yield f"data: {chunk.model_dump_json() if hasattr(chunk, 'model_dump_json') else chunk.json()}\n\n"
                    except GeneratorExit:
                        pass
                    except RateLimitError as e:
                        debug.error(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    except Exception as e:
                        logger.exception(e)
                        yield f"data: {format_exception(e, config)}\n\n"
                        return
                    finally:
                        if hasattr(response, "aclose"):
                            try:
                                await response.aclose()
                            except Exception:
                                pass
                    yield "data: [DONE]\n\n"

                headers = (
                    getattr(first_chunk, "_headers").get_dict()
                    if hasattr(first_chunk, "_headers")
                    else {}
                )
                headers = {
                    k.encode("latin-1", "ignore")
                    .decode("latin-1"): v.encode("latin-1", "ignore")
                    .decode("latin-1")
                    for k, v in headers.items()
                }
                return StreamingResponse(
                    messages_streaming(),
                    media_type="text/event-stream",
                    headers=headers,
                )
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except (MissingAuthError, NoValidHarFileError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except RateLimitError as e:
                return ErrorResponse.from_exception(
                    e, config, HTTP_429_TOO_MANY_REQUESTS
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, config, HTTP_500_INTERNAL_SERVER_ERROR
                )

        responses = {
            HTTP_200_OK: {"model": ImagesResponse},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }

        @self.app.post("/v1/media/generate", responses=responses)
        @self.app.post("/v1/images/generate", responses=responses)
        @self.app.post("/v1/images/generations", responses=responses)
        @self.app.post("/api/{provider:path}/images/generations", responses=responses)
        async def generate_image(
            request: Request,
            config: ImageGenerationConfig,
            provider: str = None,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            if provider is None:
                provider = config.provider
            if provider is None:
                provider = AppConfig.provider
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)
            if (
                config.api_key is None
                and credentials is not None
                and credentials.credentials != "secret"
            ):
                config.api_key = credentials.credentials
            try:
                response = await self.client.images.generate(
                    **config.model_dump(exclude_none=True)
                    if hasattr(config, "model_dump")
                    else config.dict(exclude_none=True),
                    provider=provider,
                )
                for image in response.data:
                    if hasattr(image, "url") and image.url.startswith("/"):
                        image.url = f"{request.base_url}{image.url.lstrip('/')}"
                return response
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, config, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, config, HTTP_500_INTERNAL_SERVER_ERROR
                )
    
        @self.app.get("/screenshot", responses=responses)
        async def image_from_url(
            url: str,
        ):
            try:
                from g4f.requests.cdp import CDPSession
                session = CDPSession()
                await session.start()
                try:
                    debug.log(f"Capturing screenshot for URL: {url}")
                    screenshot_path = await session.capture_screenshot(url, 1 if "q=" in url and "q=Hello" not in url else 3)
                    return FileResponse(
                        screenshot_path,
                        media_type="image/webp",
                        headers={"Cache-Control": "max-age=604800"},
                    )
                finally:
                    await session.close()
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        @self.app.get("/screenshot/{name:path}", responses=responses)
        async def image_from_url(
            name: str,
        ):
            safe_name = secure_filename(name)
            screenshots_dir = os.path.join(get_media_dir(), "screenshots")
            for root, _, files in os.walk(screenshots_dir):
                for file in files:
                    if file != safe_name:
                        continue
                    if not os.path.isfile(os.path.join(root, file)):
                        continue
                    return FileResponse(
                        os.path.join(root, safe_name),
                        media_type="image/webp",
                        headers={"Cache-Control": "max-age=604800"},
                    )
            return ErrorResponse.from_message("File not found", 404)

        @self.app.get(
            "/v1/providers",
            responses={
                HTTP_200_OK: {"model": List[ProviderResponseModel]},
            },
        )
        async def providers():
            return [
                {
                    "id": provider.__name__,
                    "object": "provider",
                    "created": 0,
                    "url": provider.url,
                    "label": getattr(provider, "label", None),
                }
                for provider in Provider.__providers__
                if provider.working
            ]

        @self.app.get(
            "/v1/providers/{provider}",
            responses={
                HTTP_200_OK: {"model": ProviderResponseDetailModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def providers_info(provider: str):
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)

            return {
                "id": provider.__name__,
                "object": "provider",
                "created": 0,
                "url": provider.url,
                "label": getattr(provider, "label", None),
                "image_models": getattr(provider, "image_models", []) or [],
                "vision_models": [
                    model
                    for model in [getattr(provider, "default_vision_model", None)]
                    if model
                ],
                "params": [*provider.get_parameters()]
                if hasattr(provider, "get_parameters")
                else [],
            }

        # ------------------------------------------------------------------ #
        # PA Provider routes                                                   #
        # ------------------------------------------------------------------ #

        @self.app.get(
            "/pa/providers",
            responses={
                HTTP_200_OK: {},
            },
        )
        async def pa_providers_list():
            """List all PA providers loaded from the workspace.

            Filenames are never exposed; each provider is identified by a
            stable opaque ID (SHA-256 of the path, first 8 hex chars).
            """
            from g4f.mcp.pa_provider import get_pa_registry

            return get_pa_registry().list_providers()

        @self.app.get(
            "/pa/providers/{provider_id}",
            responses={
                HTTP_200_OK: {},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def pa_providers_detail(provider_id: str):
            """Get details for a single PA provider by its opaque ID."""
            from g4f.mcp.pa_provider import get_pa_registry

            info = get_pa_registry().get_provider_info(provider_id)
            if info is None:
                return ErrorResponse.from_message(
                    f"PA provider '{provider_id}' not found", HTTP_404_NOT_FOUND
                )
            return info

        # ------------------------------------------------------------------ #
        # PA workspace static file serving (HTML/CSS/JS/images for browser)   #
        # ------------------------------------------------------------------ #

        #: MIME types that are safe to serve for browser rendering.
        #: Only these extensions are allowed; all others are refused with 403.
        _WORKSPACE_SAFE_TYPES: dict[str, str] = {
            "html": "text/html; charset=utf-8",
            "htm": "text/html; charset=utf-8",
            "css": "text/css; charset=utf-8",
            "js": "application/javascript; charset=utf-8",
            "mjs": "application/javascript; charset=utf-8",
            "json": "application/json; charset=utf-8",
            "txt": "text/plain; charset=utf-8",
            "md": "text/markdown; charset=utf-8",
            "svg": "image/svg+xml",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "ico": "image/x-icon",
            "woff": "font/woff",
            "woff2": "font/woff2",
            "ttf": "font/ttf",
            "otf": "font/otf",
            "py": "text/plain; charset=utf-8",
        }

        @self.app.get(
            "/pa/files/{file_path:path}",
            responses={
                HTTP_200_OK: {},
                HTTP_403_FORBIDDEN: {"model": ErrorResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def pa_serve_workspace_file(file_path: str, request: Request):
            """Securely serve a workspace file for browser rendering.

            Only files within ``~/.g4f/workspace`` can be served.  Path
            traversal (``..``) is blocked.  Only the MIME types listed in
            ``_WORKSPACE_SAFE_TYPES`` are served; all other extensions are
            refused with **403 Forbidden** so that sensitive file types (e.g.
            ``.env``, ``.pa.py``, ``.py``) can never be read via this route.

            HTML files are served with a ``Content-Security-Policy: sandbox``
            directive (without ``allow-same-origin``), which forces the page
            into a unique *null* browser origin.  As a result the page cannot
            access ``localStorage``, ``sessionStorage``, ``IndexedDB``, or
            cookies belonging to the g4f server origin — the browser rejects
            all such calls with a ``SecurityError``.  The actual request
            origin (``scheme://host``) is used in every source directive (e.g.
            ``default-src``) instead of ``'self'``, so that co-located CSS,
            JS, images, and fonts still load correctly despite the document
            having a null origin.

            Non-HTML sub-resources (CSS, JS, images, fonts) are served without
            the ``sandbox`` directive; they are leaf resources and do not run
            in their own browsing context.
            """
            from g4f.mcp.pa_provider import resolve_workspace_path

            # Extract user_id and workspace_secret from headers if available
            user_id = request.headers.get("x-user-id", "")
            workspace_secret = request.headers.get("x-workspace-secret", "")

            # Normalise and check for traversal
            try:
                resolved, workspace = resolve_workspace_path(
                    file_path, user_id=user_id, workspace_secret=workspace_secret, for_write=False
                )
                resolved.relative_to(workspace)
            except (ValueError, Exception):
                return ErrorResponse.from_message(
                    "Path traversal is not allowed", HTTP_403_FORBIDDEN
                )

            if not resolved.exists() or not resolved.is_file():
                return ErrorResponse.from_message(
                    f"File not found: {file_path}", HTTP_404_NOT_FOUND
                )

            ext = resolved.suffix.lstrip(".").lower()
            mime_type = _WORKSPACE_SAFE_TYPES.get(ext)
            if mime_type is None:
                return ErrorResponse.from_message(
                    f"File type '.{ext}' is not allowed for browser rendering",
                    HTTP_403_FORBIDDEN,
                )

            # Derive the actual request origin (scheme + authority) from the
            # ASGI scope via request.url — this is set by the server
            # infrastructure and is not controllable by the client (unlike the
            # Host header, which can be spoofed to inject arbitrary values into
            # the CSP).  request.url.netloc includes the port when non-default.
            request_origin = f"{request.url.scheme}://{request.url.netloc}"

            is_html = ext in ("html", "htm")
            if is_html:
                # HTML documents are served with the CSP sandbox directive
                # (without allow-same-origin).  This forces the page into a
                # unique null browsing-context origin so that it cannot access
                # the g4f server's localStorage, sessionStorage, IndexedDB, or
                # cookies.  The page can still load sub-resources (CSS, JS,
                # images) because they are referenced by the explicit
                # request_origin in the source directives.
                csp = (
                    "sandbox allow-scripts allow-forms allow-downloads allow-popups; "
                    f"default-src {request_origin} https://g4f.space; "
                    f"script-src {request_origin} 'unsafe-inline'; "
                    f"style-src {request_origin} 'unsafe-inline'; "
                    f"img-src {request_origin} data:; "
                    f"font-src {request_origin} data:; "
                    "connect-src 'none'; "
                    "object-src 'none'; "
                    "base-uri 'none';"
                )
            else:
                # Non-HTML sub-resources (CSS, JS, images, fonts) don't need
                # sandboxing — they are leaf assets without their own browsing
                # context.  Use the request origin for source directives.
                csp = (
                    f"default-src {request_origin} https://g4f.space; "
                    f"script-src {request_origin} 'unsafe-inline'; "
                    f"style-src {request_origin} 'unsafe-inline'; "
                    f"img-src {request_origin} data:; "
                    f"font-src {request_origin} data:; "
                    "connect-src 'none'; "
                    "object-src 'none'; "
                    "base-uri 'none';"
                )

            headers = {
                # Prevent the browser from sniffing a different content-type
                "X-Content-Type-Options": "nosniff",
                # Prevent this page from being framed by untrusted origins
                "X-Frame-Options": "SAMEORIGIN",
                # Basic XSS filter (belt-and-suspenders; CSP is more important)
                "X-XSS-Protection": "1; mode=block",
                "Content-Security-Policy": csp,
                # Restrict powerful browser APIs that workspace pages don't need
                "Permissions-Policy": (
                    "geolocation=(), camera=(), microphone=(), "
                    "payment=(), usb=(), fullscreen=()"
                ),
                "Cache-Control": "no-store",
            }

            return FileResponse(
                str(resolved),
                media_type=mime_type,
                headers=headers,
            )

        # ------------------------------------------------------------------ #
        # Secret conversation endpoints (per-user, stored in secret workspace) #
        # ------------------------------------------------------------------ #

        @self.app.get(
            "/v1/secret/conversations",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def list_secret_conversations_endpoint(request: Request):
            """List all secret conversations for the authenticated user."""
            from g4f.mcp.pa_provider import list_secret_conversations

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            return {"conversations": list_secret_conversations(user_id)}

        @self.app.post(
            "/v1/secret/conversations",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def save_secret_conversation_endpoint(request: Request):
            """Save a conversation to the user's secret workspace."""
            from g4f.mcp.pa_provider import save_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            try:
                body = await request.json()
            except Exception:
                return ErrorResponse.from_message("Invalid JSON body")
            result = save_secret_conversation(user_id, body, workspace_secret or None)
            return result

        @self.app.post(
            "/v1/secret/conversations/sync",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def sync_secret_conversations_endpoint(request: Request):
            """Sync (upload) multiple conversations to the user's secret workspace."""
            from g4f.mcp.pa_provider import save_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            try:
                body = await request.json()
            except Exception:
                return ErrorResponse.from_message("Invalid JSON body")
            conversations = body.get("conversations", []) if isinstance(body, dict) else body
            saved = 0
            errors = []
            for conv in conversations:
                res = save_secret_conversation(user_id, conv, workspace_secret or None)
                if res.get("saved"):
                    saved += 1
                else:
                    errors.append(res.get("error", "Unknown error"))
            return {"saved": saved, "errors": errors}

        @self.app.get(
            "/v1/secret/conversations/{conversation_id}",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def get_secret_conversation_endpoint(
            conversation_id: str, request: Request
        ):
            """Retrieve a single secret conversation by ID."""
            from g4f.mcp.pa_provider import get_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            workspace_secret = request.headers.get("x-workspace-secret", "")
            conv = get_secret_conversation(user_id, conversation_id, workspace_secret or None)
            if conv is None:
                return ErrorResponse.from_message(
                    f"Conversation '{conversation_id}' not found",
                    HTTP_404_NOT_FOUND,
                )
            return conv

        @self.app.delete(
            "/v1/secret/conversations/{conversation_id}",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def delete_secret_conversation_endpoint(
            conversation_id: str, request: Request
        ):
            """Delete a secret conversation by ID."""
            from g4f.mcp.pa_provider import delete_secret_conversation

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            deleted = delete_secret_conversation(user_id, conversation_id)
            if not deleted:
                return ErrorResponse.from_message(
                    f"Conversation '{conversation_id}' not found",
                    HTTP_404_NOT_FOUND,
                )
            return {"deleted": True, "id": conversation_id}

        # ── Cross-device workspace secret sharing ───────────────────────────

        @self.app.post(
            "/v1/secret/request",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def create_secret_request_endpoint(request: Request):
            """Create a pending secret-sharing request from a new device."""
            from g4f.mcp.pa_provider import create_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass
            device_name = body.get("device_name", "") if isinstance(body, dict) else ""
            return create_secret_request(user_id, device_name)

        @self.app.get(
            "/v1/secret/requests",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def list_secret_requests_endpoint(request: Request):
            """List pending secret-sharing requests for the online device."""
            from g4f.mcp.pa_provider import list_secret_requests

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            return {"requests": list_secret_requests(user_id)}

        @self.app.post(
            "/v1/secret/request/confirm",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def confirm_secret_request_endpoint(request: Request):
            """Confirm a secret request by sending the workspace secret."""
            from g4f.mcp.pa_provider import confirm_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            try:
                body = await request.json()
            except Exception:
                return ErrorResponse.from_message("Invalid JSON body", HTTP_400_BAD_REQUEST)
            request_id = body.get("request_id", "")
            workspace_secret = body.get("workspace_secret", "")
            if not request_id or not workspace_secret:
                return ErrorResponse.from_message(
                    "request_id and workspace_secret are required",
                    HTTP_400_BAD_REQUEST,
                )
            result = confirm_secret_request(user_id, request_id, workspace_secret)
            if "error" in result:
                return ErrorResponse.from_message(result["error"], HTTP_404_NOT_FOUND)
            return result

        @self.app.get(
            "/v1/secret/request/{request_id}",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            },
        )
        async def poll_secret_request_endpoint(
            request_id: str, request: Request
        ):
            """Poll a secret request to check if it has been confirmed."""
            from g4f.mcp.pa_provider import poll_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            return poll_secret_request(user_id, request_id)

        @self.app.delete(
            "/v1/secret/request/{request_id}",
            responses={
                HTTP_200_OK: {},
                HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
                HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            },
        )
        async def delete_secret_request_endpoint(
            request_id: str, request: Request
        ):
            """Cancel / delete a secret-sharing request."""
            from g4f.mcp.pa_provider import delete_secret_request

            user_id = request.headers.get("x-user-id", "")
            if not user_id:
                return ErrorResponse.from_message(
                    "User ID is required (provide x-user-id header)",
                    HTTP_401_UNAUTHORIZED,
                )
            deleted = delete_secret_request(user_id, request_id)
            if not deleted:
                return ErrorResponse.from_message(
                    "Request not found", HTTP_404_NOT_FOUND
                )
            return {"deleted": True, "request_id": request_id}

        responses = {
            HTTP_200_OK: {"model": TranscriptionResponseModel},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }

        @self.app.post("/v1/audio/transcriptions", responses=responses)
        @self.app.post(
            "/api/{path_provider:path}/audio/transcriptions", responses=responses
        )
        @self.app.post("/api/markitdown", responses=responses)
        async def convert(
            file: UploadFile,
            path_provider: Optional[str] = None,
            model: Annotated[Optional[str], Form()] = None,
            provider: Annotated[Optional[str], Form()] = None,
            prompt: Annotated[Optional[str], Form()] = "Transcribe this audio",
        ):
            if path_provider is not None:
                provider = path_provider
            if provider is None:
                provider = "MarkItDown"
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)
            kwargs = {"modalities": ["text"]}
            try:
                response = await self.client.chat.completions.create(
                    messages=prompt,
                    model=model,
                    provider=provider,
                    media=[[file.file, file.filename]],
                    **kwargs,
                )
                return {
                    "text": response.choices[0].message.content,
                    "model": response.model,
                    "provider": response.provider,
                }
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )

        responses = {
            HTTP_200_OK: {"model": TranscriptionResponseModel},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }


        @self.app.get("/text/{url:path}", responses={
            HTTP_200_OK: {"content": {"text/plain": {}}},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        })
        async def convert_url(
            request: Request,
            url: str
        ):
            """Convert a URL to Markdown using MarkItDown.

            The full URL (including scheme) is passed in the path, e.g.:
            GET /markitdown/https://example.com/page

            Query strings are preserved by reading them from the incoming
            request and re-appending them to the target URL, e.g.:
            GET /markitdown/https://example.com/page?foo=bar
            """
            # FastAPI strips the query string from the {url:path} parameter,
            # so re-attach it from the incoming request when present.
            query_string = request.url.query
            if query_string and "?" not in url:
                url = f"{url}?{query_string}"
            elif query_string:
                # url already contains a '?', append remaining params with '&'
                url = f"{url}&{query_string}"
            if not url.startswith(("http://", "https://")):
                return ErrorResponse.from_message(
                    f"Invalid URL: {url}. URL must start with http:// or https://",
                    HTTP_422_UNPROCESSABLE_CONTENT,
                )
            try:
                from g4f.integration.markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert_url(url)
                text = result.text_content
                if asyncio.iscoroutine(text):
                    text = await text
                return Response(text, media_type="text/plain")
            except ImportError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )

        @self.app.get("/markitdown/{url:path}", responses=responses)
        async def convert_url(
            request: Request,
            url: str,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            """Convert a URL to Markdown using MarkItDown.

            The full URL (including scheme) is passed in the path, e.g.:
            GET /markitdown/https://example.com/page

            Query strings are preserved by reading them from the incoming
            request and re-appending them to the target URL, e.g.:
            GET /markitdown/https://example.com/page?foo=bar
            """
            # FastAPI strips the query string from the {url:path} parameter,
            # so re-attach it from the incoming request when present.
            query_string = request.url.query
            if query_string and "?" not in url:
                url = f"{url}?{query_string}"
            elif query_string:
                # url already contains a '?', append remaining params with '&'
                url = f"{url}&{query_string}"
            if not url.startswith(("http://", "https://")):
                return ErrorResponse.from_message(
                    f"Invalid URL: {url}. URL must start with http:// or https://",
                    HTTP_422_UNPROCESSABLE_CONTENT,
                )
            try:
                from g4f.integration.markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert_url(url)
                text = result.text_content
                if asyncio.iscoroutine(text):
                    text = await text
                return JSONResponse(
                    {"text": text, "title": result.title, "url": url},
                )
            except ImportError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )

        responses = {
            HTTP_200_OK: {"content": {"audio/*": {}}},
            HTTP_401_UNAUTHORIZED: {"model": ErrorResponseModel},
            HTTP_404_NOT_FOUND: {"model": ErrorResponseModel},
            HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponseModel},
        }

        @self.app.post("/v1/audio/speech", responses=responses)
        @self.app.post("/api/{provider:path}/audio/speech", responses=responses)
        async def generate_speech(
            config: AudioSpeechConfig,
            provider: Optional[str] = None,
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            api_key = None
            if credentials is not None and credentials.credentials != "secret":
                api_key = credentials.credentials
            if provider is None:
                provider = config.provider
            if provider is None:
                provider = AppConfig.media_provider
            try:
                provider = AbstractClientFactory.create_provider(None, provider)
            except ProviderNotFoundError:
                return ErrorResponse.from_message(f"Provider not found: {provider}", 404)
            try:
                audio = filter_none(
                    voice=config.voice,
                    format=config.response_format,
                    language=config.language,
                )
                response = await self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"{config.instrcutions} Text: {config.input}",
                        }
                    ],
                    model=config.model,
                    provider=provider,
                    prompt=config.input,
                    api_key=api_key,
                    download_media=config.download_media,
                    **filter_none(
                        audio=audio if audio else None,
                    ),
                )
                if response.choices[0].message.audio is not None:
                    response = base64.b64decode(response.choices[0].message.audio.data)
                    return Response(
                        response,
                        media_type=f"audio/{config.response_format.replace('mp3', 'mpeg')}",
                    )
                elif isinstance(response.choices[0].message.content, AudioResponse):
                    response = response.choices[0].message.content.data
                    response = response.replace("/media", get_media_dir())

                    def delete_file():
                        try:
                            os.remove(response)
                        except Exception as e:
                            logger.exception(e)

                    return FileResponse(
                        response, background=BackgroundTask(delete_file)
                    )
            except (ModelNotFoundError, ProviderNotFoundError) as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_404_NOT_FOUND)
            except MissingAuthError as e:
                logger.exception(e)
                return ErrorResponse.from_exception(e, None, HTTP_401_UNAUTHORIZED)
            except Exception as e:
                logger.exception(e)
                return ErrorResponse.from_exception(
                    e, None, HTTP_500_INTERNAL_SERVER_ERROR
                )

        @self.app.post(
            "/v1/upload_cookies",
            responses={
                HTTP_200_OK: {"model": List[FileResponseModel]},
            },
        )
        def upload_cookies(
            files: List[UploadFile],
            credentials: Annotated[
                HTTPAuthorizationCredentials, Depends(Api.security)
            ] = None,
        ):
            response_data = []
            if not AppConfig.ignore_cookie_files:
                for file in files:
                    try:
                        if (
                            file
                            and file.filename.endswith(".json")
                            or file.filename.endswith(".har")
                        ):
                            filename = os.path.basename(file.filename)
                            with open(
                                os.path.join(get_cookies_dir(), filename), "wb"
                            ) as f:
                                shutil.copyfileobj(file.file, f)
                            response_data.append({"filename": filename})
                    finally:
                        file.file.close()
                read_cookie_files()
            return response_data

        @self.app.get(
            "/images/{filename}",
            responses={
                HTTP_200_OK: {"content": {"image/*": {}}},
                HTTP_404_NOT_FOUND: {},
            },
        )
        @self.app.get(
            "/media/{filename}",
            responses={
                HTTP_200_OK: {"content": {"image/*": {}, "audio/*": {}}, "video/*": {}},
                HTTP_404_NOT_FOUND: {},
            },
        )
        async def get_media(filename, request: Request, thumbnail: bool = False):
            def get_timestamp(str):
                m = re.match("^[0-9]+", str)
                if m:
                    return int(m.group(0))
                else:
                    return 0

            media_dir = os.path.realpath(get_media_dir())
            clean_filename = secure_filename(os.path.basename(filename))
            if not clean_filename:
                return ErrorResponse.from_message("Invalid file name", HTTP_400_BAD_REQUEST)

            target = os.path.realpath(os.path.join(media_dir, clean_filename))
            if not target.startswith(media_dir + os.sep):
                return ErrorResponse.from_message("Access denied", HTTP_403_FORBIDDEN)

            thumbnail_path = None
            if thumbnail and has_pillow:
                thumbnail_dir = os.path.realpath(os.path.join(media_dir, "thumbnails"))
                os.makedirs(thumbnail_dir, exist_ok=True)
                cand_thumb = os.path.realpath(os.path.join(thumbnail_dir, clean_filename))
                if cand_thumb.startswith(thumbnail_dir + os.sep):
                    thumbnail_path = cand_thumb

            if not os.path.isfile(target):
                decoded_name = secure_filename(os.path.basename(unquote_plus(filename)))
                if decoded_name:
                    cand_other = os.path.realpath(os.path.join(media_dir, decoded_name))
                    if cand_other.startswith(media_dir + os.sep) and os.path.isfile(cand_other):
                        target = cand_other

            ext = os.path.splitext(clean_filename)[1][1:]
            mime_type = EXTENSIONS_MAP.get(ext)
            stat_result = SimpleNamespace()
            stat_result.st_size = 0
            stat_result.st_mtime = get_timestamp(clean_filename)
            if thumbnail and has_pillow and thumbnail_path and os.path.isfile(thumbnail_path):
                stat_result.st_size = os.stat(thumbnail_path).st_size
            elif not thumbnail and os.path.isfile(target):
                stat_result.st_size = os.stat(target).st_size
            headers = {
                "cache-control": "public, max-age=31536000",
                "last-modified": formatdate(stat_result.st_mtime, usegmt=True),
                "etag": f'"{hashlib.md5(clean_filename.encode()).hexdigest()}"',
                **(
                    {
                        "content-length": str(stat_result.st_size),
                    }
                    if stat_result.st_size
                    else {}
                ),
                **(
                    {}
                    if thumbnail or mime_type is None
                    else {
                        "content-type": mime_type,
                    }
                ),
            }
            response = FileResponse(
                target,
                headers=headers,
                filename=clean_filename,
            )
            try:
                if_none_match = request.headers["if-none-match"]
                etag = response.headers["etag"]
                if etag in [tag.strip(" W/") for tag in if_none_match.split(",")]:
                    return NotModifiedResponse(response.headers)
            except KeyError:
                pass
            if not os.path.isfile(target) and mime_type is not None:
                source_url = get_source_url(str(request.query_params))
                ssl = None
                if source_url is None:
                    backend_url = os.environ.get("G4F_BACKEND_URL")
                    if backend_url:
                        source_url = f"{backend_url}/media/{clean_filename}"
                        ssl = False
                if source_url is not None:
                    if not is_safe_url(source_url):
                        return ErrorResponse.from_message("Invalid or unsafe source URL", HTTP_400_BAD_REQUEST)
                    try:
                        await copy_media([source_url], target=target, ssl=ssl)
                        debug.log(f"File copied from {source_url}")
                    except Exception as e:
                        debug.error(f"Download failed:  {source_url}")
                        debug.error(e)
                        return ErrorResponse.from_message("Failed to fetch remote media", HTTP_502_BAD_GATEWAY)
            if thumbnail and has_pillow and thumbnail_path:
                try:
                    if not os.path.isfile(thumbnail_path) and os.path.isfile(target):
                        image = Image.open(target)
                        process_image(image, save=thumbnail_path)
                        debug.log(f"Thumbnail created: {thumbnail_path}")
                except Exception as e:
                    logger.exception(e)
            if thumbnail and has_pillow and thumbnail_path and os.path.isfile(thumbnail_path):
                result = thumbnail_path
            else:
                result = target
            if not os.path.isfile(result) or not result.startswith(media_dir + os.sep):
                return ErrorResponse.from_message("File not found", HTTP_404_NOT_FOUND)

            async def stream():
                with open(result, "rb") as file:
                    while True:
                        if request is not None and await request.is_disconnected():
                            break
                        chunk = file.read(65536)
                        if not chunk:
                            break
                        yield chunk

            return StreamingResponse(stream(), headers=headers)

        @self.app.get(
            "/thumbnail/{filename}",
            responses={
                HTTP_200_OK: {"content": {"image/*": {}, "audio/*": {}}, "video/*": {}},
                HTTP_404_NOT_FOUND: {},
            },
        )
        async def get_media_thumbnail(filename: str, request: Request):
            return await get_media(filename, request, True)

        @self.app.get("/logs", response_class=HTMLResponse)
        async def logs_inspector():
            return HTMLResponse(_LOGS_HTML)

        @self.app.get("/api/logs")
        async def get_logs(limit: int = 500, offset: int = 0):
            entries = list(_request_log)
            total = len(entries)
            start = max(0, total - limit - offset)
            end = total - offset if offset < total else total
            page = list(reversed(entries[start:end]))
            return JSONResponse(
                {"total": total, "entries": page}, headers={"Cache-Control": "no-store"}
            )

        @self.app.delete("/api/logs")
        async def clear_logs():
            _request_log.clear()
            return JSONResponse({"status": "cleared"})


def format_exception(
    e: Union[Exception, str],
    config: Union[ChatCompletionsConfig, ImageGenerationConfig] = None,
    image: bool = False,
) -> str:
    provider = AppConfig.media_provider if image else AppConfig.provider
    model = AppConfig.model
    if config is not None:
        if config.provider is not None:
            provider = config.provider
        if config.model is not None:
            model = config.model
    if isinstance(e, str):
        message = e
    elif isinstance(e, ModelNotFoundError):
        message = "ModelNotFoundError: Model not found"
    elif isinstance(e, ProviderNotFoundError):
        message = "ProviderNotFoundError: Provider not found"
    elif isinstance(e, MissingAuthError):
        message = "MissingAuthError: Authentication required"
    else:
        message = "Request execution failed"
    return json.dumps(
        {
            "error": {"message": message},
            **filter_none(
                model=model, provider=getattr(provider, "__name__", provider)
            ),
        }
    )


def run_api(
    host: str = "0.0.0.0",
    port: int = None,
    bind: str = None,
    debug: bool = False,
    use_colors: bool = None,
    **kwargs,
) -> None:
    print(
        f"Starting server... [g4f v-{g4f.version.utils.current_version}]"
        + (" (debug)" if debug else "")
    )

    if use_colors is None:
        use_colors = debug

    if bind is not None:
        host, port = bind.split(":")

    if port is None:
        port = DEFAULT_PORT

    if AppConfig.demo and debug:
        method = "create_app_with_demo_and_debug"
    elif AppConfig.gui and debug:
        method = "create_app_with_gui_and_debug"
    else:
        method = "create_app_debug" if debug else "create_app"

    uvicorn_options = {
        "timeout_keep_alive": 65,
        "backlog": 2048,
        # Avoid hanging forever on shutdown when a long-running request (e.g.
        # a browser-based provider login) is still in-flight.
        "timeout_graceful_shutdown": 10,
    }
    uvicorn_options.update(filter_none(**kwargs))

    uvicorn.run(
        f"g4f.api:{method}",
        host=host,
        port=int(port),
        factory=True,
        use_colors=use_colors,
        **uvicorn_options,
    )
