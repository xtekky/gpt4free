from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
import random
from urllib.parse import unquote
from copy import deepcopy

from .crypt import decrypt, encrypt
from ...requests import StreamSession
from ...cookies import get_cookies_dir
from ...errors import NoValidHarFileError
from ... import debug

arkose_url = "https://tcr9i.chat.openai.com/fc/gt2/public_key/35536E1E-65B4-4D96-9D97-6ADB7EFF8147"
backend_url = "https://chatgpt.com/backend-api/conversation"
backend_anon_url = "https://chatgpt.com/backend-anon/conversation"
start_url = "https://chatgpt.com/"
conversation_url = "https://chatgpt.com/c/"

class RequestConfig:
    cookies: dict = None
    headers: dict = None
    access_token: str = None
    proof_token: list = None
    turnstile_token: str = None
    arkose_request: arkReq = None
    arkose_token: str = None
    headers: dict = {}
    cookies: dict = {}
    data_build: str = "prod-db8e51e8414e068257091cf5003a62d3d4ee6ed0"

class arkReq:
    def __init__(self, arkURL, arkBx, arkHeader, arkBody, arkCookies, userAgent):
        self.arkURL = arkURL
        self.arkBx = arkBx
        self.arkHeader = arkHeader
        self.arkBody = arkBody
        self.arkCookies = arkCookies
        self.userAgent = userAgent

def get_har_files():
    if not os.access(get_cookies_dir(), os.R_OK):
        raise NoValidHarFileError("har_and_cookies dir is not readable")
    harPath = []
    for root, _, files in os.walk(get_cookies_dir()):
        for file in files:
            if file.endswith(".har"):
                harPath.append(os.path.join(root, file))
    if not harPath:
        raise NoValidHarFileError("No .har file found")
    harPath.sort(key=lambda x: os.path.getmtime(x))
    return harPath

def readHAR():
    for path in get_har_files():
        with open(path, 'rb') as file:
            try:
                harFile = json.loads(file.read())
            except json.JSONDecodeError:
                # Error: not a HAR file!
                continue
            for v in harFile['log']['entries']:
                v_headers = get_headers(v)
                if arkose_url == v['request']['url']:
                    RequestConfig.arkose_request = parseHAREntry(v)
                elif v['request']['url'].startswith(start_url):
                    try:
                        match = re.search(r'"accessToken":"(.*?)"', v["response"]["content"]["text"])
                        if match:
                            RequestConfig.access_token = match.group(1)
                    except KeyError:
                        pass
                    try:
                        if "openai-sentinel-proof-token" in v_headers:
                            RequestConfig.headers = v_headers
                            RequestConfig.proof_token = json.loads(base64.b64decode(
                                v_headers["openai-sentinel-proof-token"].split("gAAAAAB", 1)[-1].encode()
                            ).decode())
                        if "openai-sentinel-turnstile-token" in v_headers:
                            RequestConfig.turnstile_token = v_headers["openai-sentinel-turnstile-token"]
                        if "authorization" in v_headers:
                            RequestConfig.access_token = v_headers["authorization"].split(" ")[1]
                        RequestConfig.cookies = {c['name']: c['value'] for c in v['request']['cookies']}
                    except Exception as e:
                        debug.log(f"Error on read headers: {e}")
    if RequestConfig.proof_token is None:
        raise NoValidHarFileError("No proof_token found in .har files")

def get_headers(entry) -> dict:
    return {h['name'].lower(): h['value'] for h in entry['request']['headers'] if h['name'].lower() not in ['content-length', 'cookie'] and not h['name'].startswith(':')}

def parseHAREntry(entry) -> arkReq:
    tmpArk = arkReq(
        arkURL=entry['request']['url'],
        arkBx="",
        arkHeader=get_headers(entry),
        arkBody={p['name']: unquote(p['value']) for p in entry['request']['postData']['params'] if p['name'] not in ['rnd']},
        arkCookies={c['name']: c['value'] for c in entry['request']['cookies']},
        userAgent=""
    )
    tmpArk.userAgent = tmpArk.arkHeader.get('user-agent', '')
    bda = tmpArk.arkBody["bda"]
    bw = tmpArk.arkHeader['x-ark-esync-value']
    tmpArk.arkBx = decrypt(bda, tmpArk.userAgent + bw)
    return tmpArk

def genArkReq(chatArk: arkReq) -> arkReq:
    tmpArk: arkReq = deepcopy(chatArk)
    if tmpArk is None or not tmpArk.arkBody or not tmpArk.arkHeader:
        raise RuntimeError("The .har file is not valid")
    bda, bw = getBDA(tmpArk)

    tmpArk.arkBody['bda'] = base64.b64encode(bda.encode()).decode()
    tmpArk.arkBody['rnd'] = str(random.random())
    tmpArk.arkHeader['x-ark-esync-value'] = bw
    return tmpArk

async def sendRequest(tmpArk: arkReq, proxy: str = None) -> str:
    async with StreamSession(headers=tmpArk.arkHeader, cookies=tmpArk.arkCookies, proxies={"https": proxy}) as session:
        async with session.post(tmpArk.arkURL, data=tmpArk.arkBody) as response:
            data = await response.json()
            arkose = data.get("token")
    if "sup=1|rid=" not in arkose:
        return RuntimeError("No valid arkose token generated")
    return arkose

def getBDA(arkReq: arkReq):
    bx = arkReq.arkBx
    
    bx = re.sub(r'"key":"n","value":"\S*?"', f'"key":"n","value":"{getN()}"', bx)
    oldUUID_search = re.search(r'"key":"4b4b269e68","value":"(\S*?)"', bx)
    if oldUUID_search:
        oldUUID = oldUUID_search.group(1)
        newUUID = str(uuid.uuid4())
        bx = bx.replace(oldUUID, newUUID)

    bw = getBw(getBt())
    encrypted_bx = encrypt(bx, arkReq.userAgent + bw)
    return encrypted_bx, bw

def getBt() -> int:
    return int(time.time())

def getBw(bt: int) -> str:
    return str(bt - (bt % 21600))

def getN() -> str:
    timestamp = str(int(time.time()))
    return base64.b64encode(timestamp.encode()).decode()

async def get_request_config(proxy: str) -> RequestConfig:
    if RequestConfig.proof_token is None:
        readHAR()
    if RequestConfig.arkose_request is not None:
        RequestConfig.arkose_token = await sendRequest(genArkReq(RequestConfig.arkose_request), proxy)
    return RequestConfig