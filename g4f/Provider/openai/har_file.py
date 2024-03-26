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

class arkReq:
    def __init__(self, arkURL, arkBx, arkHeader, arkBody, arkCookies, userAgent):
        self.arkURL = arkURL
        self.arkBx = arkBx
        self.arkHeader = arkHeader
        self.arkBody = arkBody
        self.arkCookies = arkCookies
        self.userAgent = userAgent

arkPreURL = "https://tcr9i.chat.openai.com/fc/gt2/public_key/35536E1E-65B4-4D96-9D97-6ADB7EFF8147"
sessionUrl = "https://chat.openai.com/api/auth/session"
chatArk: arkReq = None
accessToken: str = None
cookies: dict = None

def readHAR():
    dirPath = "./"
    harPath = []
    chatArks = []
    accessToken = None
    cookies = {}
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".har"):
                harPath.append(os.path.join(root, file))
        if harPath:
            break
    if not harPath:
        raise RuntimeError("No .har file found")
    for path in harPath:
        with open(path, 'rb') as file:
            try:
                harFile = json.loads(file.read())
            except json.JSONDecodeError:
                # Error: not a HAR file!
                continue
            for v in harFile['log']['entries']:
                if arkPreURL in v['request']['url']:
                    chatArks.append(parseHAREntry(v))
                elif v['request']['url'] == sessionUrl:
                    accessToken = json.loads(v["response"]["content"]["text"]).get("accessToken")
                    cookies = {c['name']: c['value'] for c in v['request']['cookies']}
    if not accessToken:
        RuntimeError("No accessToken found in .har files")
    if not chatArks:
        return None, accessToken, cookies
    return chatArks.pop(), accessToken, cookies

def parseHAREntry(entry) -> arkReq:
    tmpArk = arkReq(
        arkURL=entry['request']['url'],
        arkBx="",
        arkHeader={h['name'].lower(): h['value'] for h in entry['request']['headers'] if h['name'].lower() not in ['content-length', 'cookie'] and not h['name'].startswith(':')},
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
    if not chatArk:
        raise RuntimeError("No .har file with arkose found")

    tmpArk: arkReq = deepcopy(chatArk)
    if tmpArk is None or not tmpArk.arkBody or not tmpArk.arkHeader:
        raise RuntimeError("The .har file is not valid")
    bda, bw = getBDA(tmpArk)

    tmpArk.arkBody['bda'] = base64.b64encode(bda.encode()).decode()
    tmpArk.arkBody['rnd'] = str(random.random())
    tmpArk.arkHeader['x-ark-esync-value'] = bw
    return tmpArk

async def sendRequest(tmpArk: arkReq, proxy: str = None):
    async with StreamSession(headers=tmpArk.arkHeader, cookies=tmpArk.arkCookies, proxies={"https": proxy}) as session:
        async with session.post(tmpArk.arkURL, data=tmpArk.arkBody) as response:
            arkose = (await response.json()).get("token")
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

async def getArkoseAndAccessToken(proxy: str):
    global chatArk, accessToken, cookies
    if chatArk is None or accessToken is None:
        chatArk, accessToken, cookies = readHAR()
    if chatArk is None:
        return None, accessToken, cookies
    newReq = genArkReq(chatArk)
    return await sendRequest(newReq, proxy), accessToken, cookies