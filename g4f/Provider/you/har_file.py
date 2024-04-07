from __future__ import annotations

import json
import os
import random
import uuid

from ...requests import StreamSession, raise_for_status

class NoValidHarFileError(Exception):
    ...

class arkReq:
    def __init__(self, arkURL, arkHeaders, arkBody, arkCookies, userAgent):
        self.arkURL = arkURL
        self.arkHeaders = arkHeaders
        self.arkBody = arkBody
        self.arkCookies = arkCookies
        self.userAgent = userAgent

arkPreURL = "https://telemetry.stytch.com/submit"
chatArks: list = None

def readHAR():
    dirPath = "./"
    harPath = []
    chatArks = []
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".har"):
                harPath.append(os.path.join(root, file))
        if harPath:
            break
    if not harPath:
        raise NoValidHarFileError("No .har file found")
    for path in harPath:
        with open(path, 'rb') as file:
            try:
                harFile = json.load(file)
            except json.JSONDecodeError:
                # Error: not a HAR file!
                continue
            for v in harFile['log']['entries']:
                if arkPreURL in v['request']['url']:
                    chatArks.append(parseHAREntry(v))
    if not chatArks:
        raise NoValidHarFileError("No telemetry in .har files found")
    return chatArks

def parseHAREntry(entry) -> arkReq:
    tmpArk = arkReq(
        arkURL=entry['request']['url'],
        arkHeaders={h['name'].lower(): h['value'] for h in entry['request']['headers'] if h['name'].lower() not in ['content-length', 'cookie'] and not h['name'].startswith(':')},
        arkBody=entry['request']['postData']['text'],
        arkCookies={c['name']: c['value'] for c in entry['request']['cookies']},
        userAgent=""
    )
    tmpArk.userAgent = tmpArk.arkHeaders.get('user-agent', '')
    return tmpArk

async def sendRequest(tmpArk: arkReq, proxy: str = None):
    async with StreamSession(headers=tmpArk.arkHeaders, cookies=tmpArk.arkCookies, proxies={"all": proxy}) as session:
        async with session.post(tmpArk.arkURL, data=tmpArk.arkBody) as response:
            await raise_for_status(response)
            return await response.text()

async def get_dfp_telemetry_id(proxy: str = None):
    return str(uuid.uuid4())
    global chatArks
    if chatArks is None:
        chatArks = readHAR()
    return await sendRequest(random.choice(chatArks), proxy)