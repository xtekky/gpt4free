from __future__ import annotations

import json
import os
import os.path
import random
import requests

from ...requests import StreamSession, raise_for_status
from ...errors import MissingRequirementsError
from ... import debug

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
    global chatArks
    if chatArks is None:
        chatArks = readHAR()
    return await sendRequest(random.choice(chatArks), proxy)

async def get_telemetry_ids(proxy: str = None) -> list:
    if debug.logging:
        print('Getting telemetry_id for you.com with nodriver')
    try:
        from nodriver import start
    except ImportError:
        raise MissingRequirementsError('Install "nodriver" package | pip install -U nodriver')
    try:
        browser = await start()
        tab = browser.main_tab
        await browser.get("https://you.com")

        while not await tab.evaluate('"GetTelemetryID" in this'):
            await tab.sleep(1)

        async def get_telemetry_id():
            public_token = "public-token-live-507a52ad-7e69-496b-aee0-1c9863c7c819"
            telemetry_url = "https://telemetry.stytch.com/submit"
            return await tab.evaluate(f'this.GetTelemetryID("{public_token}", "{telemetry_url}");', await_promise=True)

        # for _ in range(500):
        #     with open("hardir/you.com_telemetry_ids.txt", "a") as f:
        #         f.write((await get_telemetry_id()) + "\n")

        return [await get_telemetry_id() for _ in range(4)]
    finally:
        try:
            await tab.close()
        except Exception as e:
            print(f"Error occurred while closing tab: {str(e)}")
        try:
            await browser.stop()
        except Exception as e:
            pass

    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'Connection': 'keep-alive',
        'Content-type': 'application/x-www-form-urlencoded',
        'Origin': 'https://you.com',
        'Referer': 'https://you.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }
    
    proxies = {
            'http': proxy,
            'https': proxy} if proxy else None

    response = requests.post('https://telemetry.stytch.com/submit', 
                            headers=headers, data=payload, proxies=proxies)

    if '-' in response.text:
        print(f'telemetry generated: {response.text}')
    
    return (response.text)
