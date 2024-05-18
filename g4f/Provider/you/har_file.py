from __future__ import annotations

import json
import os
import os.path
import random
import logging

from ...requests import StreamSession, raise_for_status
from ...cookies import get_cookies_dir
from ...errors import MissingRequirementsError
from ... import debug

logging.basicConfig(level=logging.ERROR)

class NoValidHarFileError(Exception):
    ...

class arkReq:
    def __init__(self, arkURL, arkHeaders, arkBody, arkCookies, userAgent):
        self.arkURL = arkURL
        self.arkHeaders = arkHeaders
        self.arkBody = arkBody
        self.arkCookies = arkCookies
        self.userAgent = userAgent

telemetry_url = "https://telemetry.stytch.com/submit"
public_token = "public-token-live-507a52ad-7e69-496b-aee0-1c9863c7c819"
chatArks: list = None

def readHAR():
    harPath = []
    chatArks = []
    for root, dirs, files in os.walk(get_cookies_dir()):
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
                if v['request']['url'] == telemetry_url:
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
    async with StreamSession(headers=tmpArk.arkHeaders, cookies=tmpArk.arkCookies, proxy=proxy) as session:
        async with session.post(tmpArk.arkURL, data=tmpArk.arkBody) as response:
            await raise_for_status(response)
            return await response.text()

async def create_telemetry_id(proxy: str = None):
    global chatArks
    if chatArks is None:
        chatArks = readHAR()
    return await sendRequest(random.choice(chatArks), proxy)

async def get_telemetry_ids(proxy: str = None) -> list:
    try:
        return [await create_telemetry_id(proxy)]
    except NoValidHarFileError as e:
        if debug.logging:
            logging.error(e)

    try:
        from nodriver import start
    except ImportError:
        raise MissingRequirementsError('Add .har file from you.com or install "nodriver" package | pip install -U nodriver')
    if debug.logging:
        logging.error('Getting telemetry_id for you.com with nodriver')

    browser = page = None
    try:
        browser = await start(
            browser_args=None if proxy is None else [f"--proxy-server={proxy}"],
        )
        page = await browser.get("https://you.com")
        while not await page.evaluate('"GetTelemetryID" in this'):
            await page.sleep(1)
        async def get_telemetry_id():
            return await page.evaluate(
                f'this.GetTelemetryID("{public_token}", "{telemetry_url}");',
                await_promise=True
            )
        return [await get_telemetry_id()]
    finally:
        try:
            if page is not None:
                await page.close()
            if browser is not None:
                await browser.stop()
        except Exception as e:
            if debug.logging:
                logging.error(e)
