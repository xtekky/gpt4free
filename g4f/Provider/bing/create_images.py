import asyncio
import time, json, os
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import Generator

from ..create_images import CreateImagesProvider
from ..helper import get_cookies, get_event_loop
from ...webdriver import WebDriver, get_driver_cookies, get_browser
from ...base_provider import ProviderType

BING_URL = "https://www.bing.com"

def wait_for_login(driver: WebDriver, timeout: int = 1200) -> None:
    driver.get(f"{BING_URL}/")
    value = driver.get_cookie("_U")
    if value:
        return
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout error")
        value = driver.get_cookie("_U")
        if value:
            return
        time.sleep(0.5)

def create_session(cookies: dict) -> ClientSession:
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh-TW;q=0.7,zh;q=0.6",
        "content-type": "application/x-www-form-urlencoded",
        "referrer-policy": "origin-when-cross-origin",
        "referrer": "https://www.bing.com/images/create/",
        "origin": "https://www.bing.com",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.54",
        "sec-ch-ua": "\"Microsoft Edge\";v=\"111\", \"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"111\"",
        "sec-ch-ua-mobile": "?0",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
    }
    if cookies:
        headers["cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
    return ClientSession(headers=headers)

async def create_images(session: ClientSession, prompt: str, proxy: str = None, timeout: int = 300) -> list:
    url_encoded_prompt = quote(prompt)    
    payload = f"q={url_encoded_prompt}&rt=4&FORM=GENCRE"
    url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=4&FORM=GENCRE"
    async with session.post(
        url,
        allow_redirects=False,
        data=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        errors = [
            "this prompt is being reviewed",
            "this prompt has been blocked",
            "we're working hard to offer image creator in more languages"
        ]
        text = (await response.text()).lower()
        for error in errors:
            if error in text:
                raise RuntimeError(f"Create images failed: {error}")
    if response.status != 302:
        url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=3&FORM=GENCRE"
        async with session.post(url, allow_redirects=False, proxy=proxy, timeout=timeout) as response:
            if response.status != 302:
                raise RuntimeError(f"Create images failed. Status Code: {response.status}")

    redirect_url = response.headers["Location"].replace("&nfy=1", "")
    redirect_url = f"{BING_URL}{redirect_url}"
    request_id = redirect_url.split("id=")[1]
    async with session.get(redirect_url) as response:
        response.raise_for_status()

    polling_url = f"{BING_URL}/images/create/async/results/{request_id}?q={url_encoded_prompt}"
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Timeout error after {timeout} seconds")
        async with session.get(polling_url) as response:
            if response.status != 200:
                raise RuntimeError(f"Polling images faild. Status Code: {response.status}")
            text = await response.text()
            if not text:
                await asyncio.sleep(1)
            else:
                break
    error = None
    try:
        error = json.loads(text).get("errorMessage")
    except:
        pass
    if error == "Pending":
        raise RuntimeError("Prompt is been blocked")
    elif error:
        raise RuntimeError(error)
    return read_images(text)

def read_images(text: str) -> list:
    html_soup = BeautifulSoup(text, "html.parser")
    tags = html_soup.find_all("img")
    image_links = [img["src"] for img in tags if "mimg" in img["class"]]
    images = [link.split("?w=")[0] for link in image_links]
    bad_images = [
        "https://r.bing.com/rp/in-2zU3AJUdkgFe7ZKv19yPBHVs.png",
        "https://r.bing.com/rp/TX9QuO3WzcCJz1uaaSwQAz39Kb0.jpg",
    ]
    if any(im in bad_images for im in images):
        raise RuntimeError("Bad images found")
    if not images:
        raise RuntimeError("No images found")
    return images

def format_images_markdown(images: list, prompt: str) -> str:
    images = [f"[![#{idx+1} {prompt}]({image}?w=200&h=200)]({image})" for idx, image in enumerate(images)]
    images = "\n".join(images)
    start_flag = "<!-- generated images start -->\n"
    end_flag = "<!-- generated images end -->\n"
    return f"\n{start_flag}{images}\n{end_flag}\n"

async def create_images_markdown(cookies: dict, prompt: str, proxy: str = None) -> str:
    session = create_session(cookies)
    try:
        images = await create_images(session, prompt, proxy)
        return format_images_markdown(images, prompt)
    finally:
        await session.close()

def get_cookies_from_browser(proxy: str = None) -> dict:
    driver = get_browser(proxy=proxy)
    try:
        wait_for_login(driver)
        return get_driver_cookies(driver)
    finally:
        driver.quit()

def create_completion(prompt: str, cookies: dict = None, proxy: str = None) -> Generator:
    loop = get_event_loop()
    if not cookies:
        cookies = get_cookies(".bing.com")
    if "_U" not in cookies:
        login_url = os.environ.get("G4F_LOGIN_URL")
        if login_url:
            yield f"Please login: [Bing]({login_url})\n\n"
        cookies = get_cookies_from_browser(proxy)
    yield loop.run_until_complete(create_images_markdown(cookies, prompt, proxy))

async def create_async(prompt: str, cookies: dict = None, proxy: str = None) -> str:
    if not cookies:
        cookies = get_cookies(".bing.com")
    if "_U" not in cookies:
        cookies = get_cookies_from_browser(proxy)
    return await create_images_markdown(cookies, prompt, proxy)

def patch_provider(provider: ProviderType) -> CreateImagesProvider:
    return CreateImagesProvider(provider, create_completion, create_async)