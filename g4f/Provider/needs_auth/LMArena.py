from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse, parse_qs



from g4f.image import to_bytes, detect_file_type

try:
    import curl_cffi
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

try:
    import zendriver as nodriver
    from zendriver import cdp
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import get_args_from_nodriver, raise_for_status, merge_cookies
from ...requests import StreamSession
from ...cookies import get_cookies_dir
from ...tools.files import secure_filename
from ...errors import ModelNotFoundError, CloudflareError, MissingAuthError, MissingRequirementsError, \
    RateLimitError
from ...providers.response import FinishReason, Usage, JsonConversation, ImageResponse, Reasoning, PlainTextResponse, \
    JsonRequest
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin
from ..helper import get_last_user_message
from ... import debug


def uuid7():
    """
    Generate a UUIDv7 using Unix epoch (milliseconds since 1970-01-01)
    matching the browser's implementation.
    """
    timestamp_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = timestamp_ms << 80
    uuid_int |= (0x7000 | rand_a) << 64
    uuid_int |= (0x8000000000000000 | rand_b)

    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


# Global variables to manage Image Cache
ImagesCache: Dict[str, dict[str, str]] = {}


def check_link_expiry(url):
    # Parse the URL and its query parameters
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)
    amz_date_str = params.get("X-Amz-Date", [None])[0]
    expires_delta = params.get("X-Amz-Expires", [None])[0]
    if not amz_date_str or not expires_delta:
        return False
    creation_time = datetime.strptime(amz_date_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    expiry_time = creation_time.timestamp() + int(expires_delta)
    current_time = datetime.now(timezone.utc).timestamp()
    return current_time <= expiry_time


if has_nodriver:
    async def click_trunstile(page: nodriver.Tab, element='document.getElementById("cf-turnstile")'):
        for _ in range(3):
            size = None
            for idx in range(15):
                size = await page.js_dumps(f'{element}?.getBoundingClientRect()||{{}}')
                debug.log(f"Found size: {size.get('x'), size.get('y')}")
                if "x" not in size:
                    break
                await page.flash_point(size.get("x") + idx * 3, size.get("y") + idx * 3)
                await page.mouse_click(size.get("x") + idx * 3, size.get("y") + idx * 3)
                await asyncio.sleep(2)
            if "x" not in size:
                break
        debug.log("Finished clicking trunstile.")


class LMArena(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    label = "LMArena"
    url = "https://arena.ai"
    share_url = None
    create_evaluation = "https://arena.ai/nextjs-api/stream/create-evaluation"
    post_to_evaluation = "https://arena.ai/nextjs-api/stream/post-to-evaluation/{id}"
    models_url = "https://arena.ai/?mode=direct"
    working = True
    active_by_default = True
    use_stream_timeout = False

    _models_loaded = False
    image_cache = True
    _next_actions = {
        "generateUploadUrl": "7012303914af71fce235a732cde90253f7e2986f2b",
        "getSignedUrl": "605373b76a30947cc26be49fc7b00c885910e21559",
        "updateTouConsent": "40efff1040868c07750a939a0d8120025f246dfe28",
        "createPointwiseFeedback": "605a0e3881424854b913fe1d76d222e50731b6037b",
        "createPairwiseFeedback": "600777eb84863d7e79d85d214130d3214fc744c80f",
        "getProxyImage": "60049198d4936e6b7acc63719b63b89284c58683e6",
        "deleteEvaluationSession": "6009c985d7e84eae2ec94547453ba388005b22e2a5",
        "getEmailProvider": "607c2dd3d84af5a00b322b577498d1b2a739c5dfe0",
        "deleteAccount": "40a57e8c369eaf8a82483fae2f8106489ce041dffd",
    }

    @classmethod
    def load_models(cls, models_data: str):
        cls.text_models = {model["publicName"]: model["id"] for model in models_data if
                            "text" in model["capabilities"]["outputCapabilities"]}
        cls.image_models = {model["publicName"]: model["id"] for model in models_data if
                                "image" in model["capabilities"]["outputCapabilities"]}
        cls.video_models = {model["publicName"]: model["id"] for model in models_data if
                                "video" in model["capabilities"]["outputCapabilities"]}
        cls.vision_models = [model["publicName"] for model in models_data if
                                "image" in model["capabilities"]["inputCapabilities"]]
        cls.models = list(cls.text_models) + list(cls.image_models)
        cls.default_model = list(cls.text_models.keys())[0]
        cls._models_loaded = True

    @classmethod
    def load_models_from_cache(cls):
        models_path = Path(get_cookies_dir()) / ".models" / f"{secure_filename(cls.models_url)}.json"
        if models_path.exists():
            try:
                data = models_path.read_text()
                models_data = json.loads(data)
                for key, value in models_data.items():
                    setattr(cls, key, value)
                cls._models_loaded = True
                debug.log(f"Loaded models from cache: {cls.models}")
            except Exception as e:
                debug.error(f"Failed to load cached models from {models_path}: {e}")

    @classmethod
    def get_models(cls, timeout: int = None, **kwargs) -> list[str]:
        if not cls._models_loaded and has_curl_cffi:
            # Try to load models from cache
            args = cls.read_args()
            if not args:
                cls.load_models_from_cache()
            if cls._models_loaded:
                return cls.models
            response = curl_cffi.get(cls.models_url, **args, timeout=timeout)
            if response.ok:
                for line in response.text.splitlines():
                    if "initialModels" not in line:
                        continue
                    line = line.split("initialModels", maxsplit=1)[-1].split("initialModelAId")[0][3:-3]
                    line = line.encode("utf-8").decode("unicode_escape")
                    models = json.loads(line)
                    cls.load_models(models)
                    cls.live += 1
                    break
                try:
                    models_path = Path(get_cookies_dir()) / ".models" / f"{secure_filename(cls.models_url)}.json"
                    models_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(models_path, "w") as f:
                        json.dump({
                            "text_models": cls.text_models,
                            "image_models": cls.image_models,
                            "video_models": cls.video_models,
                            "vision_models": cls.vision_models,
                            "models": cls.models,
                            "default_model": cls.default_model
                        }, f, indent=4)
                except Exception as e:
                    debug.error(f"Failed to cache models to {models_path}: {e}")
            else:
                cls.live -= 1
                cls.load_models_from_cache()
                debug.log(f"Failed to load models from {cls.url}: {response.status_code} {response.reason}")
        return cls.models

    @classmethod
    async def get_models_async(cls) -> list[str]:
        if not cls._models_loaded:
            async with StreamSession() as session:
                async with session.get(cls.models_url) as response:
                    await cls.__load_actions(await response.text())
        return cls.models

    @classmethod
    async def get_args_from_nodriver(cls, proxy, clear_cookies=False):
        cache_file = cls.get_cache_file()
        grecaptcha = []

        async def callback(page: nodriver.Tab):
            try:
                button = await page.find("Accept Cookies")
            except TimeoutError:
                button = None
            if button:
                await button.click()
            else:
                debug.log("No 'Accept Cookies' button found, skipping.")
            await asyncio.sleep(1)
            try:
                textarea = await page.select('textarea[name="message"]')
            except TimeoutError:
                textarea = None
            if textarea:
                await textarea.send_keys("Hello")
            # await asyncio.sleep(1)
            # button = await page.select('button[type="submit"]')
            # if button:
            #     await button.click()
            # button = await page.find("Agree")
            # if button:
            #     await button.click()
            # else:
            #     debug.log("No 'Agree' button found, skipping.")
            # await asyncio.sleep(1)
            # try:
            #     element = await page.select('[style="display: grid;"]')
            # except TimeoutError:
            #     element = None
            # if element:
            #     await click_trunstile(page, 'document.querySelector(\'[style="display: grid;"]\')')
            while not await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0'):
                debug.log("No authentication cookie found, waiting for authenticate.")
                #await page.select('#cf-turnstile', 300)
                #debug.log("Found Element: 'cf-turnstile'")
                await asyncio.sleep(3)
                #await click_trunstile(page)
            while not await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0'):
                await asyncio.sleep(1)
            while not await page.evaluate('!!document.querySelector(\'textarea\')'):
                await asyncio.sleep(1)
            while not await page.evaluate('window.grecaptcha && window.grecaptcha.enterprise'):
                await asyncio.sleep(1)
            captcha = await page.evaluate(
                """window.grecaptcha.enterprise.execute('6LeTGMcsAAAAALuIlkVwIxaAuZA8VledA6d3Nnb0',  { action: 'chat_submit' }  );""",
                await_promise=True)
            grecaptcha.append(captcha)
            debug.log("Obtained grecaptcha token.")
            html = await page.get_content()
            await cls.__load_actions(html)

        args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback,
                                            clear_cookies_except=["cf_clearance", "app_banner_state"] if clear_cookies else None)

        with cache_file.open("w") as f:
            json.dump(args, f)

        return args, next(iter(grecaptcha))

    @classmethod
    async def get_grecaptcha(cls, args, proxy):
        cache_file = cls.get_cache_file()
        grecaptcha = []

        async def callback(page: nodriver.Tab):
            while not await page.evaluate('window.grecaptcha && window.grecaptcha.enterprise'):
                await asyncio.sleep(1)
            captcha = await page.evaluate(
                """new Promise((resolve) => {
                    window.grecaptcha.enterprise.ready(async () => {
                        try {
                            const token = await window.grecaptcha.enterprise.execute(
                                '6LeTGMcsAAAAALuIlkVwIxaAuZA8VledA6d3Nnb0',
                                { action: 'chat_submit' }
                            );
                            resolve(token);
                        } catch (e) {
                            console.error("[LMArena API] reCAPTCHA execute failed:", e);
                            resolve(null);
                        }
                    });
                });""",
                await_promise=True
            )
            if isinstance(captcha, str):
                grecaptcha.append(captcha)
            else:
                raise Exception(captcha)
            html = await page.get_content()
            await cls.__load_actions(html)

        args = await get_args_from_nodriver(
            cls.url, proxy=proxy, callback=callback
        )

        with cache_file.open("w") as f:
            json.dump(args, f)

        return args, next(iter(grecaptcha))

    @classmethod
    async def __load_actions(cls, html):
        def pars_children(data):
            data = data["children"]
            if len(data) < 4:
                return
            if data[1] in ["div", "defs", "style", "script"]:
                return
            if data[0] == "$":
                pars_data(data[3])
            else:
                for child in data:
                    if isinstance(child, list) and len(data) >= 4:
                        pars_data(child[3])

        def pars_data(data):
            if not isinstance(data, (list, dict)):
                return
            if isinstance(data, dict):
                json_data = data
            elif data[0] == "$":
                if data[1] in ["div", "defs", "style", "script"]:
                    return
                json_data = data[3]
            else:
                return
            if not json_data:
                return
            if 'userState' in json_data:
                debug.log(json_data)
            elif 'initialModels' in json_data:
                models = json_data["initialModels"]
                cls.load_models(models)
            elif 'children' in json_data:
                pars_children(json_data)

        line_pattern = re.compile("^([0-9a-fA-F]+):(.*)")
        pattern = r'self\.__next_f\.push\((\[[\s\S]*?\])\)(?=<\/script>)'
        matches = re.findall(pattern, html)
        for match in matches:
            # Parse the JSON array
            data = json.loads(match)
            for chunk in data[1].split("\n"):
                match = line_pattern.match(chunk)
                if not match:
                    continue
                chunk_id, chunk_data = match.groups()
                if chunk_data.startswith("I["):
                    data = json.loads(chunk_data[1:])
                    async with StreamSession() as session:
                        if "Evaluation" == data[2]:
                            js_files = dict(zip(data[1][::2], data[1][1::2]))
                            for js_id, js in list(js_files.items())[::-1]:
                                js_url = f"{cls.url}/_next/{js}"
                                async with session.get(js_url) as js_response:
                                    js_text = await js_response.text()
                                    if "createServerReference" in js_text:
                                        cls.__extract_actions(js_text)

                elif chunk_data.startswith(("[", "{")):
                    try:
                        data = json.loads(chunk_data)
                        pars_data(data)
                    except json.decoder.JSONDecodeError:
                        ...

    @classmethod
    def __extract_actions(cls, js_text):
        # updateTouConsent, createPointwiseFeedback, createPairwiseFeedback, generateUploadUrl, getSignedUrl, getProxyImage
        start_id = re.findall(r'\("([a-f0-9]{40,})".*?"(\w+)"\)', js_text)
        for v, k in start_id:
            if len(v) == 42:
                cls._next_actions[k] = v
                debug.log(f"{k}: {v}")
            else:
                debug.error(f"wrong {k} value: {v}")

    @classmethod
    async def prepare_images(cls, args, media: list[tuple]) -> list[dict[str, str]]:
        files = []
        if not media:
            return files
        async with StreamSession(**args, ) as session:
            for index, (_file, file_name) in enumerate(media):
                data_bytes = to_bytes(_file)
                # Check Cache
                hasher = hashlib.md5()
                hasher.update(data_bytes)
                image_hash = hasher.hexdigest()
                file = ImagesCache.get(image_hash)
                if cls.image_cache and file:
                    if check_link_expiry(file.get("url")):
                        debug.log("Using cached image")
                        files.append(file)
                        continue
                    debug.log("Expiry cached image")

                extension, file_type = detect_file_type(data_bytes)
                file_name = file_name or f"file-{len(data_bytes)}{extension}"
                async with session.post(
                        url=cls.url,
                        json=[file_name, file_type],
                        headers={
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": cls._next_actions["generateUploadUrl"],
                            "referer": cls.url
                        }
                ) as response:
                    await raise_for_status(response)
                    text = await response.text()
                    line = next(filter(lambda x: x.startswith("1:"), text.split("\n")), "")
                    if not line:
                        raise Exception("Failed to get upload URL")
                    chunk = json.loads(line[2:])
                    if not chunk.get("success"):
                        raise Exception("Failed to get upload URL")
                    uploadUrl = chunk.get("data", {}).get("uploadUrl")
                    key = chunk.get("data", {}).get("key")
                    if not uploadUrl:
                        raise Exception("Failed to get upload URL")

                async with session.put(
                    url=uploadUrl,
                    headers={
                        "content-type": file_type,
                    },
                    data=data_bytes,
                ) as response:
                    await raise_for_status(response)
                async with session.post(
                        url=cls.url,
                        json=[key],
                        headers={
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": cls._next_actions["getSignedUrl"],
                            "referer": cls.url
                        }
                ) as response:
                    await raise_for_status(response)
                    text = await response.text()
                    line = next(filter(lambda x: x.startswith("1:"), text.split("\n")), "")
                    if not line:
                        raise Exception("Failed to get download URL")

                    chunk = json.loads(line[2:])
                    if not chunk.get("success"):
                        raise Exception("Failed to get download URL")
                    image_url = chunk.get("data", {}).get("url")
                    uploaded_file = {
                        "name": key,
                        "contentType": file_type,
                        "url": image_url
                    }
                debug.log(f"Uploaded image to: {image_url}")
                ImagesCache[image_hash] = uploaded_file
                files.append(uploaded_file)
        return files

    @classmethod
    def read_args(cls, args: dict = {}):
        cache_file = cls.get_cache_file()
        if not args and cache_file.exists():
            try:
                with cache_file.open("r") as f:
                    args = json.load(f)
            except json.JSONDecodeError:
                debug.log(f"Cache file {cache_file} is corrupted, removing it.")
                cache_file.unlink()
                args = None
        return args

    @classmethod
    async def get_quota(cls, **kwargs):
        args = cls.read_args()
        if not args:
            raise MissingAuthError("No authentication arguments found.")
        return {key: len(value) if value else 0 for key, value in args.items()}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: JsonConversation = None,
        media: MediaListType = None,
        proxy: str = None,
        timeout: int = None,
        **kwargs
    ) -> AsyncResult:
        prompt = get_last_user_message(messages)
        cache_file = cls.get_cache_file()
        args = cls.read_args(kwargs.get("lmarena_args", {}))
        grecaptcha = kwargs.pop("grecaptcha", "")
        _need_clear_cookies = False
        for _ in range(2):
            if args:
                pass
            elif has_nodriver:
                args, grecaptcha = await cls.get_args_from_nodriver(proxy, _need_clear_cookies)
            else:
                raise MissingRequirementsError("No auth file found and nodriver is not available.")

            if not cls._models_loaded:
                # change to async
                await cls.get_models_async()

            def get_mode_id(_model):
                model_id = None
                # if not model:
                #     model = cls.default_model
                if _model in cls.model_aliases:
                    _model = cls.model_aliases[_model]
                if _model in cls.text_models:
                    model_id = cls.text_models[_model]
                elif _model in cls.image_models:
                    model_id = cls.image_models[_model]
                elif _model in cls.video_models:
                    model_id = cls.video_models[_model]
                elif _model:
                    raise ModelNotFoundError(f"Model '{_model}' is not supported by LMArena provider.")
                return model_id

            modelA:str = model
            modelB:str = kwargs.get("modelB", "")
            modelAId = get_mode_id(modelA)
            modelBId = get_mode_id(modelB) if modelB else None
            if modelAId and modelBId:
                mode = "side-by-side"
            elif modelAId:
                mode = "direct"
            else:
                mode = "battle"
            if conversation and getattr(conversation, "evaluationSessionId", None):
                url = cls.post_to_evaluation.format(id=conversation.evaluationSessionId)
                evaluationSessionId = conversation.evaluationSessionId
            else:
                url = cls.create_evaluation
                evaluationSessionId = str(uuid7())
            is_image_model = modelA in cls.image_models
            userMessageId = str(uuid7())
            modelAMessageId = str(uuid7())
            modelBMessageId = str(uuid7())
            if not grecaptcha and has_nodriver:
                debug.log("No grecaptcha token found, obtaining new one...")
                args, grecaptcha = await cls.get_grecaptcha(args, proxy)
            files = await cls.prepare_images(args, media)
            data = {
                "id": evaluationSessionId,
                "mode": mode,
                "userMessageId": userMessageId,
                "modelAMessageId": modelAMessageId,
                "userMessage": {
                    "content": prompt,
                    "experimental_attachments": files,
                    "metadata": {}
                },
                "modality": "image" if is_image_model else "chat",
                "recaptchaV3Token": grecaptcha
            }
            if modelAId:
                data["modelAId"] = modelAId
            if modelBId:
                data["modelBId"] = modelBId
            if mode in ["side-by-side", "battle"]:
                data["modelBMessageId"] = modelBMessageId

            yield JsonRequest.from_dict(data)
            try:
                async with StreamSession(**args, timeout=timeout or 5 * 60) as session:
                    async with session.post(
                            url,
                            json=data,
                            proxy=proxy,
                    ) as response:
                        await raise_for_status(response)
                        args["cookies"] = merge_cookies(args["cookies"], response)
                        async for chunk in response.iter_lines():
                            line = chunk.decode()
                            yield PlainTextResponse(line)
                            if line.startswith("a0:"):
                                chunk = json.loads(line[3:])
                                if chunk == "hasArenaError":
                                    raise ModelNotFoundError("LMArena Beta encountered an error: hasArenaError")
                                yield chunk
                            elif line.startswith("b0:"):
                                ...
                            elif line.startswith("ag:"):
                                chunk = json.loads(line[3:])
                                yield Reasoning(chunk)
                            elif (line.startswith("a2:") or line.startswith("b2:")) and line == 'a2:[{"type":"heartbeat"}]':
                                # 'a2:[{"type":"heartbeat"}]'
                                continue
                            elif line.startswith("a2:"):
                                chunk = json.loads(line[3:])
                                __images = [image.get("image") for image in chunk if image.get("image")]
                                if __images:
                                    yield ImageResponse(__images, prompt, {"model": modelA})

                            elif line.startswith("b2:"):
                                chunk = json.loads(line[3:])
                                __images = [image.get("image") for image in chunk if image.get("image")]
                                if __images:
                                    yield ImageResponse(__images, prompt, {"model": modelB})

                            elif line.startswith("ad:"):
                                yield JsonConversation(evaluationSessionId=evaluationSessionId)
                                finish = json.loads(line[3:])
                                if "finishReason" in finish:
                                    yield FinishReason(finish["finishReason"])
                                if "usage" in finish:
                                    yield Usage(**finish["usage"])
                            elif line.startswith("bd:"):
                                ...
                            elif line.startswith("a3:"):
                                raise RuntimeError(f"LMArena: {json.loads(line[3:])}")
                            elif line.startswith("b3:"):
                                ...
                            else:
                                debug.log(f"LMArena: Unknown line prefix: {line[:2]}")
                break
            except (CloudflareError, MissingAuthError) as error:
                args = None
                debug.error(error)
                debug.log(f"{cls.__name__}: Cloudflare error")
                continue
            except RateLimitError as error:
                args = None
                _need_clear_cookies = True
                debug.error(error)
                continue
            except:
                raise
        if args:
            debug.log("Save args to cache file:", str(cache_file))
            with cache_file.open("w") as f:
                f.write(json.dumps(args))

def get_content_type(url: str) -> str:
    if url.endswith(".webp"):
        return "image/webp"
    elif url.endswith(".png"):
        return "image/png"
    elif url.endswith(".jpg") or url.endswith(".jpeg"):
        return "image/jpeg"
    else:
        return "application/octet-stream"
