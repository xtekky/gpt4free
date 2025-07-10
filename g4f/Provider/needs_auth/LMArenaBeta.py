from __future__ import annotations

import time
import uuid
import json
import asyncio

from ...typing import AsyncResult, Messages
from ...requests import StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies
from ...requests import DEFAULT_HEADERS, has_nodriver
from ...errors import ModelNotFoundError
from ...providers.response import FinishReason, Usage, JsonConversation, ImageResponse
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin,AuthFileMixin
from ..helper import get_last_user_message
from ... import debug

models = [
    {"id":"7a55108b-b997-4cff-a72f-5aa83beee918","publicName":"gemini-2.0-flash-001","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"f44e280a-7914-43ca-a25d-ecfcc5d48d09","publicName":"claude-3-5-sonnet-20241022","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"bd2c8278-af7a-4ec3-84db-0a426c785564","publicName":"grok-3-preview-02-24","organization":"xai","provider":"xaiPrivate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"9513524d-882e-4350-b31e-e4584440c2c8","publicName":"chatgpt-4o-latest-20250326","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"ce2092c1-28d4-4d42-a1e0-6b061dfe0b20","publicName":"gemini-2.5-flash-preview-05-20","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"49bd7403-c7fd-4d91-9829-90a91906ad6c","publicName":"llama-4-maverick-03-26-experimental","organization":"meta","provider":"meta","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"14e9311c-94d2-40c2-8c54-273947e208b0","publicName":"gpt-4.1-2025-04-14","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"885976d3-d178-48f5-a3f4-6e13e0718872","publicName":"qwq-32b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"aba0d185-6e8d-4cec-9933-20bc2ca3112a","publicName":"folsom-exp-v1.5","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"69bbf7d4-9f44-447e-a868-abc4f7a31810","publicName":"gemini-2.0-flash-preview-image-generation","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"c5a11495-081a-4dc6-8d9a-64a4fd6f7bbc","publicName":"claude-3-7-sonnet-20250219","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"ee116d12-64d6-48a8-88e5-b2d06325cdd2","publicName":"claude-opus-4-20250514","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"789e245f-eafe-4c72-b563-d135e93988fc","publicName":"gemma-3-27b-it","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"b3064d9c-eb58-4f46-b178-5c8a08724dc7","publicName":"stephen","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"f6fbf06c-532c-4c8a-89c7-f3ddcfb34bd1","publicName":"claude-3-5-haiku-20241022","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"7699c8d4-0742-42f9-a117-d10e84688dab","publicName":"grok-3-mini-beta","organization":"xai","provider":"xaiPublic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0f785ba1-efcb-472d-961e-69f7b251c7e3","publicName":"command-a-03-2025","organization":"cohere","provider":"cohere","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"30ab90f5-e020-4f83-aff5-f750d2e78769","publicName":"deepseek-r1-0528","organization":"deepseek","provider":"deepseek","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"6a5437a7-c786-467b-b701-17b0bc8c8231","publicName":"gpt-4.1-mini-2025-04-14","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"a14546b5-d78d-4cf6-bb61-ab5b8510a9d6","publicName":"amazon.nova-pro-v1:0","organization":"amazon","provider":"amazonBedrock","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"c680645e-efac-4a81-b0af-da16902b2541","publicName":"o3-mini","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"be98fcfd-345c-4ae1-9a82-a19123ebf1d2","publicName":"claude-3-7-sonnet-20250219-thinking-32k","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"cb0f1e24-e8e9-4745-aabc-b926ffde7475","publicName":"o3-2025-04-16","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"f1102bbf-34ca-468f-a9fc-14bcf63f315b","publicName":"o4-mini-2025-04-16","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"7fff29a7-93cc-44ab-b685-482c55ce4fa6","publicName":"gemini-2.5-flash-preview-04-17","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"27b9f8c6-3ee1-464a-9479-a8b3c2a48fd4","publicName":"mistral-medium-2505","organization":"mistral","provider":"mistral","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"51ad1d79-61e2-414c-99e3-faeb64bb6b1b","publicName":"imagen-3.0-generate-002","organization":"google","provider":"googleVertex","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"2f5253e4-75be-473c-bcfc-baeb3df0f8ad","publicName":"deepseek-v3-0324","organization":"deepseek","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"6e855f13-55d7-4127-8656-9168a9f4dcc0","publicName":"gpt-image-1","organization":"openai","provider":"customOpenai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"34ee5a83-8d85-4d8b-b2c1-3b3413e9ed98","publicName":"ideogram-v2","organization":"Ideogram","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"17e31227-36d7-4a7a-943a-7ebffa3a00eb","publicName":"photon","organization":"luma-ai","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"ac44dd10-0666-451c-b824-386ccfea7bcc","publicName":"claude-sonnet-4-20250514","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"68e498cb-a1b3-45fa-ae84-1b746d48652f","publicName":"X-preview","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"9a066f6a-7205-4325-8d0b-d81cc4b049c0","publicName":"qwen3-30b-a3b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"fe8003fc-2e5d-4a3f-8f07-c1cff7ba0159","publicName":"qwen-max-2025-01-25","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"2595a594-fa54-4299-97cd-2d7380d21c80","publicName":"qwen3-235b-a22b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"dcbd7897-5a37-4a34-93f1-76a24c7bb028","publicName":"llama-3.3-70b-instruct","organization":"meta","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0337ee08-8305-40c0-b820-123ad42b60cf","publicName":"gemini-2.5-pro-preview-05-06","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"5b5ad048-73b6-4cc2-a27f-2d2c2c2379a7","publicName":"glm-4-air-250414","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"bb97bc68-131c-4ea4-a59e-03a6252de0d2","publicName":"dall-e-3","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"b70ab012-18e7-4d6f-a887-574e05de6c20","publicName":"recraft-v3","organization":"Recraft","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"eb5da04f-9b28-406b-bf06-4539158c66ef","publicName":"anonymous-bot-0514","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"1a400d9a-f61c-4bc2-89b4-a9b7e77dff12","publicName":"qwen3-235b-a22b-no-thinking","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"20ba260d-509a-422d-a35d-9b0bd8bcc258","publicName":"hunyuan-large-vision","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"2e1af1cb-8443-4f3e-8d60-113992bfb491","publicName":"hunyuan-turbos-20250416","organization":"tencent","provider":"tencent","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"5055d347-5ae4-4bc5-9496-0a2b42f0c331","publicName":"flux-kontext-pro","organization":"BFL","provider":"bfl","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"6c632456-c58e-44f7-b3f4-657d8da656fd","publicName":"cobalt-exp-beta-v14","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"c8835275-3999-4660-a8e9-ecf7bbf11e67","publicName":"cobalt-exp-beta-v13","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"24c05961-494b-458b-b259-2eb687a637c9","publicName":"imagen-4.0-generate-preview-05-20","organization":"google","provider":"googleVertex","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"39b185cb-aba9-4232-99ea-074883a5ccd4","publicName":"stephen-v2","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"e2d9d353-6dbe-4414-bf87-bd289d523726","publicName":"gemini-2.5-pro-preview-06-05","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"00dbed2a-3632-4119-8c26-b13ca25cf1a9","publicName":"stephen-vision","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"4653dded-a46b-442a-a8fe-9bb9730e2453","publicName":"claude-sonnet-4-20250514-thinking-32k","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"3b5e9593-3dc0-4492-a3da-19784c4bde75","publicName":"claude-opus-4-20250514-thinking-16k","organization":"anthropic","provider":"anthropic","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}}
]

text_models = {model["publicName"]: model["id"] for model in models if "text" in model["capabilities"]["outputCapabilities"]}
image_models = {model["publicName"]: model["id"] for model in models if "image" in model["capabilities"]["outputCapabilities"]}

class LMArenaBeta(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    label = "LMArena Beta"
    url = "https://beta.lmarena.ai"
    api_endpoint = "https://beta.lmarena.ai/api/stream/create-evaluation"
    working = True
    active_by_default = has_nodriver

    default_model = list(text_models.keys())[0]
    models = list(text_models) + list(image_models)
    image_models = list(image_models)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: JsonConversation = None,
        proxy: str = None,
        timeout: int = None,
        **kwargs
    ) -> AsyncResult:
        cache_file = cls.get_cache_file()
        if cache_file.exists() and cache_file.stat().st_mtime > time.time() - 60 * 30:
            with cache_file.open("r") as f:
                args = json.load(f)
        elif has_nodriver:
            try:
                async def callback(page):
                    while not await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0'):
                        await asyncio.sleep(1)
                    while not await page.evaluate('document.querySelector(\'textarea[name="text"]\')'):
                        await asyncio.sleep(1)
                args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback)
            except (RuntimeError, FileNotFoundError) as e:
                debug.log(f"Nodriver is not available:", e)
                args = {"headers": DEFAULT_HEADERS, "cookies": {}, "impersonate": "chrome"}
        else:
            args = {"headers": DEFAULT_HEADERS, "cookies": {}, "impersonate": "chrome"}

        # Build the JSON payload
        is_image_model = model in image_models
        if not model:
            model = cls.default_model
        if model in image_models:
            model = image_models[model]
        elif model in text_models:
            model = text_models[model]
        elif model in cls.model_aliases:
            model = cls.model_aliases[model]
            debug.log(f"Using model alias: {model}")
        else:
            raise ModelNotFoundError(f"Model '{model}' is not supported by LMArena Beta.")
        userMessageId = str(uuid.uuid4())
        modelAMessageId = str(uuid.uuid4())
        evaluationSessionId = str(uuid.uuid4())
        prompt = get_last_user_message(messages)
        data = {
            "id": evaluationSessionId,
            "mode": "direct",
            "modelAId": model,
            "userMessageId": userMessageId,
            "modelAMessageId": modelAMessageId,
            "messages": [
                {
                    "id": userMessageId,
                    "role": "user",
                    "content": prompt,
                    "experimental_attachments": [],
                    "parentMessageIds": [] if conversation is None else conversation.message_ids,
                    "participantPosition": "a",
                    "modelId": None,
                    "evaluationSessionId": evaluationSessionId,
                    "status": "pending",
                    "failureReason": None
                },
                {
                    "id": modelAMessageId,
                    "role": "assistant",
                    "content": "",
                    "experimental_attachments": [],
                    "parentMessageIds": [userMessageId],
                    "participantPosition": "a",
                    "modelId": model,
                    "evaluationSessionId": evaluationSessionId,
                    "status": "pending",
                    "failureReason": None
                }
            ],
            "modality": "image" if is_image_model else "chat"
        }
        async with StreamSession(**args, timeout=timeout) as session:
            async with session.post(
                cls.api_endpoint,
                json=data,
                proxy=proxy
            ) as response:
                await raise_for_status(response)
                args["cookies"] = merge_cookies(args["cookies"], response)
                async for chunk in response.iter_lines():
                    line = chunk.decode()
                    if line.startswith("af:"):
                        yield JsonConversation(message_ids=[modelAMessageId])
                    elif line.startswith("a0:"):
                        chunk = json.loads(line[3:])
                        if chunk == "hasArenaError":
                            raise ModelNotFoundError("LMArena Beta encountered an error: hasArenaError")
                        yield chunk
                    elif line.startswith("a2:"):
                        yield ImageResponse([image.get("image") for image in json.loads(line[3:])], prompt)
                    elif line.startswith("ad:"):
                        finish = json.loads(line[3:])
                        if "finishReason" in finish:
                            yield FinishReason(finish["finishReason"])
                        if "usage" in finish:
                            yield Usage(**finish["usage"])

        # Save the args to cache file
        with cache_file.open("w") as f:
            json.dump(args, f)
