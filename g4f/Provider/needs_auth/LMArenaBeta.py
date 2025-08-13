from __future__ import annotations

import time
import uuid
import json
import asyncio
import os
import requests
import json

try:
    import curl_cffi
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies, has_nodriver
from ...errors import ModelNotFoundError, CloudflareError, MissingAuthError
from ...providers.response import FinishReason, Usage, JsonConversation, ImageResponse
from ...tools.media import merge_media
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin,AuthFileMixin
from ..helper import get_last_user_message
from ... import debug

models = [
    {"id":"43390b9c-cf16-4e4e-a1be-3355bb5b6d5e","publicName":"flux-1-kontext-pro","organization":"bfl","provider":"bfl","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"14e9311c-94d2-40c2-8c54-273947e208b0","publicName":"gpt-4.1-2025-04-14","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"6e855f13-55d7-4127-8656-9168a9f4dcc0","publicName":"gpt-image-1","organization":"openai","provider":"customOpenai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"e2d9d353-6dbe-4414-bf87-bd289d523726","publicName":"gemini-2.5-pro","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"ee116d12-64d6-48a8-88e5-b2d06325cdd2","publicName":"claude-opus-4-20250514","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0633b1ef-289f-49d4-a834-3d475a25e46b","publicName":"flux-1-kontext-max","organization":"bfl","provider":"bfl","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"f31bd225-2eee-4ba7-bd1e-15fe104b9b1c","publicName":"imagen-4.0-ultra-generate-preview-06-06","organization":"google","provider":"googleVertex","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"ce2092c1-28d4-4d42-a1e0-6b061dfe0b20","publicName":"gemini-2.5-flash","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"cb0f1e24-e8e9-4745-aabc-b926ffde7475","publicName":"o3-2025-04-16","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"9513524d-882e-4350-b31e-e4584440c2c8","publicName":"chatgpt-4o-latest-20250326","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"69f5d38a-45f5-4d3a-9320-b866a4035ed9","publicName":"mistral-small-3.1-24b-instruct-2503","organization":"mistral","provider":"mistral","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"e059dc1c-fdd1-4776-9671-ea65138eeda5","publicName":"seedance-1-lite-text-to-video","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"789e245f-eafe-4c72-b563-d135e93988fc","publicName":"gemma-3-27b-it","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"0caa771f-682f-484b-a7ad-70f7cb1f4183","publicName":"steve","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0f785ba1-efcb-472d-961e-69f7b251c7e3","publicName":"command-a-03-2025","organization":"cohere","provider":"cohere","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"6a5437a7-c786-467b-b701-17b0bc8c8231","publicName":"gpt-4.1-mini-2025-04-14","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"a14546b5-d78d-4cf6-bb61-ab5b8510a9d6","publicName":"amazon.nova-pro-v1:0","organization":"amazon","provider":"amazonBedrock","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"c680645e-efac-4a81-b0af-da16902b2541","publicName":"o3-mini","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"7699c8d4-0742-42f9-a117-d10e84688dab","publicName":"grok-3-mini-beta","organization":"xai","provider":"xaiPublic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"30ab90f5-e020-4f83-aff5-f750d2e78769","publicName":"deepseek-r1-0528","organization":"deepseek","provider":"deepseek","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"f1102bbf-34ca-468f-a9fc-14bcf63f315b","publicName":"o4-mini-2025-04-16","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"04ec9a17-c597-49df-acf0-963da275c246","publicName":"gemini-2.5-flash-lite-preview-06-17-thinking","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"d799a034-0ab6-48c1-817a-62e591143f39","publicName":"amazon-nova-experimental-chat-05-14","organization":"amazon","provider":"amazon","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"be98fcfd-345c-4ae1-9a82-a19123ebf1d2","publicName":"claude-3-7-sonnet-20250219-thinking-32k","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"f6fbf06c-532c-4c8a-89c7-f3ddcfb34bd1","publicName":"claude-3-5-haiku-20241022","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"27b9f8c6-3ee1-464a-9479-a8b3c2a48fd4","publicName":"mistral-medium-2505","organization":"mistral","provider":"mistral","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"51ad1d79-61e2-414c-99e3-faeb64bb6b1b","publicName":"imagen-3.0-generate-002","organization":"google","provider":"googleVertex","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"2f5253e4-75be-473c-bcfc-baeb3df0f8ad","publicName":"deepseek-v3-0324","organization":"deepseek","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"6337f479-2fc8-4311-a76b-8c957765cd68","publicName":"magistral-medium-2506","organization":"mistral","provider":"mistral","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"34ee5a83-8d85-4d8b-b2c1-3b3413e9ed98","publicName":"ideogram-v2","organization":"Ideogram","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"17e31227-36d7-4a7a-943a-7ebffa3a00eb","publicName":"photon","organization":"luma-ai","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"68e498cb-a1b3-45fa-ae84-1b746d48652f","publicName":"X-preview","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0888fb9f-6f1e-42ea-81d6-340f9541510a","publicName":"kling-2-master-image-to-video","capabilities":{"inputCapabilities":{"text":True,"image":{"requiresUpload":True}},"outputCapabilities":{"video":True}}},
    {"id":"b5ad3ab7-fc56-4ecd-8921-bd56b55c1159","publicName":"llama-4-maverick-17b-128e-instruct","organization":"meta","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"9a066f6a-7205-4325-8d0b-d81cc4b049c0","publicName":"qwen3-30b-a3b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"39b185cb-aba9-4232-99ea-074883a5ccd4","publicName":"stephen-v2","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"fe8003fc-2e5d-4a3f-8f07-c1cff7ba0159","publicName":"qwen-max-2025-01-25","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"2595a594-fa54-4299-97cd-2d7380d21c80","publicName":"qwen3-235b-a22b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"dcbd7897-5a37-4a34-93f1-76a24c7bb028","publicName":"llama-3.3-70b-instruct","organization":"meta","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"5b5ad048-73b6-4cc2-a27f-2d2c2c2379a7","publicName":"glm-4-air-250414","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"44882393-edb8-468f-9a39-d13d961ae364","publicName":"step1x-edit","capabilities":{"inputCapabilities":{"text":True,"image":{"requiresUpload":True}},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"ac44dd10-0666-451c-b824-386ccfea7bcc","publicName":"claude-sonnet-4-20250514","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"0bbd24e1-d18e-42fb-a87e-016e835acb08","publicName":"stonebloom","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"4f65970e-c258-4200-a424-bc2d158a0ee3","publicName":"kling-2.1-master","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"c5a11495-081a-4dc6-8d9a-64a4fd6f7bbc","publicName":"claude-3-7-sonnet-20250219","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"87e8d160-049e-4b4e-adc4-7f2511348539","publicName":"minimax-m1","organization":"minimax","provider":"minimax","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"bb97bc68-131c-4ea4-a59e-03a6252de0d2","publicName":"dall-e-3","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"b70ab012-18e7-4d6f-a887-574e05de6c20","publicName":"recraft-v3","organization":"Recraft","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"2820a312-8ff0-4b22-8271-c040efd83761","publicName":"step-1o-turbo-202506","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"2e354090-7df5-4578-9de8-85617481695e","publicName":"veo2","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"4653dded-a46b-442a-a8fe-9bb9730e2453","publicName":"claude-sonnet-4-20250514-thinking-32k","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"eb5da04f-9b28-406b-bf06-4539158c66ef","publicName":"anonymous-bot-0514","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"1a400d9a-f61c-4bc2-89b4-a9b7e77dff12","publicName":"qwen3-235b-a22b-no-thinking","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"896a3848-ae03-4651-963b-7d8f54b61ae8","publicName":"gemma-3n-e4b-it","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"9e8525b7-fe50-4e50-bf7f-ad1d3d205d3c","publicName":"flux-1.1-pro","organization":"Black Forest Labs","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"1fab3f3c-bb6d-4219-b1fa-969535f4f22b","publicName":"veo3","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"db141cc1-227f-4074-aeb9-d49db03329bc","publicName":"pika-2.2","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"3b5e9593-3dc0-4492-a3da-19784c4bde75","publicName":"claude-opus-4-20250514-thinking-16k","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"e3c9ea42-5f42-496b-bc80-c7e8ee5653cc","publicName":"stephen-vision-csfix","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"e652c45e-8699-4392-94f0-7834e7464137","publicName":"hailuo-02-standard","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"video":True}}},
    {"id":"bbad1d17-6aa5-4321-949c-d11fb6289241","publicName":"mistral-small-2506","organization":"mistral","provider":"mistral","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"f7e2ed7a-f0b9-40ef-853a-20036e747232","publicName":"ideogram-v3-quality","organization":"Ideogram","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"849f6833-8598-4329-b7a6-e765eb2a0c0d","publicName":"wolfstride","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"149619f1-f1d5-45fd-a53e-7d790f156f20","publicName":"grok-3-mini-high","organization":"xai","provider":"xaiPublic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"dd5e7509-3249-43b7-b3bf-6e158c97b1c1","publicName":"imagen-4.0-generate-preview-06-06","organization":"google","provider":"googleVertex","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"b222be23-bd55-4b20-930b-a30cc84d3afd","publicName":"gemini-2.5-pro-grounding","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"b9edb8e9-4e98-49e7-8aaf-ae67e9797a11","publicName":"grok-4-0709","organization":"xai","provider":"openrouter","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"33453cd6-8704-4b82-a7c0-12e8042ef1e3","publicName":"grok-3-search","organization":"xai","provider":"xaiSearch","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"c8711485-d061-4a00-94d2-26c31b840a3d","publicName":"ppl-sonar-pro-high","organization":"perplexity","provider":"perplexity","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"24145149-86c9-4690-b7c9-79c7db216e5c","publicName":"ppl-sonar-reasoning-pro-high","organization":"perplexity","provider":"perplexity","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"fbe08e9a-3805-4f9f-a085-7bc38e4b51d1","publicName":"o3-search","organization":"openai","provider":"openaiResponses","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"25bcb878-749e-49f4-ac05-de84d964bcee","publicName":"claude-opus-4-search","organization":"anthropic","provider":"anthropicSearch","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"14dbdb19-708f-4210-8e12-ce52b5c5296a","publicName":"api-gpt-4o-search","organization":"openai","provider":"openai","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"search":True}}},
    {"id":"0dde746c-3dbc-42be-b8f5-f38bd1595baa","publicName":"seedream-3","organization":"bytedance","provider":"replicate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"c889da11-70c3-4d15-a26b-6e3be4aa7fdf","publicName":"cresylux","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"e2969ebb-6450-4bc4-87c9-bbdcf95840da","publicName":"seededit-3.0","capabilities":{"inputCapabilities":{"text":True,"image":{"requiresUpload":True}},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"2e1af1cb-8443-4f3e-8d60-113992bfb491","publicName":"hunyuan-turbos-20250416","organization":"tencent","provider":"tencent","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"eb90ae46-a73a-4f27-be8b-40f090592c9a","publicName":"flux-1-kontext-dev","organization":"bfl","provider":"bfl","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"1efdf1f1-8879-493b-8e4a-14b61e978847","publicName":"ernie-x1-turbo-32k-preview","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"7a3626fc-4e64-4c9e-821f-b449a4b43b6a","publicName":"kimi-k2-0711-preview","organization":"moonshot","provider":"moonshot","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"a8af3f6a-3443-4af8-afec-8a3310f29545","publicName":"nettle","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"f22c1d4e-6ed1-421f-aecf-9149366a770f","publicName":"clownfish","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"7b1b3cfc-fde3-455c-b15b-81af55b44bec","publicName":"octopus","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"fea7b01b-a3c2-40c9-a577-49bbe62eefb9","publicName":"kraken-07152025-1","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"f69cd6f2-1269-4ccb-b927-91cff1624aac","publicName":"kraken-07152025-2","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"7a55108b-b997-4cff-a72f-5aa83beee918","publicName":"gemini-2.0-flash-001","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"cbdb6d5d-d87c-4b44-832b-08b8ba6421dc","publicName":"folsom-07152025-1","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"c9ef2f9e-5410-4d3b-b65e-b5af317c9afd","publicName":"bagel","organization":"bytedance","provider":"customReplicate","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"f44e280a-7914-43ca-a25d-ecfcc5d48d09","publicName":"claude-3-5-sonnet-20241022","organization":"anthropic","provider":"googleVertexAnthropic","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"bd2c8278-af7a-4ec3-84db-0a426c785564","publicName":"grok-3-preview-02-24","organization":"xai","provider":"xaiPrivate","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"20ba260d-509a-422d-a35d-9b0bd8bcc258","publicName":"hunyuan-large-vision","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"c28823c1-40fd-4eaf-9825-e28f11d1f8b2","publicName":"llama-4-scout-17b-16e-instruct","organization":"meta","provider":"fireworks","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"text":True}}},
    {"id":"49bd7403-c7fd-4d91-9829-90a91906ad6c","publicName":"llama-4-maverick-03-26-experimental","organization":"meta","provider":"meta","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"885976d3-d178-48f5-a3f4-6e13e0718872","publicName":"qwq-32b","organization":"alibaba","provider":"alibaba","capabilities":{"inputCapabilities":{"text":True},"outputCapabilities":{"text":True}}},
    {"id":"69bbf7d4-9f44-447e-a868-abc4f7a31810","publicName":"gemini-2.0-flash-preview-image-generation","organization":"google","provider":"google","capabilities":{"inputCapabilities":{"text":True,"image":True},"outputCapabilities":{"image":{"aspectRatios":["1:1"]}}}},
    {"id":"e413081f-b7c7-4212-8a21-5c8dbf865c21","publicName":"seedance-1-lite-image-to-video","capabilities":{"inputCapabilities":{"text":True,"image":{"requiresUpload":True}},"outputCapabilities":{"video":True}}}
]
text_models = {model["publicName"]: model["id"] for model in models if "text" in model["capabilities"]["outputCapabilities"]}
image_models = {model["publicName"]: model["id"] for model in models if "image" in model["capabilities"]["outputCapabilities"]}
vision_models = [model["publicName"] for model in models if "image" in model["capabilities"]["inputCapabilities"]]

class LMArenaBeta(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    label = "LMArena (New)"
    url = "https://lmarena.ai"
    share_url = None
    api_endpoint = "https://lmarena.ai/api/stream/create-evaluation"
    working = True
    active_by_default = True

    default_model = list(text_models.keys())[0]
    models = list(text_models) + list(image_models)
    model_aliases = {
        "flux-kontext": "flux-1-kontext-pro",
    }
    image_models = image_models
    text_models = text_models
    vision_models = vision_models
    looked = False
    _models_loaded = False

    @classmethod
    def get_models(cls) -> list[str]:
        if not cls._models_loaded and has_curl_cffi:
            cache_file = cls.get_cache_file()
            args = {}
            if cache_file.exists():
                try:
                    with cache_file.open("r") as f:
                        args = json.load(f)
                except json.JSONDecodeError:
                    debug.log(f"Cache file {cache_file} is corrupted, removing it.")
                    cache_file.unlink()
                    args = {}
                if not args:
                    return cls.models
                response = curl_cffi.get(f"{cls.url}/?mode=direct", **args)
                if response.ok:
                    for line in response.text.splitlines():
                        if "initialModels" in line:
                            line = line.split("initialModels", maxsplit=1)[-1].split("initialModelAId")[0][3:-3].replace('\\', '')
                            models = json.loads(line)
                            cls.text_models = {model["publicName"]: model["id"] for model in models if "text" in model["capabilities"]["outputCapabilities"]}
                            cls.image_models = {model["publicName"]: model["id"] for model in models if "image" in model["capabilities"]["outputCapabilities"]}
                            cls.vision_models = [model["publicName"] for model in models if "image" in model["capabilities"]["inputCapabilities"]]
                            cls.models = list(cls.text_models) + list(cls.image_models)
                            cls.default_model = list(cls.text_models.keys())[0]
                            cls._models_loaded = True
                            break
                else:
                    debug.log(f"Failed to load models from {cls.url}: {response.status_code} {response.reason}")
        return cls.models

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
        if cls.share_url is None:
            cls.share_url = os.getenv("G4F_SHARE_URL")
        prompt = get_last_user_message(messages)
        cache_file = cls.get_cache_file()
        args = None
        if cache_file.exists():
            try:
                with cache_file.open("r") as f:
                    args = json.load(f)
            except json.JSONDecodeError:
                debug.log(f"Cache file {cache_file} is corrupted, removing it.")
                cache_file.unlink()
                args = None
        for _ in range(2):
            if args:
                pass
            elif has_nodriver or cls.share_url is None:
                async def callback(page):
                    await page.find("Ask anything…", 120)
                    button = await page.find("Accept Cookies")
                    if button:
                        await button.click()
                    else:
                        debug.log("No 'Accept Cookies' button found, skipping.")
                    if not await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0'):
                        debug.log("No authentication cookie found, trying to authenticate.")
                        await page.select('#cf-turnstile', 300)
                        debug.log("Found Element: 'cf-turnstile'")
                        await asyncio.sleep(3)
                        for _ in range(3):
                            size = None
                            for idx in range(15):
                                size = await page.js_dumps('document.getElementById("cf-turnstile")?.getBoundingClientRect()||{}')
                                debug.log("Found size:", {size.get("x"), size.get("y")})
                                if "x" not in size:
                                    break
                                await page.flash_point(size.get("x") + idx * 2, size.get("y") + idx * 2)
                                await page.mouse_click(size.get("x") + idx * 2, size.get("y") + idx * 2)
                                await asyncio.sleep(1)
                            if "x" not in size:
                                break
                        debug.log("Clicked on the turnstile.")
                    while not await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0'):
                        await asyncio.sleep(1)
                    while not await page.evaluate('document.querySelector(\'textarea\')'):
                        await asyncio.sleep(1)
                args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback)
                with cache_file.open("w") as f:
                    json.dump(args, f)
            elif not cls.looked:
                cls.looked = True
                try:
                    debug.log("No cache file found, trying to fetch from share URL.")
                    response = requests.get(cls.share_url, params={
                        "prompt": prompt,
                        "model": model,
                        "provider": cls.__name__
                    })
                    raise_for_status(response)
                    text, *args = response.text.split("\n" * 10 + "<!--", 1)
                    if args:
                        debug.log("Save args to cache file:", str(cache_file))
                        with cache_file.open("w") as f:
                            f.write(args[0].strip())
                    yield text
                finally:
                    cls.looked = False
                return

            if not cls._models_loaded:
                cls.get_models()
            is_image_model = model in image_models
            if not model:
                model = cls.default_model
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            if model in cls.text_models:
                model_id = cls.text_models[model]
            elif model in cls.image_models:
                model_id = cls.image_models[model]
            else:
                raise ModelNotFoundError(f"Model '{model}' is not supported by LMArena Beta.")

            userMessageId = str(uuid.uuid4())
            modelAMessageId = str(uuid.uuid4())
            evaluationSessionId = str(uuid.uuid4())
            data = {
                "id": evaluationSessionId,
                "mode": "direct",
                "modelAId": model_id,
                "userMessageId": userMessageId,
                "modelAMessageId": modelAMessageId,
                "messages": [
                    {
                        "id": userMessageId,
                        "role": "user",
                        "content": prompt,
                        "experimental_attachments": [
                            {
                                "name": name or os.path.basename(url),
                                "contentType": get_content_type(url),
                                "url": url
                            }
                            for url, name in list(merge_media(media, messages))
                            if isinstance(url, str) and url.startswith("https://")
                        ],
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
            try:
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
                break
            except (CloudflareError, MissingAuthError):
                args = None
                debug.log(f"{cls.__name__}: Cloudflare error")
                continue
        if args and os.getenv("G4F_SHARE_AUTH"):
            yield "\n" * 10
            yield "<!--"
            yield json.dumps(args)
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

async def switch_to_frame(browser, frame_id):
    """
    change iframe
    let iframe = document.querySelector("YOUR_IFRAME_SELECTOR")
    let iframe_tab = iframe.contentWindow.document.body;
    """
    iframe_tab = next(
        filter(
        lambda x: str(x.target.target_id) == str(frame_id), browser.targets
        )
    )
    return iframe_tab