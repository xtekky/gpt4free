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

try:
    import nodriver
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies
from ...errors import ModelNotFoundError, CloudflareError, MissingAuthError, MissingRequirementsError
from ...providers.response import FinishReason, Usage, JsonConversation, ImageResponse, Reasoning
from ...tools.media import merge_media
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin,AuthFileMixin
from ..helper import get_last_user_message
from ... import debug

models = [
    {'id': '983bc566-b783-4d28-b24c-3c8b08eb1086', 'publicName': 'gpt-5-high', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'e884e85b-c998-44d8-b38d-db42a300a318', 'publicName': 'gemini-2.5-flash-image-preview (nano-banana)',
     'organization': 'google', 'provider': 'google-genai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': True, 'requiresUpload': False}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '4b11c78c-08c8-461c-938e-5fc97d56a40d', 'publicName': 'gpt-5-chat', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '2ec9f1a6-126f-4c65-a102-15ac401dcea4', 'publicName': 'imagen-4.0-generate-preview-06-06',
     'organization': 'google', 'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'eb90ae46-a73a-4f27-be8b-40f090592c9a', 'publicName': 'flux-1-kontext-dev', 'organization': 'bfl',
     'provider': 'bfl', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False}},
                                         'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'ee116d12-64d6-48a8-88e5-b2d06325cdd2', 'publicName': 'claude-opus-4-20250514', 'organization': 'anthropic',
     'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'e2d9d353-6dbe-4414-bf87-bd289d523726', 'publicName': 'gemini-2.5-pro', 'organization': 'google',
     'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'ce2092c1-28d4-4d42-a1e0-6b061dfe0b20', 'publicName': 'gemini-2.5-flash', 'organization': 'google',
     'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'cb0f1e24-e8e9-4745-aabc-b926ffde7475', 'publicName': 'o3-2025-04-16', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9513524d-882e-4350-b31e-e4584440c2c8', 'publicName': 'chatgpt-4o-latest-20250326', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'b6a05a03-88db-4d2b-bb10-41ddea0f27d6', 'publicName': 'catalina',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '0f785ba1-efcb-472d-961e-69f7b251c7e3', 'publicName': 'command-a-03-2025', 'organization': 'cohere',
     'provider': 'cohere', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '6a5437a7-c786-467b-b701-17b0bc8c8231', 'publicName': 'gpt-4.1-mini-2025-04-14', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'a14546b5-d78d-4cf6-bb61-ab5b8510a9d6', 'publicName': 'amazon.nova-pro-v1:0', 'organization': 'amazon',
     'provider': 'amazonBedrock',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c680645e-efac-4a81-b0af-da16902b2541', 'publicName': 'o3-mini', 'organization': 'openai',
     'provider': 'openai', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '7699c8d4-0742-42f9-a117-d10e84688dab', 'publicName': 'grok-3-mini-beta', 'organization': 'xai',
     'provider': 'xaiPublic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '79af9ac3-f361-40db-85ea-f0bdbdc76f84', 'publicName': 'phantom-0821-1',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'f1102bbf-34ca-468f-a9fc-14bcf63f315b', 'publicName': 'o4-mini-2025-04-16', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '04ec9a17-c597-49df-acf0-963da275c246', 'publicName': 'gemini-2.5-flash-lite-preview-06-17-thinking',
     'organization': 'google', 'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'd799a034-0ab6-48c1-817a-62e591143f39', 'publicName': 'amazon-nova-experimental-chat-05-14',
     'organization': 'amazon', 'provider': 'amazon',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'be98fcfd-345c-4ae1-9a82-a19123ebf1d2', 'publicName': 'claude-3-7-sonnet-20250219-thinking-32k',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'f6fbf06c-532c-4c8a-89c7-f3ddcfb34bd1', 'publicName': 'claude-3-5-haiku-20241022',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '27b9f8c6-3ee1-464a-9479-a8b3c2a48fd4', 'publicName': 'mistral-medium-2505', 'organization': 'mistral',
     'provider': 'mistral',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '51ad1d79-61e2-414c-99e3-faeb64bb6b1b', 'publicName': 'imagen-3.0-generate-002', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '2f5253e4-75be-473c-bcfc-baeb3df0f8ad', 'publicName': 'deepseek-v3-0324', 'organization': 'deepseek',
     'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '36e4900d-5df2-46e1-9bd3-ef4028ab50b0', 'publicName': 'velocilux',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '6337f479-2fc8-4311-a76b-8c957765cd68', 'publicName': 'magistral-medium-2506', 'organization': 'mistral',
     'provider': 'mistral',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '34ee5a83-8d85-4d8b-b2c1-3b3413e9ed98', 'publicName': 'ideogram-v2', 'organization': 'Ideogram',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '17e31227-36d7-4a7a-943a-7ebffa3a00eb', 'publicName': 'photon', 'organization': 'luma-ai',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'b5ad3ab7-fc56-4ecd-8921-bd56b55c1159', 'publicName': 'llama-4-maverick-17b-128e-instruct',
     'organization': 'meta', 'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9a066f6a-7205-4325-8d0b-d81cc4b049c0', 'publicName': 'qwen3-30b-a3b', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '39b185cb-aba9-4232-99ea-074883a5ccd4', 'publicName': 'stephen-v2',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '2595a594-fa54-4299-97cd-2d7380d21c80', 'publicName': 'qwen3-235b-a22b', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'dcbd7897-5a37-4a34-93f1-76a24c7bb028', 'publicName': 'llama-3.3-70b-instruct', 'organization': 'meta',
     'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '34c89088-1c15-4cff-96fd-52ced7a4d5a9', 'publicName': 'cogitolux',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'ac44dd10-0666-451c-b824-386ccfea7bcc', 'publicName': 'claude-sonnet-4-20250514',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '38abc02f-5cf2-49d1-a243-b2eb75ca3cc8', 'publicName': 'potato',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c5a11495-081a-4dc6-8d9a-64a4fd6f7bbc', 'publicName': 'claude-3-7-sonnet-20250219',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '87e8d160-049e-4b4e-adc4-7f2511348539', 'publicName': 'minimax-m1', 'organization': 'minimax',
     'provider': 'minimax',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'bb97bc68-131c-4ea4-a59e-03a6252de0d2', 'publicName': 'dall-e-3', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'b70ab012-18e7-4d6f-a887-574e05de6c20', 'publicName': 'recraft-v3', 'organization': 'Recraft',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'a8d1d310-e485-4c50-8f27-4bff18292a99', 'publicName': 'qwen3-30b-a3b-instruct-2507',
     'organization': 'alibaba', 'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '4653dded-a46b-442a-a8fe-9bb9730e2453', 'publicName': 'claude-sonnet-4-20250514-thinking-32k',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'eb5da04f-9b28-406b-bf06-4539158c66ef', 'publicName': 'anonymous-bot-0514',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '1a400d9a-f61c-4bc2-89b4-a9b7e77dff12', 'publicName': 'qwen3-235b-a22b-no-thinking',
     'organization': 'alibaba', 'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '896a3848-ae03-4651-963b-7d8f54b61ae8', 'publicName': 'gemma-3n-e4b-it', 'organization': 'google',
     'provider': 'google', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9e8525b7-fe50-4e50-bf7f-ad1d3d205d3c', 'publicName': 'flux-1.1-pro', 'organization': 'Black Forest Labs',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '3b5e9593-3dc0-4492-a3da-19784c4bde75', 'publicName': 'claude-opus-4-20250514-thinking-16k',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'e3c9ea42-5f42-496b-bc80-c7e8ee5653cc', 'publicName': 'stephen-vision-csfix',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'bbad1d17-6aa5-4321-949c-d11fb6289241', 'publicName': 'mistral-small-2506', 'organization': 'mistral',
     'provider': 'mistral',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'f7e2ed7a-f0b9-40ef-853a-20036e747232', 'publicName': 'ideogram-v3-quality', 'organization': 'Ideogram',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '149619f1-f1d5-45fd-a53e-7d790f156f20', 'publicName': 'grok-3-mini-high', 'organization': 'xai',
     'provider': 'xaiPublic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'b9edb8e9-4e98-49e7-8aaf-ae67e9797a11', 'publicName': 'grok-4-0709', 'organization': 'xai',
     'provider': 'openrouter',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '24145149-86c9-4690-b7c9-79c7db216e5c', 'publicName': 'ppl-sonar-reasoning-pro-high',
     'organization': 'perplexity', 'provider': 'perplexity',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': '0dde746c-3dbc-42be-b8f5-f38bd1595baa', 'publicName': 'seedream-3', 'organization': 'bytedance',
     'provider': 'replicate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '812c93cc-5f88-4cff-b9ca-c11a26599b0e', 'publicName': 'qwen-max-2025-08-15', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '25bcb878-749e-49f4-ac05-de84d964bcee', 'publicName': 'claude-opus-4-search', 'organization': 'anthropic',
     'provider': 'anthropicSearch',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': 'b222be23-bd55-4b20-930b-a30cc84d3afd', 'publicName': 'gemini-2.5-pro-grounding', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': 'd079ef40-3b20-4c58-ab5e-243738dbada5', 'publicName': 'glm-4.5', 'organization': 'zai', 'provider': 'zai',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'fbe08e9a-3805-4f9f-a085-7bc38e4b51d1', 'publicName': 'o3-search', 'organization': 'openai',
     'provider': 'openaiResponses',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': 'c8711485-d061-4a00-94d2-26c31b840a3d', 'publicName': 'ppl-sonar-pro-high', 'organization': 'perplexity',
     'provider': 'perplexity',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': '96ae95fd-b70d-49c3-91cc-b58c7da1090b', 'publicName': 'claude-opus-4-1-20250805',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '68b6f90d-9dd5-4995-97d0-7ea13c0c82ba', 'publicName': 'Bailing-Lite-250220',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '51a47cc6-5ef9-4ac7-a59c-4009230d7564', 'publicName': 'gemini-2.5-pro-grounding-exp',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'ec3beb4b-7229-4232-bab9-670ee52dd711', 'publicName': 'gpt-oss-20b', 'organization': 'openai',
     'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '6fe1ec40-3219-4c33-b3e7-0e65658b4194', 'publicName': 'qwen-vl-max-2025-08-13', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '80caa6ac-05cd-4403-88e1-ef0164c8b1a8', 'publicName': 'veo3', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': '1b677c7e-49dd-4045-9ce0-d1aedcb9bbbc', 'publicName': 'veo3-fast', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': '08d8dcc6-2ab5-45ae-9bf1-353480f1f7ee', 'publicName': 'veo2', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': '7a3626fc-4e64-4c9e-821f-b449a4b43b6a', 'publicName': 'kimi-k2-0711-preview', 'organization': 'moonshot',
     'provider': 'moonshot',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'ee7cb86e-8601-4585-b1d0-7c7380f8f6f4', 'publicName': 'qwen3-235b-a22b-instruct-2507',
     'organization': 'alibaba', 'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '5b3383a9-6bca-4f71-8210-78895c9d84d5', 'publicName': 'ray2', 'organization': 'luma-ai', 'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                      'outputCapabilities': {'video': True}}},
    {'id': '6ee9f901-17b5-4fbe-9cc2-13c16497c23b', 'publicName': 'gpt-oss-120b', 'organization': 'openai',
     'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '5a3b3520-c87d-481f-953c-1364687b6e8f', 'publicName': 'lucid-origin', 'organization': 'leonardo-ai',
     'provider': 'leonardo-ai',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '6e855f13-55d7-4127-8656-9168a9f4dcc0', 'publicName': 'gpt-image-1', 'organization': 'openai',
     'provider': 'customOpenai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': True, 'requiresUpload': False}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '1ea13a81-93a7-4804-bcdd-693cd72e302d', 'publicName': 'step-3', 'organization': 'stepfun',
     'provider': 'stepfun',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'af033cbd-ec6c-42cc-9afa-e227fc12efe8', 'publicName': 'qwen3-coder-480b-a35b-instruct',
     'organization': 'Alibaba', 'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'f8aec69d-e077-4ed1-99be-d34f48559bbf', 'publicName': 'imagen-4.0-ultra-generate-preview-06-06',
     'organization': 'google', 'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'a071b843-0fc2-4fcf-b644-023509635452', 'publicName': 'veo3-audio', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': '7bfb254a-5d32-4ce2-b6dc-2c7faf1d5fe8', 'publicName': 'glm-4.5-air', 'organization': 'zai',
     'provider': 'zai', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '48fe3167-5680-4903-9ab5-2f0b9dc05815', 'publicName': 'nightride-on',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '638fb8b8-1037-4ee5-bfba-333392575a5d', 'publicName': 'EB45-vision',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c822ec98-38e9-4e43-a434-982eb534824f', 'publicName': 'nightride-on-v2',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '86d767b0-2574-4e47-a256-a22bcace9f56', 'publicName': 'grok-4-search', 'organization': 'xai',
     'provider': 'xaiSearch',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': '27035fb8-a25b-4ec9-8410-34be18328afd', 'publicName': 'mistral-medium-2508', 'organization': 'mistral',
     'provider': 'mistral',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '16b8e53a-cc7b-4608-a29a-20d4dac77cf2', 'publicName': 'qwen3-235b-a22b-thinking-2507',
     'organization': 'alibaba', 'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '217158f9-793e-4ffe-a197-6de9448432fc', 'publicName': 'ray2', 'organization': 'luma-ai', 'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': '69bbf7d4-9f44-447e-a868-abc4f7a31810', 'publicName': 'gemini-2.0-flash-preview-image-generation',
     'organization': 'google', 'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': True, 'requiresUpload': False}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'ba99b6cb-e981-48f4-a5be-ace516ee2731', 'publicName': 'hailuo-02-standard', 'organization': 'minimax',
     'provider': 'minimax',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'e705b65f-82cd-40cb-9630-d9e6ca92d06f', 'publicName': 'seedance-v1-pro', 'organization': 'bytedance',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'f1a5a6ab-e1b1-4247-88ac-49395291c1e3', 'publicName': 'not-a-new-model',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c15b93ed-e87b-467f-8f9f-d830fd7aa54d', 'publicName': 'lmarena-internal-test-only',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '0862885e-ef53-4d0d-b9c4-4c8f68f453ce', 'publicName': 'diffbot-small-xl', 'organization': 'diffbot',
     'provider': 'diffbot',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': 'ad6e8b44-d4e0-4544-85d4-ad05b83b0bf2', 'publicName': 'spuddle',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '43390b9c-cf16-4e4e-a1be-3355bb5b6d5e', 'publicName': 'flux-1-kontext-pro', 'organization': 'bfl',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False}},
                                         'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '30dfdea7-b4bc-4dab-8515-5d93917c7f4f', 'publicName': 'deepseek-v3.1', 'organization': 'deepseek',
     'provider': 'deepseek',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9fe82ee1-c84f-417f-b0e7-cab4ae4cf3f3', 'publicName': 'qwen-image-prompt-extend', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '86de5aea-fc0c-4c36-b65a-7afc443a32d2', 'publicName': 'pika-v2.2', 'organization': 'pika', 'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': '0633b1ef-289f-49d4-a834-3d475a25e46b', 'publicName': 'flux-1-kontext-max',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'f1a2eb6f-fc30-4806-9e00-1efd0d73cbc4', 'publicName': 'claude-opus-4-1-20250805-thinking-16k',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'e9dd5a96-c066-48b0-869f-eb762030b5ed', 'publicName': 'EB45-turbo',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9bbbca46-b6c2-4919-83a8-87ef1c559c4e', 'publicName': 'veo3-fast-audio', 'organization': 'google',
     'provider': 'googleVertex',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'cff3fc67-4207-4dff-967f-f4de61115836', 'publicName': 'deepseek-v3.1-thinking', 'organization': 'deepseek',
     'provider': 'deepseek',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '5fd3caa8-fe4c-41a5-a22c-0025b58f4b42', 'publicName': 'gpt-5-mini-high', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '9dab0475-a0cc-4524-84a2-3fd25aa8c768', 'publicName': 'glm-4.5v', 'organization': 'zai', 'provider': 'zai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '8afbc476-52af-4ebc-aa33-2ffdd7e19153', 'publicName': 'hailuo-02-pro', 'organization': 'minimax',
     'provider': 'minimax',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'd63b03fb-8bc8-4ed8-9a50-6ccb683ac2b1', 'publicName': 'kling-v2.1-master', 'organization': 'kling',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': '4c8dde6e-1b2c-45b9-91c3-413b2ceafffb', 'publicName': 'seedance-v1-lite', 'organization': 'bytedance',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                         'outputCapabilities': {'video': True}}},
    {'id': 'ea96cfc8-953a-4c3c-a229-1107c55b7479', 'publicName': 'kling-v2.1-standard', 'organization': 'kling',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                         'outputCapabilities': {'video': True}}},
    {'id': '4ddc4e52-2867-49b6-a603-5aab24a566ca', 'publicName': 'seedance-v1-pro', 'organization': 'bytedance',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                         'outputCapabilities': {'video': True}}},
    {'id': 'c3d0e5c8-f4b3-417a-8cb8-2ccf757d3869', 'publicName': 'sora', 'organization': 'openai',
     'provider': 'azureOpenAI',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'd942b564-191c-41c5-ae22-400a930a2cfe', 'publicName': 'claude-opus-4-1-search', 'organization': 'anthropic',
     'provider': 'anthropicSearch',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': '2dc249b3-98da-44b4-8d1e-6666346a8012', 'publicName': 'gpt-5-nano-high', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'f9b9f030-9ebc-4765-bf76-c64a82a72dfd', 'publicName': 'pika-v2.2', 'organization': 'pika', 'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                      'outputCapabilities': {'video': True}}},
    {'id': 'ee39672b-d216-4bf4-b639-00469c4f886d', 'publicName': 'qwen-image-edit', 'organization': 'alibaba',
     'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False, 'requiresUpload': True}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'd14d9b23-1e46-4659-b157-a3804ba7e2ef', 'publicName': 'gpt-5-search', 'organization': 'openai',
     'provider': 'openaiResponses',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}},
    {'id': '2e1af1cb-8443-4f3e-8d60-113992bfb491', 'publicName': 'hunyuan-turbos-20250416', 'organization': 'tencent',
     'provider': 'tencent',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '23848331-9f93-404f-85f0-3c3b4ece177e', 'publicName': 'mai-1-preview', 'organization': 'microsoft-ai',
     'provider': 'microsoftAi',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c382b9c6-d31b-488e-86c1-e84d8427eb30', 'publicName': 'hailuo-02-fast', 'organization': 'minimax',
     'provider': 'minimax', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                             'outputCapabilities': {'video': True}}},
    {'id': '13ce11ba-def2-4c80-a70b-b0b2c14d293e', 'publicName': 'seedance-v1-lite', 'organization': 'bytedance',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': 'e4e58f18-c04f-47cd-8d11-4b2ece7b617e', 'publicName': 'nano-banana',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': True, 'requiresUpload': False}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '0754baa1-ab91-42d0-ba74-522aa8e5b8e2', 'publicName': 'runway-gen4-turbo', 'organization': 'runway',
     'provider': 'replicate', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                               'outputCapabilities': {'video': True}}},
    {'id': 'efdb7e05-2091-4e88-af9e-4ea6168d2f85', 'publicName': 'kling-v2.1-master', 'organization': 'kling',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                         'outputCapabilities': {'video': True}}},
    {'id': 'e2969ebb-6450-4bc4-87c9-bbdcf95840da', 'publicName': 'seededit-3.0',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False, 'requiresUpload': True}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': 'f4809219-14a8-47fe-9705-8685085513e7', 'publicName': 'mochi-v1', 'organization': 'genmo', 'provider': 'fal',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': '32bff2df-00e6-409b-ad3f-bfbad87cc49f', 'publicName': 'hidream-e1.1',
     'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False, 'requiresUpload': True}},
                      'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}},
    {'id': '264e6e2f-b66a-4e27-a859-8145ff32d6f6', 'publicName': 'wan-v2.2-a14b', 'organization': 'alibaba',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}},
    {'id': '7a55108b-b997-4cff-a72f-5aa83beee918', 'publicName': 'gemini-2.0-flash-001', 'organization': 'google',
     'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '3a91bb37-39fb-471c-8aa2-a89b98d280d0', 'publicName': 'wan-v2.2-a14b', 'organization': 'alibaba',
     'provider': 'fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}},
                                         'outputCapabilities': {'video': True}}},
    {'id': 'f44e280a-7914-43ca-a25d-ecfcc5d48d09', 'publicName': 'claude-3-5-sonnet-20241022',
     'organization': 'anthropic', 'provider': 'googleVertexAnthropic',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'bd2c8278-af7a-4ec3-84db-0a426c785564', 'publicName': 'grok-3-preview-02-24', 'organization': 'xai',
     'provider': 'xaiPrivate',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': 'c28823c1-40fd-4eaf-9825-e28f11d1f8b2', 'publicName': 'llama-4-scout-17b-16e-instruct',
     'organization': 'meta', 'provider': 'fireworks',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '230a568c-9e39-4dbf-9bd8-a7e130111b4f', 'publicName': 'phantom-0822-1',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '49bd7403-c7fd-4d91-9829-90a91906ad6c', 'publicName': 'llama-4-maverick-03-26-experimental',
     'organization': 'meta', 'provider': 'meta',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '885976d3-d178-48f5-a3f4-6e13e0718872', 'publicName': 'qwq-32b', 'organization': 'alibaba',
     'provider': 'alibaba',
     'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}},
    {'id': '69f5d38a-45f5-4d3a-9320-b866a4035ed9', 'publicName': 'mistral-small-3.1-24b-instruct-2503',
     'organization': 'mistral', 'provider': 'mistral',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '789e245f-eafe-4c72-b563-d135e93988fc', 'publicName': 'gemma-3-27b-it', 'organization': 'google',
     'provider': 'google',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}},
    {'id': '14e9311c-94d2-40c2-8c54-273947e208b0', 'publicName': 'gpt-4.1-2025-04-14', 'organization': 'openai',
     'provider': 'openai',
     'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}}]

text_models = {model["publicName"]: model["id"] for model in models if "text" in model["capabilities"]["outputCapabilities"]}
image_models = {model["publicName"]: model["id"] for model in models if "image" in model["capabilities"]["outputCapabilities"]}
vision_models = [model["publicName"] for model in models if "image" in model["capabilities"]["inputCapabilities"]]

if has_nodriver:
    async def click_trunstile(page: nodriver.Tab, element = 'document.getElementById("cf-turnstile")'):
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
    url = "https://lmarena.ai"
    share_url = None
    api_endpoint = "https://lmarena.ai/nextjs-api/stream/create-evaluation"
    working = True
    active_by_default = True
    use_stream_timeout = False

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
                            cls.live += 1
                            break
                else:
                    cls.live -= 1
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
                    element = await page.select('[style="display: grid;"]')
                    if element:
                        await click_trunstile(page, 'document.querySelector(\'[style="display: grid;"]\')')
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
                        await click_trunstile(page)
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
                    if response.headers.get("Content-Type", "").startswith("image/"):
                        yield ImageResponse(str(response.url), prompt)
                    else:
                        text, *args = response.text.split("\n" * 10 + "<!--", 1)
                        if args:
                            debug.log("Save args to cache file:", str(cache_file))
                            with cache_file.open("w") as f:
                                f.write(args[0].strip())
                        yield text
                finally:
                    cls.looked = False
                return
            else:
                raise MissingRequirementsError("No auth file found and nodriver is not available.")

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
                            elif line.startswith("a3:"):
                                raise RuntimeError(f"LMArena: {json.loads(line[3:])}")
                            else:
                                debug.log(f"LMArena: Unknown line prefix: {line}")
                break
            except (CloudflareError, MissingAuthError):
                args = None
                debug.log(f"{cls.__name__}: Cloudflare error")
                continue
            except:
                raise
        if args and os.getenv("G4F_SHARE_AUTH") and not kwargs.get("action"):
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