from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import time
from typing import Dict
from urllib.parse import urlparse

import requests

from g4f.image import to_bytes, detect_file_type

try:
    import curl_cffi

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

try:
    import nodriver
    from nodriver import cdp
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import get_args_from_nodriver, raise_for_status, merge_cookies
from ...requests import StreamSession
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

models = [{'id': '019a98f7-afcd-779f-8dcb-856cc3b3f078', 'organization': 'google', 'provider': 'googleVertexGlobal', 'publicName': 'gemini-3-pro', 'name': 'gemini-3-pro', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 1, 'rankByModality': {'chat': 1, 'webdev': 4}}, {'id': '019a9389-a9d3-77a8-afbb-4fe4dd3d8630', 'organization': 'xai', 'provider': 'xaiResearch', 'publicName': 'grok-4.1-thinking', 'name': 'grok-4.1-thinking', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 2, 'rankByModality': {'chat': 2, 'webdev': 28}}, {'id': '019b0ad8-856e-74fc-871c-86ccbb2b1d35', 'organization': 'google', 'provider': 'googleWithThoughtSignatures', 'publicName': 'gemini-3-flash', 'name': 'fiercefalcon', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 3, 'rankByModality': {'chat': 3, 'webdev': 5}}, {'id': '019ab8b2-9bcf-79b5-9fb5-149a7c67b7c0', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-5-20251101-thinking-32k', 'name': 'claude-opus-4-5-20251101-thinking-32k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 4, 'rankByModality': {'chat': 4, 'webdev': 1}}, {'id': '019adbec-8396-71cc-87d5-b47f8431a6a6', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-5-20251101', 'name': 'claude-opus-4-5-20251101-vertex', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 5, 'rankByModality': {'chat': 5, 'webdev': 3}}, {'id': '019a9389-a4d8-748d-9939-b4640198302e', 'organization': 'xai', 'provider': 'xaiResearch', 'publicName': 'grok-4.1', 'name': 'grok-4.1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 6, 'rankByModality': {'chat': 6}}, {'id': '019b1424-f775-76d9-8c1b-dbb9c54ac4fb', 'organization': 'google', 'provider': 'googleWithThoughtSignatures', 'publicName': 'gemini-3-flash (thinking-minimal)', 'name': 'ghostfalcon-20251212', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 7, 'rankByModality': {'chat': 7, 'webdev': 12}}, {'id': '019a8548-a2b1-70ce-b1be-eba096d41f58', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.1-high', 'name': 'gpt-5.1-high', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 8, 'rankByModality': {'chat': 8, 'webdev': 9007199254740991}}, {'id': '51a47cc6-5ef9-4ac7-a59c-4009230d7564', 'publicName': 'gemini-2.5-pro-grounding-exp', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 9, 'rankByModality': {'chat': 9}}, {'id': '0199f060-b306-7e1f-aeae-0ebb4e3f1122', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'gemini-2.5-pro', 'name': 'gemini-2.5-pro', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 9, 'rankByModality': {'chat': 9, 'webdev': 27}}, {'id': 'b0ea1407-2f92-4515-b9cc-b22a6d6c14f2', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-sonnet-4-5-20250929-thinking-32k', 'name': 'claude-sonnet-4-5-20250929-thinking-32k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 10, 'rankByModality': {'chat': 10, 'webdev': 9}}, {'id': 'f1a2eb6f-fc30-4806-9e00-1efd0d73cbc4', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-1-20250805-thinking-16k', 'name': 'claude-opus-4-1-20250805-thinking-16k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 11, 'rankByModality': {'chat': 11}}, {'id': '019a2d13-28a5-7205-908c-0a58de904617', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-sonnet-4-5-20250929', 'name': 'claude-sonnet-4-5-20250929', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 12, 'rankByModality': {'chat': 12, 'webdev': 11}}, {'id': '96ae95fd-b70d-49c3-91cc-b58c7da1090b', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-1-20250805', 'name': 'claude-opus-4-1-20250805', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 14, 'rankByModality': {'chat': 14, 'webdev': 10}}, {'id': '019b1448-dafa-7f92-90c3-50e159c2263c', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.2-high', 'name': 'gpt-5.2-high', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 15, 'rankByModality': {'chat': 15}}, {'id': '0199c1e0-3720-742d-91c8-787788b0a19b', 'organization': 'openai', 'provider': 'openai', 'publicName': 'chatgpt-4o-latest-20250326', 'name': 'chatgpt-4o-latest-20250326', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 16, 'rankByModality': {'chat': 16}}, {'id': '019b1448-d548-78f4-8b98-788d72cbd057', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.2', 'name': 'gpt-5.2', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 17, 'rankByModality': {'chat': 17}}, {'id': '019a7ebf-0f3f-7518-8899-fca13e32d9dc', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.1', 'name': 'gpt-5.1', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 18, 'rankByModality': {'chat': 18, 'webdev': 15}}, {'id': '983bc566-b783-4d28-b24c-3c8b08eb1086', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-high', 'name': 'gpt-5-high', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 19, 'rankByModality': {'chat': 19}}, {'id': 'cb0f1e24-e8e9-4745-aabc-b926ffde7475', 'organization': 'openai', 'provider': 'openai', 'publicName': 'o3-2025-04-16', 'name': 'o3-2025-04-16', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 20, 'rankByModality': {'chat': 20}}, {'id': '812c93cc-5f88-4cff-b9ca-c11a26599b0e', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-max-preview', 'name': 'qwen3-max-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 21, 'rankByModality': {'chat': 21}}, {'id': '019aa41a-0a13-714a-beb1-be4a918a4b56', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-4-1-fast-reasoning', 'name': 'grok-4-1-fast-reasoning', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 22, 'rankByModality': {'chat': 22, 'webdev': 25}}, {'id': '019a4ca9-720d-75f5-9012-883ce8ff61df', 'organization': 'baidu', 'provider': 'baiduyiyan', 'publicName': 'ernie-5.0-preview-1103', 'name': 'bridge-mind', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 23, 'rankByModality': {'chat': 23}}, {'id': '019a59bc-8bb8-7933-92eb-fe143770c211', 'organization': 'moonshot', 'provider': 'moonshot', 'publicName': 'kimi-k2-thinking-turbo', 'name': 'kimi-k2-thinking-turbo', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 24, 'rankByModality': {'chat': 24, 'webdev': 16}}, {'id': 'f595e6f1-6175-4880-a9eb-377e390819e4', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.6', 'name': 'glm-4.6', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 25, 'rankByModality': {'chat': 25, 'webdev': 13}}, {'id': '4b11c78c-08c8-461c-938e-5fc97d56a40d', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-chat', 'name': 'gpt-5-chat', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 26, 'rankByModality': {'chat': 26}}, {'id': '98ad8b8b-12cd-46cd-98de-99edde7e03eb', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-max-2025-09-23', 'name': 'qwen3-max-2025-09-23', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 27, 'rankByModality': {'chat': 27}}, {'id': '3b5e9593-3dc0-4492-a3da-19784c4bde75', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-20250514-thinking-16k', 'name': 'claude-opus-4-20250514-thinking-16k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 29, 'rankByModality': {'chat': 29}}, {'id': 'ee7cb86e-8601-4585-b1d0-7c7380f8f6f4', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-235b-a22b-instruct-2507', 'name': 'qwen3-235b-a22b-instruct-2507', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 30, 'rankByModality': {'chat': 30}}, {'id': '019adb32-bb7a-77eb-882f-b8e3aaa2b2fd', 'organization': 'deepseek', 'provider': 'deepseekToolCalling', 'publicName': 'deepseek-v3.2-thinking', 'name': 'deepseek-v3.2-thinking', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 32, 'rankByModality': {'chat': 32, 'webdev': 14}}, {'id': '71023e9b-7361-498a-b6db-f2d2a83883fd', 'organization': 'xai', 'provider': 'xaiResearch', 'publicName': 'grok-4-fast-chat', 'name': 'grok-4-fast', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 33, 'rankByModality': {'chat': 33}}, {'id': '019acbac-df7c-73dc-9716-ebe040daaa4e', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'mistral-large-3', 'name': 'jaguar', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 34, 'rankByModality': {'chat': 34, 'webdev': 26}}, {'id': 'b88e983b-9459-473d-8bf1-753932f1679a', 'organization': 'moonshot', 'provider': 'moonshot', 'publicName': 'kimi-k2-0905-preview', 'name': 'kimi-k2-0905-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 36, 'rankByModality': {'chat': 36}}, {'id': '019adb32-b716-7591-9a2f-c6882973e340', 'organization': 'deepseek', 'provider': 'deepseek', 'publicName': 'deepseek-v3.2', 'name': 'deepseek-v3.2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 37, 'rankByModality': {'chat': 37, 'webdev': 22}}, {'id': '7a3626fc-4e64-4c9e-821f-b449a4b43b6a', 'organization': 'moonshot', 'provider': 'moonshot', 'publicName': 'kimi-k2-0711-preview', 'name': 'kimi-k2-0711-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 39, 'rankByModality': {'chat': 39}}, {'id': '716aa8ca-d729-427f-93ab-9579e4a13e98', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-vl-235b-a22b-instruct', 'name': 'qwen3-vl-235b-a22b-instruct', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 42, 'rankByModality': {'chat': 42}}, {'id': '14e9311c-94d2-40c2-8c54-273947e208b0', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-4.1-2025-04-14', 'name': 'gpt-4.1-2025-04-14', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 44, 'rankByModality': {'chat': 44}}, {'id': 'ee116d12-64d6-48a8-88e5-b2d06325cdd2', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-opus-4-20250514', 'name': 'claude-opus-4-20250514', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 45, 'rankByModality': {'chat': 45}}, {'id': '27035fb8-a25b-4ec9-8410-34be18328afd', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'mistral-medium-2508', 'name': 'mistral-medium-2508', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 46, 'rankByModality': {'chat': 46}}, {'id': 'b9edb8e9-4e98-49e7-8aaf-ae67e9797a11', 'organization': 'xai', 'provider': 'openrouter', 'publicName': 'grok-4-0709', 'name': 'grok-4-0709', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 48, 'rankByModality': {'chat': 48}}, {'id': 'd079ef40-3b20-4c58-ab5e-243738dbada5', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.5', 'name': 'glm-4.5', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 49, 'rankByModality': {'chat': 49}}, {'id': '0199f059-3877-7cfe-bc80-e01b1a4a83de', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'gemini-2.5-flash', 'name': 'gemini-2.5-flash', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 50, 'rankByModality': {'chat': 50}}, {'id': 'fc700d46-c4c1-4fec-88b5-f086876ae0bb', 'organization': 'google', 'provider': 'google', 'publicName': 'gemini-2.5-flash-preview-09-2025', 'name': 'gemini-2.5-flash-preview-09-2025', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 51, 'rankByModality': {'chat': 51}}, {'id': '0199e8e9-01ed-73e0-96ba-cf43b286bf10', 'organization': 'anthropic', 'provider': 'anthropic', 'publicName': 'claude-haiku-4-5-20251001', 'name': 'claude-haiku-4-5-20251001', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 52, 'rankByModality': {'chat': 52, 'webdev': 21}}, {'id': '19b3730a-0369-49ba-ad9c-09e7337937f0', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-4-fast-reasoning', 'name': 'grok-4-fast-reasoning', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 53, 'rankByModality': {'chat': 53, 'webdev': 29}}, {'id': '351fe482-eb6c-4536-857b-909e16c0bf52', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-next-80b-a3b-instruct', 'name': 'qwen3-next-80b-a3b-instruct', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 55, 'rankByModality': {'chat': 55}}, {'id': '6fcbe051-f521-4dc7-8986-c429eb6191bf', 'organization': 'meituan', 'provider': 'meituan', 'publicName': 'longcat-flash-chat', 'name': 'longcat-flash-chat', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 56, 'rankByModality': {'chat': 56}}, {'id': '1a400d9a-f61c-4bc2-89b4-a9b7e77dff12', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-235b-a22b-no-thinking', 'name': 'qwen3-235b-a22b-no-thinking', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 57, 'rankByModality': {'chat': 57}}, {'id': '4653dded-a46b-442a-a8fe-9bb9730e2453', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-sonnet-4-20250514-thinking-32k', 'name': 'claude-sonnet-4-20250514-thinking-32k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 58, 'rankByModality': {'chat': 58}}, {'id': '16b8e53a-cc7b-4608-a29a-20d4dac77cf2', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-235b-a22b-thinking-2507', 'name': 'qwen3-235b-a22b-thinking-2507', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 59, 'rankByModality': {'chat': 59}}, {'id': '03c511f5-0d35-4751-aae6-24f918b0d49e', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-vl-235b-a22b-thinking', 'name': 'qwen3-vl-235b-a22b-thinking', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 61, 'rankByModality': {'chat': 61}}, {'id': '5fd3caa8-fe4c-41a5-a22c-0025b58f4b42', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-mini-high', 'name': 'gpt-5-mini-high', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 62, 'rankByModality': {'chat': 62}}, {'id': '2f5253e4-75be-473c-bcfc-baeb3df0f8ad', 'organization': 'deepseek', 'provider': 'fireworks', 'publicName': 'deepseek-v3-0324', 'name': 'deepseek-v3-0324', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 63, 'rankByModality': {'chat': 63}}, {'id': 'f1102bbf-34ca-468f-a9fc-14bcf63f315b', 'organization': 'openai', 'provider': 'openai', 'publicName': 'o4-mini-2025-04-16', 'name': 'o4-mini-2025-04-16', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 64, 'rankByModality': {'chat': 64}}, {'id': '6a3a1e04-050e-4cb4-9052-b9ac4bec0c38', 'organization': 'tencent', 'provider': 'tencent', 'publicName': 'hunyuan-vision-1.5-thinking', 'name': 'hunyuan-vision-1.5-thinking', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 65, 'rankByModality': {'chat': 65}}, {'id': 'ac44dd10-0666-451c-b824-386ccfea7bcc', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-sonnet-4-20250514', 'name': 'claude-sonnet-4-20250514', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 67, 'rankByModality': {'chat': 67}}, {'id': 'be98fcfd-345c-4ae1-9a82-a19123ebf1d2', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-3-7-sonnet-20250219-thinking-32k', 'name': 'claude-3-7-sonnet-20250219-thinking-32k', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 69, 'rankByModality': {'chat': 69}}, {'id': 'af033cbd-ec6c-42cc-9afa-e227fc12efe8', 'organization': 'Alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-coder-480b-a35b-instruct', 'name': 'qwen3-coder-480b-a35b-instruct', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 70, 'rankByModality': {'chat': 70, 'webdev': 20}}, {'id': 'ba8c2392-4c47-42af-bfee-c6c057615a91', 'organization': 'tencent', 'provider': 'tencent', 'publicName': 'hunyuan-t1-20250711', 'name': 'hunyuan-t1-20250711', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 71, 'rankByModality': {'chat': 71}}, {'id': '27b9f8c6-3ee1-464a-9479-a8b3c2a48fd4', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'mistral-medium-2505', 'name': 'mistral-medium-2505', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 72, 'rankByModality': {'chat': 72}}, {'id': 'a8d1d310-e485-4c50-8f27-4bff18292a99', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-30b-a3b-instruct-2507', 'name': 'qwen3-30b-a3b-instruct-2507', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 73, 'rankByModality': {'chat': 73}}, {'id': '6a5437a7-c786-467b-b701-17b0bc8c8231', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-4.1-mini-2025-04-14', 'name': 'gpt-4.1-mini-2025-04-14', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 74, 'rankByModality': {'chat': 74}}, {'id': '75555628-8c14-402a-8d6e-43c19cb40116', 'organization': 'google', 'provider': 'google', 'publicName': 'gemini-2.5-flash-lite-preview-09-2025-no-thinking', 'name': 'gemini-2.5-flash-lite-preview-09-2025-no-thinking', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 76, 'rankByModality': {'chat': 76}}, {'id': '04ec9a17-c597-49df-acf0-963da275c246', 'organization': 'google', 'provider': 'google', 'publicName': 'gemini-2.5-flash-lite-preview-06-17-thinking', 'name': 'gemini-2.5-flash-lite-preview-06-17-thinking', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 77, 'rankByModality': {'chat': 77}}, {'id': '2595a594-fa54-4299-97cd-2d7380d21c80', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-235b-a22b', 'name': 'qwen3-235b-a22b', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 78, 'rankByModality': {'chat': 78}}, {'id': 'f44e280a-7914-43ca-a25d-ecfcc5d48d09', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-3-5-sonnet-20241022', 'name': 'claude-3-5-sonnet-20241022', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 80, 'rankByModality': {'chat': 80}}, {'id': 'c5a11495-081a-4dc6-8d9a-64a4fd6f7bbc', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-3-7-sonnet-20250219', 'name': 'claude-3-7-sonnet-20250219', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 81, 'rankByModality': {'chat': 81}}, {'id': '7bfb254a-5d32-4ce2-b6dc-2c7faf1d5fe8', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.5-air', 'name': 'glm-4.5-air', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 82, 'rankByModality': {'chat': 82}}, {'id': '73cf8705-98c8-4b75-8d04-e3746e1c1565', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-next-80b-a3b-thinking', 'name': 'qwen3-next-80b-a3b-thinking', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 83, 'rankByModality': {'chat': 83}}, {'id': '87e8d160-049e-4b4e-adc4-7f2511348539', 'organization': 'minimax', 'provider': 'minimax', 'publicName': 'minimax-m1', 'name': 'minimax-m1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 84, 'rankByModality': {'chat': 84}}, {'id': '789e245f-eafe-4c72-b563-d135e93988fc', 'organization': 'google', 'provider': 'google', 'publicName': 'gemma-3-27b-it', 'name': 'gemma-3-27b-it', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 85, 'rankByModality': {'chat': 85}}, {'id': '149619f1-f1d5-45fd-a53e-7d790f156f20', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-3-mini-high', 'name': 'grok-3-mini-high', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 87, 'rankByModality': {'chat': 87}}, {'id': '7a55108b-b997-4cff-a72f-5aa83beee918', 'organization': 'google', 'provider': 'google', 'publicName': 'gemini-2.0-flash-001', 'name': 'gemini-2.0-flash-001', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 88, 'rankByModality': {'chat': 88}}, {'id': '7699c8d4-0742-42f9-a117-d10e84688dab', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-3-mini-beta', 'name': 'grok-3-mini-beta', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 90, 'rankByModality': {'chat': 90}}, {'id': 'bbad1d17-6aa5-4321-949c-d11fb6289241', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'mistral-small-2506', 'name': 'mistral-small-2506', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 91, 'rankByModality': {'chat': 91}}, {'id': '019aebfd-af0e-7f0c-8f0d-96c588e4cd3b', 'organization': 'prime-intellect', 'provider': 'openrouter', 'publicName': 'intellect-3', 'name': 'intellect-3', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 92, 'rankByModality': {'chat': 92}}, {'id': '6ee9f901-17b5-4fbe-9cc2-13c16497c23b', 'organization': 'openai', 'provider': 'fireworks', 'publicName': 'gpt-oss-120b', 'name': 'gpt-oss-120b', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 94, 'rankByModality': {'chat': 94}}, {'id': '9dab0475-a0cc-4524-84a2-3fd25aa8c768', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.5v', 'name': 'glm-4.5v', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 95, 'rankByModality': {'chat': 95}}, {'id': '0f785ba1-efcb-472d-961e-69f7b251c7e3', 'organization': 'cohere', 'provider': 'cohere', 'publicName': 'command-a-03-2025', 'name': 'command-a-03-2025', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 96, 'rankByModality': {'chat': 96}}, {'id': '019a4c75-256c-790b-9088-4694cc63c507', 'organization': 'amazon', 'provider': 'amazon', 'publicName': 'amazon-nova-experimental-chat-10-20', 'name': 'amazon-nova-experimental-chat-10-20', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 98, 'rankByModality': {'chat': 98}}, {'id': 'c680645e-efac-4a81-b0af-da16902b2541', 'organization': 'openai', 'provider': 'openai', 'publicName': 'o3-mini', 'name': 'o3-mini', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 99, 'rankByModality': {'chat': 99}}, {'id': '71f96ca9-4cf8-4be7-bac2-2231613930a6', 'organization': 'ant-group', 'provider': 'antgroup', 'publicName': 'ling-flash-2.0', 'name': 'ling-flash-2.0', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 101, 'rankByModality': {'chat': 101}}, {'id': '019a27e0-e7d8-7b0b-877c-a2106c6eb87d', 'organization': 'minimax', 'provider': 'minimax', 'publicName': 'minimax-m2', 'name': 'minimax-m2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rank': 102, 'rankByModality': {'chat': 102, 'webdev': 18}}, {'id': '1ea13a81-93a7-4804-bcdd-693cd72e302d', 'organization': 'stepfun', 'provider': 'stepfun', 'publicName': 'step-3', 'name': 'step-3', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 104, 'rankByModality': {'chat': 104}}, {'id': '2dc249b3-98da-44b4-8d1e-6666346a8012', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-nano-high', 'name': 'gpt-5-nano-high', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 114, 'rankByModality': {'chat': 114}}, {'id': '019ae300-83b7-7717-a1e0-31accd1ff6fa', 'organization': 'amazon', 'provider': 'amazonBedrock', 'publicName': 'nova-2-lite', 'name': 'global.amazon.nova-2-lite-v1:0', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 115, 'rankByModality': {'chat': 115}}, {'id': '885976d3-d178-48f5-a3f4-6e13e0718872', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwq-32b', 'name': 'qwq-32b', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 121, 'rankByModality': {'chat': 121}}, {'id': 'b5ad3ab7-fc56-4ecd-8921-bd56b55c1159', 'organization': 'meta', 'provider': 'fireworks', 'publicName': 'llama-4-maverick-17b-128e-instruct', 'name': 'llama-4-maverick-17b-128e-instruct', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 125, 'rankByModality': {'chat': 125}}, {'id': '019b0aa7-334a-78e8-b2a8-885f31f4fc0c', 'organization': 'nvidia', 'provider': 'nvidia', 'publicName': 'nvidia-nemotron-3-nano-30b-a3b-bf16', 'name': 'december-chatbot', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 126, 'rankByModality': {'chat': 126}}, {'id': '9a066f6a-7205-4325-8d0b-d81cc4b049c0', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-30b-a3b', 'name': 'qwen3-30b-a3b', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 128, 'rankByModality': {'chat': 128}}, {'id': 'f6fbf06c-532c-4c8a-89c7-f3ddcfb34bd1', 'organization': 'anthropic', 'provider': 'googleVertexAnthropic', 'publicName': 'claude-3-5-haiku-20241022', 'name': 'claude-3-5-haiku-20241022', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 131, 'rankByModality': {'chat': 131}}, {'id': '11ad4114-c868-4fed-b6e7-d535dc9c62f8', 'organization': 'ant-group', 'provider': 'antgroup', 'publicName': 'ring-flash-2.0', 'name': 'ring-flash-2.0', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 137, 'rankByModality': {'chat': 137}}, {'id': 'dcbd7897-5a37-4a34-93f1-76a24c7bb028', 'organization': 'meta', 'provider': 'fireworks', 'publicName': 'llama-3.3-70b-instruct', 'name': 'llama-3.3-70b-instruct', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 139, 'rankByModality': {'chat': 139}}, {'id': '896a3848-ae03-4651-963b-7d8f54b61ae8', 'organization': 'google', 'provider': 'google', 'publicName': 'gemma-3n-e4b-it', 'name': 'gemma-3n-e4b-it', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 140, 'rankByModality': {'chat': 140}}, {'id': 'ec3beb4b-7229-4232-bab9-670ee52dd711', 'organization': 'openai', 'provider': 'fireworks', 'publicName': 'gpt-oss-20b', 'name': 'gpt-oss-20b', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 142, 'rankByModality': {'chat': 142}}, {'id': '019a6f77-e20d-7c1d-a7cd-8bd926e7395d', 'organization': 'inception-ai', 'provider': 'openrouter', 'publicName': 'mercury', 'name': 'mercury', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 150, 'rankByModality': {'chat': 150}}, {'id': '019ac2ef-27e1-769f-8258-d131f79e28ef', 'organization': 'allenai', 'provider': 'openrouter', 'publicName': 'olmo-3-32b-think', 'name': 'olmo-3-32b-think', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 153, 'rankByModality': {'chat': 153}}, {'id': '6337f479-2fc8-4311-a76b-8c957765cd68', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'magistral-medium-2506', 'name': 'magistral-medium-2506', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 158, 'rankByModality': {'chat': 158}}, {'id': '69f5d38a-45f5-4d3a-9320-b866a4035ed9', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'mistral-small-3.1-24b-instruct-2503', 'name': 'mistral-small-3.1-24b-instruct-2503', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rank': 159, 'rankByModality': {'chat': 159}}, {'id': '4ddb69f5-391a-4f78-af92-7d7328c18ab1', 'organization': 'ibm', 'provider': 'ibm', 'publicName': 'ibm-granite-h-small', 'name': 'ibm-granite-h-small', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rank': 167, 'rankByModality': {'chat': 167}}, {'id': '019aa97e-7b68-7c21-add9-b17edc20ff02', 'publicName': 'anonymous-1111', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b2357-c7c5-74f0-96d3-bd98d6d33fa0', 'publicName': 'lucky-lark', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b2498-86aa-7221-9963-a62a24a4703e', 'publicName': 'ghostfalcon-20251215', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rankByModality': {'chat': 9007199254740991, 'webdev': 9007199254740991}}, {'id': '019b2498-89df-743c-8136-b6846f5d6df6', 'publicName': 'fiercefalcon-20251215', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rankByModality': {'chat': 9007199254740991, 'webdev': 9007199254740991}}, {'id': '019adcc2-c05e-7b1a-a950-92b1ca78087b', 'publicName': 'evo-logic', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019ab9ad-caac-78f4-8007-bf0e082771d0', 'publicName': 'raptor-1123', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019ab9ad-d128-7858-8745-6da5936ccb5e', 'publicName': 'raptor-1124', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019ade1a-a918-79dd-9ef9-ba22f97c977d', 'publicName': 'micro-mango', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '42015285-534d-4e6b-9a9a-9061c2f73e1c', 'publicName': 'x1-1-preview-0915', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199e3d1-a308-77b9-a650-41453e8ef2fb', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-vl-8b-thinking', 'name': 'qwen3-vl-8b-thinking', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199e3d1-a713-7de2-a5dd-a1583cad9532', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-vl-8b-instruct', 'name': 'qwen3-vl-8b-instruct', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a184d-efba-7743-979a-324e949aa1c4', 'publicName': 'ernie-exp-251024', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b287d-9ffc-7ce4-95be-d93929cdd94f', 'publicName': 'step-3-mini-2511', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'b4a681ed-df4e-476f-89c6-a992a5783e60', 'publicName': 'EB45-turbo-vl-0906', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '39b185cb-aba9-4232-99ea-074883a5ccd4', 'publicName': 'stephen-v2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019acb94-2eca-79a2-bc6c-585bf303df54', 'publicName': 'dark-dragon', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '8d1f38a1-51a6-4030-ae4b-e19fb503e4fa', 'publicName': 'x1-turbo-0906', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019ae523-87e2-702f-8416-9266b20630d6', 'publicName': 'raptor-1202', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019ae5e0-1ba6-701e-a482-a4b8bb23fe36', 'publicName': 'frame-flow', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019aa41a-01bf-726c-913c-8a65b8f4c879', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-4-1-fast-non-reasoning', 'name': 'grok-4-1-fast-non-reasoning', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019afca4-2085-78ee-847a-e221cdb50501', 'publicName': 'raptor-llm-1205', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b2d23-7f4f-7544-891e-bb9c434f485c', 'publicName': 'anonymous-1215', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019aec73-6e86-784c-885a-7fe129025f95', 'publicName': 'beluga-1128-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019af0b1-5c1d-7c24-8260-621ed4b32b96', 'publicName': 'phantom-1203-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0923-102a-7b40-b9ba-95110a235709', 'publicName': 'beluga-1128-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019af1f8-aa23-7188-8a73-e717ae24a7d8', 'publicName': 'phantom-mm-1125-1', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b00de-7b51-7d3e-8d27-01b44b156c5d', 'publicName': 'holo-scope', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0442-ff8e-72ac-99c0-9fc71eaf6db2', 'publicName': 'integrated-info', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0582-32cb-7f4b-9402-1f833c3b2a9e', 'publicName': 'beluga-1203-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b2a43-6750-7119-808e-09c32d66af6a', 'publicName': 'master-node', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'a14546b5-d78d-4cf6-bb61-ab5b8510a9d6', 'organization': 'amazon', 'provider': 'amazonBedrock', 'publicName': 'amazon.nova-pro-v1:0', 'name': 'amazon.nova-pro-v1:0', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b092f-6ecf-7945-9e1a-43dbe448fc1b', 'publicName': 'beluga-1202-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b2eea-8cab-7525-9e70-2be764ff8ef6', 'publicName': 'cogilux', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199c1d5-51b8-7ead-a0a8-3f59234682fa', 'publicName': 'gpt-5-high-no-system-prompt', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '638fb8b8-1037-4ee5-bfba-333392575a5d', 'publicName': 'EB45-vision', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b33ae-2ee4-7de8-89f3-d0fef62fe698', 'organization': 'allenai', 'provider': 'openrouter', 'publicName': 'olmo-3.1-32b-think', 'name': 'olmo-3.1-32b-think', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0e9a-c833-7e18-8a31-31c5a2415284', 'publicName': 'fire-bird', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b33cd-0c2b-7b75-8238-0396e5928c38', 'publicName': 'beluga-1211-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0fe9-a2a2-7a2b-ad88-abfc70c4eaa3', 'publicName': 'anonymous-1212', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'c15b93ed-e87b-467f-8f9f-d830fd7aa54d', 'publicName': 'lmarena-internal-test-only', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'f1a5a6ab-e1b1-4247-88ac-49395291c1e3', 'publicName': 'not-a-new-model', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b0bd7-efa8-7024-a3be-a2ee84cf4b66', 'publicName': 'raptor-1.8-1208', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b33ce-c65a-7d39-bfe7-a358dcd07a39', 'publicName': 'beluga-1214-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b352d-36a8-7dd3-bbec-2bc78852d6c0', 'publicName': 'mimo-v2-flash', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rankByModality': {'chat': 9007199254740991, 'webdev': 9007199254740991}}, {'id': 'e9dd5a96-c066-48b0-869f-eb762030b5ed', 'publicName': 'EB45-turbo', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199f437-08bd-7907-81e7-098e2eab750a', 'publicName': 'raptor-vision-1015', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b00f1-3d28-7616-b27f-f4ebc9c25780', 'publicName': 'silentnova', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b151a-7c3b-72a2-8811-0bf9317c2ef5', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.6v', 'name': 'glm-4.6v', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1448-ff14-7c98-a1ac-726fece799ec', 'publicName': 'gpt-5.2-no-system-prompt', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1449-0313-7911-b836-419e2ed79b2e', 'publicName': 'gpt-5.2-high-no-system-prompt', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1536-49c0-73b2-8d45-403b8571568d', 'organization': 'zai', 'provider': 'zai', 'publicName': 'glm-4.6v-flash', 'name': 'glm-4.6v-flash', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '9af435c8-1f53-4b78-a400-c1f5e9fe09b0', 'publicName': 'leepwal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a33c6-e248-7091-94c0-af68f29390b8', 'publicName': 'blackhawk', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1644-8045-7af4-a02a-1bb21cd26344', 'publicName': 'jet-force', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1699-ff5f-7f8d-a158-9b04cc99737e', 'publicName': 'kernel-sense', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b1994-ac41-7f07-8f05-bbaeaf9aaea8', 'publicName': 'december-chatbot2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'ac31e980-8bf1-4637-adba-cf9ffa8b6343', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-max-2025-09-26', 'name': 'qwen3-max-2025-09-26', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'f23d6df4-4395-4404-897f-bdedc909e783', 'publicName': 'raptor', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a6617-4f6a-77ab-af79-097f7aa7c86d', 'publicName': 'winter-wind', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'ee3588cd-1fe1-484a-bcc9-f92065b8380c', 'organization': 'xiaomi', 'provider': 'xiaomi', 'publicName': 'MiMo-7B', 'name': 'MiMo-7B', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199c9dc-e157-7458-bd49-5942363be215', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen3-omni-flash', 'name': 'qwen3-omni-flash', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a3aeb-371d-7474-a1b7-b8f4207a9a2c', 'publicName': 'qwen3-max-thinking', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'e3c9ea42-5f42-496b-bc80-c7e8ee5653cc', 'publicName': 'stephen-vision-csfix', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0304b4de-e544-48d4-8490-ad9123bc26e3', 'publicName': 'monster', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199eb70-db25-73ea-b944-f18fe0c3c0cd', 'publicName': 'ernie-exp-vl-251016', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '6fe1ec40-3219-4c33-b3e7-0e65658b4194', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen-vl-max-2025-08-13', 'name': 'qwen-vl-max-2025-08-13', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199edd2-88be-76b8-9aaa-5fc6b9c53503', 'publicName': 'ling-1t', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'd8444b25-e302-4d3b-8561-7ded087ee03c', 'publicName': 'monterey', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a3e64-eae6-7178-a9ae-2894c25ada46', 'publicName': 'sunshine-ai', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a5658-ba6f-7185-b79d-51516ef94314', 'publicName': 'gauss', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a026f-30b2-7ffa-9714-7180577666e2', 'publicName': 'qwen3-max-2025-10-20', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a17b5-5e1e-7df6-9e1a-6c4338f8b6ff', 'organization': 'minimax', 'provider': 'minimax', 'publicName': 'minimax-m2-preview', 'name': 'minimax-m2-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True, 'web': True}}, 'rankByModality': {'chat': 9007199254740991, 'webdev': 9007199254740991}}, {'id': '019a7b93-af36-7f06-bb46-2b6b0548fc93', 'publicName': 'whisperfall', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a842c-7e62-725a-8a2a-f76f5ec786dd', 'publicName': 'neon', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a6c82-2ead-7475-8612-55dabba11c31', 'publicName': 'viper', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a19ad-97fe-72fb-89ad-d02355152e55', 'organization': 'baidu', 'provider': 'baiduyiyan', 'publicName': 'ernie-5.0-preview-1120', 'name': 'ernie-5.0-preview-1120', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199de70-e6ad-7276-9020-a7502bed99ad', 'publicName': 'flying-octopus', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a19ad-8f50-7282-8cb9-5e49b9052aff', 'publicName': 'ernie-exp-251023', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a26b4-2a20-7864-b1f7-d471ec1f4e3f', 'publicName': 'ernie-exp-251027', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '48fe3167-5680-4903-9ab5-2f0b9dc05815', 'publicName': 'nightride-on', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a19ad-a47e-7828-aa1f-5c7f2c4f65fe', 'publicName': 'ring-1t', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a1dce-b0d7-75cc-8742-997a0639f061', 'publicName': 'ernie-exp-251025', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a26b4-2540-7a7d-96bf-b6b377e8d1ec', 'publicName': 'ernie-exp-251026', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': 'c822ec98-38e9-4e43-a434-982eb534824f', 'publicName': 'nightride-on-v2', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '0199fdfb-8d7f-7435-a362-e68678c308c2', 'publicName': 'raptor-llm-1017', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a26b4-2f39-7a0e-818d-254d0f37a85a', 'publicName': 'raptor-llm-1024', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a5658-b301-7c8e-acb2-707cad8d6177', 'publicName': 'newton', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a4ca3-54ee-7c78-9cbf-19584ddfb7be', 'publicName': 'aegis-core', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a7011-49dd-7e6e-94e1-ced43eac205c', 'publicName': 'raptor-vision-1107', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a7904-198f-71dc-b671-b9e39b950a37', 'publicName': 'raptor-1110', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a6617-53aa-7d9f-940d-94afc98aab1f', 'publicName': 'rain-drop', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '6c98ce8c-41d1-42cd-b2e3-292c6519add5', 'publicName': 'redwood', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '19ad5f04-38c6-48ae-b826-f7d5bbfd79f7', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-high-new-system-prompt', 'name': 'gpt-5-high-new-system-prompt', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '1c0259b5-dff7-48ce-bca1-b6957675463b', 'organization': 'xiaomi', 'provider': 'xiaomiVision', 'publicName': 'MiMo-VL-7B-RL-2508', 'name': 'MiMo-VL-7B-RL-2508', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019a9e80-e469-722e-8f9a-585860a5859d', 'publicName': 'raptor-1119', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019aa3fb-ec59-7b49-af33-57d0981509e0', 'publicName': 'ling-1t-1031', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'text': True}}, 'rankByModality': {'chat': 9007199254740991}}, {'id': '019b144b-678a-7d13-a17e-973d4953ac35', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.2-high', 'name': 'gpt-5.2-high-code', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 2, 'rankByModality': {'webdev': 2}}, {'id': '019a0ec1-e54d-7354-be40-62fb6f0e5d43', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5-medium', 'name': 'gpt-5-medium', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 6, 'rankByModality': {'webdev': 6}}, {'id': '019b144b-5ffb-7192-be63-79533ef5d854', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.2', 'name': 'gpt-5.2-code', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 7, 'rankByModality': {'webdev': 7}}, {'id': '019a95f0-02c7-7f62-a820-c809f767a222', 'organization': 'openai', 'provider': 'openai', 'publicName': 'gpt-5.1-medium', 'name': 'gpt-5.1-medium', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 8, 'rankByModality': {'webdev': 8}}, {'id': '019a84e2-e1b7-718b-8b8d-589079121b9b', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.1-codex', 'name': 'gpt-5.1-codex', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 17, 'rankByModality': {'webdev': 17}}, {'id': '019a9b1b-4c52-7245-83de-4cdaaf9e0f2b', 'organization': 'kwai', 'provider': 'kwai', 'publicName': 'KAT-Coder-Pro-V1', 'name': 'KAT-Coder-Pro-V1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 23, 'rankByModality': {'webdev': 23}}, {'id': '019a84e2-e6bd-7123-a5f6-626d1766d4f2', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.1-codex-mini', 'name': 'gpt-5.1-codex-mini', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 24, 'rankByModality': {'webdev': 24}}, {'id': '019a5d6b-1dd1-77ff-81df-108d460d6481', 'organization': 'xai', 'provider': 'xaiPublic', 'publicName': 'grok-code-fast-1', 'name': 'grok-code-fast-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 30, 'rankByModality': {'webdev': 30}}, {'id': '019a6a30-cd7d-7431-8c4a-7be88deebf43', 'organization': 'mistral', 'provider': 'mistral', 'publicName': 'devstral-medium-2507', 'name': 'devstral-medium-2507', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rank': 31, 'rankByModality': {'webdev': 31}}, {'id': '019abd9a-f6ff-7081-8bdb-ff0a352dc601', 'publicName': 'robin', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019ac30c-70ce-72af-b6e9-cfbc52d46c19', 'publicName': 'robin-high', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019b303a-ad72-75fc-87d8-9728e0015acd', 'publicName': 'zebra', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019aeb38-cc3b-7421-a472-0bfaaeace035', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.1-codex-max', 'name': 'gpt-5.1-codex-max', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019b0fe1-5dfd-7c0e-88a7-489650364da5', 'publicName': 'lambda-1201-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019b0ff9-6d2c-703f-bef7-4da3c417b261', 'publicName': 'micro-mango', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019a9afe-d7c8-7c5d-ad57-d34170bcebae', 'publicName': 'f1031_wda', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'web': True}}, 'rankByModality': {'webdev': 9007199254740991}}, {'id': '019b2893-2508-7686-a63d-4861997565d2', 'organization': 'openai', 'provider': 'customOpenai', 'publicName': 'gpt-image-1.5', 'name': 'gpt-image-1.5', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 1, 'rankByModality': {'image': 1}}, {'id': '019abc10-e78d-7932-b725-7f1563ed8a12', 'organization': 'google', 'provider': 'google-genai', 'publicName': 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'name': 'gemini-3-pro-image-preview-2k', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 2, 'rankByModality': {'image': 2}}, {'id': '019aa208-5c19-7162-ae3b-0a9ddbb1e16a', 'organization': 'google', 'provider': 'google-genai', 'publicName': 'gemini-3-pro-image-preview (nano-banana-pro)', 'name': 'gemini-3-pro-image-preview', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 3, 'rankByModality': {'image': 3}}, {'id': '019b2a10-15dd-78c6-b8ae-bafea83efd6e', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-2-max', 'name': 'flux-2-max', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 4, 'rankByModality': {'image': 4}}, {'id': '019abed6-d96e-7a2b-bf69-198c28bef281', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-2-flex', 'name': 'flux-2-flex', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 5, 'rankByModality': {'image': 5}}, {'id': '0199ef2a-583f-7088-b704-b75fd169401d', 'organization': 'google', 'provider': 'google-genai', 'publicName': 'gemini-2.5-flash-image-preview (nano-banana)', 'name': 'gemini-2.5-flash-image-preview (nano-banana)', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 6, 'rankByModality': {'image': 6}}, {'id': '019abcf4-5600-7a8b-864d-9b8ab7ab7328', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-2-pro', 'name': 'flux-2-pro', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 7, 'rankByModality': {'image': 7}}, {'id': '7766a45c-1b6b-4fb8-9823-2557291e1ddd', 'organization': 'tencent', 'provider': 'tencent', 'publicName': 'hunyuan-image-3.0', 'name': 'hunyuan-image-3.0', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 8, 'rankByModality': {'image': 8}}, {'id': '019ae6a0-4773-77d5-8ffb-cc35813e063c', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-2-dev', 'name': 'flux-2-dev', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 9, 'rankByModality': {'image': 9}}, {'id': '019abd43-b052-7eec-aa57-e895e45c9723', 'organization': 'bytedance', 'provider': 'bytedance', 'publicName': 'seedream-4.5', 'name': 'autumn', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 10, 'rankByModality': {'image': 10}}, {'id': '019a5050-2875-78ed-ae3a-d9a51a438685', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'wan2.5-t2i-preview', 'name': 'wan2.5-t2i-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 14, 'rankByModality': {'image': 14}}, {'id': '019a5050-1695-7b5d-b045-ba01ff08ec28', 'publicName': 'wan2.5-preview', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 14, 'rankByModality': {'image': 14}}, {'id': '32974d8d-333c-4d2e-abf3-f258c0ac1310', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedream-4-high-res-fal', 'name': 'seedream-4-high-res-fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 15, 'rankByModality': {'image': 15}}, {'id': '69f90b32-01dc-43e1-8c48-bf494f8f4f38', 'publicName': 'gpt-image-1-high-fidelity', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 17, 'rankByModality': {'image': 17}}, {'id': '6e855f13-55d7-4127-8656-9168a9f4dcc0', 'organization': 'openai', 'provider': 'customOpenai', 'publicName': 'gpt-image-1', 'name': 'gpt-image-1', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 17, 'rankByModality': {'image': 17}}, {'id': '0199c238-f8ee-7f7d-afc1-7e28fcfd21cf', 'organization': 'openai', 'provider': 'customOpenai', 'publicName': 'gpt-image-1-mini', 'name': 'gpt-image-1-mini', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 18, 'rankByModality': {'image': 18}}, {'id': '1b407d5c-1806-477c-90a5-e5c5a114f3bc', 'organization': 'microsoft-ai', 'provider': 'maiImage', 'publicName': 'mai-image-1', 'name': 'mai-image-1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 19, 'rankByModality': {'image': 19}}, {'id': 'd8771262-8248-4372-90d5-eb41910db034', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedream-3', 'name': 'seedream-3', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 20, 'rankByModality': {'image': 20}}, {'id': '0633b1ef-289f-49d4-a834-3d475a25e46b', 'publicName': 'flux-1-kontext-max', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 21, 'rankByModality': {'image': 21}}, {'id': '9fe82ee1-c84f-417f-b0e7-cab4ae4cf3f3', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen-image-prompt-extend', 'name': 'qwen-image-prompt-extend', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 22, 'rankByModality': {'image': 22}}, {'id': '28a8f330-3554-448c-9f32-2c0a08ec6477', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-1-kontext-pro', 'name': 'flux-1-kontext-pro', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 23, 'rankByModality': {'image': 23}}, {'id': '51ad1d79-61e2-414c-99e3-faeb64bb6b1b', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'imagen-3.0-generate-002', 'name': 'imagen-3.0-generate-002', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 24, 'rankByModality': {'image': 24}}, {'id': '73378be5-cdba-49e7-b3d0-027949871aa6', 'organization': 'Ideogram', 'provider': 'fal', 'publicName': 'ideogram-v3-quality', 'name': 'ideogram-v3-quality', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 26, 'rankByModality': {'image': 26}}, {'id': 'e7c9fa2d-6f5d-40eb-8305-0980b11c7cab', 'organization': 'luma-ai', 'provider': 'fal', 'publicName': 'photon', 'name': 'photon', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 27, 'rankByModality': {'image': 27}}, {'id': 'b88d5814-1d20-49cc-9eb6-e362f5851661', 'organization': 'Recraft', 'provider': 'fal', 'publicName': 'recraft-v3', 'name': 'recraft-v3', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 28, 'rankByModality': {'image': 28}}, {'id': '5a3b3520-c87d-481f-953c-1364687b6e8f', 'organization': 'leonardo-ai', 'provider': 'leonardo-ai', 'publicName': 'lucid-origin', 'name': 'lucid-origin', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 29, 'rankByModality': {'image': 29}}, {'id': '69bbf7d4-9f44-447e-a868-abc4f7a31810', 'organization': 'google', 'provider': 'google', 'publicName': 'gemini-2.0-flash-preview-image-generation', 'name': 'gemini-2.0-flash-preview-image-generation', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 32, 'rankByModality': {'image': 32}}, {'id': 'bb97bc68-131c-4ea4-a59e-03a6252de0d2', 'organization': 'openai', 'provider': 'openai', 'publicName': 'dall-e-3', 'name': 'dall-e-3', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 33, 'rankByModality': {'image': 33}}, {'id': 'eb90ae46-a73a-4f27-be8b-40f090592c9a', 'organization': 'bfl', 'provider': 'bfl', 'publicName': 'flux-1-kontext-dev', 'name': 'flux-1-kontext-dev', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rank': 35, 'rankByModality': {'image': 35}}, {'id': '019aafce-f609-727d-a998-0fa521fcb1d5', 'publicName': 'tangerine', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019adb32-afa4-749e-9992-39653b52fe13', 'organization': 'shengshu', 'provider': 'shengshu', 'publicName': 'vidu-q2-image', 'name': 'viduq2-image', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b207a-d836-7742-a312-b2b247b2366e', 'organization': 'reve', 'provider': 'fal', 'publicName': 'reve-v1.1-fast', 'name': 'epsilon-fast', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': 'f44fd4f8-af30-480f-8ce2-80b2bdfea55e', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'imagen-4.0-fast-generate-001', 'name': 'imagen-4.0-fast-generate-001', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019ae2a3-304d-773b-b333-daa050124a92', 'publicName': 'gemini-3-pro-image-preview-4k (nano-banana-pro)', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019ae6da-6438-7077-9d2d-b311a35645f8', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'imagen-4.0-ultra-generate-001', 'name': 'imagen-4.0-ultra-generate-001', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019ae6da-6788-761a-8253-e0bb2bf2e3a9', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'imagen-4.0-generate-001', 'name': 'imagen-4.0-generate-001', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019aeb62-c6ea-788e-88f9-19b1b48325b5', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'wan2.5-i2i-preview', 'name': 'wan2.5-i2i-preview', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b2eac-06ad-7e71-a9ac-67a6765f2619', 'organization': 'openai', 'provider': 'customOpenai', 'publicName': 'chatgpt-image-latest (20251216)', 'name': 'chatgpt-image-latest-20251216', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b1093-8ff2-7c6f-b51c-8774830e52e5', 'publicName': 'hazel-small-2', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b1093-93d0-7655-a29b-c00b1a4104d4', 'publicName': 'hazel-small-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b00fe-89e8-7fda-9f6c-36d16813fb48', 'publicName': 'hazel-gen-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': 'e2969ebb-6450-4bc4-87c9-bbdcf95840da', 'publicName': 'seededit-3.0', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': 'a9a26426-5377-4efa-bef9-de71e29ad943', 'organization': 'tencent', 'provider': 'fal', 'publicName': 'hunyuan-image-2.1', 'name': 'hunyuan-image-2.1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b00ff-a0f5-71e9-b43c-d02661323dfc', 'publicName': 'hazel-gen-4', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '32bff2df-00e6-409b-ad3f-bfbad87cc49f', 'publicName': 'hidream-e1.1', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '019b169a-09c6-79ea-9400-1566d793ade1', 'organization': 'reve', 'provider': 'reve', 'publicName': 'reve-v1.1', 'name': 'epsilon', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': True}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '995cf221-af30-466d-a809-8e0985f83649', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'qwen-image-edit', 'name': 'qwen-image-edit', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True, 'multipleImages': False}}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '0199e980-d247-7dd3-9ca1-77092f126f05', 'publicName': 'hunyuan-image-3.0-fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'image': {'aspectRatios': ['1:1']}}}, 'rankByModality': {'image': 9007199254740991}}, {'id': '0199eacf-1119-7e20-a53e-86b740c32f03', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3.1-fast-audio', 'name': 'veo-3.1-fast-audio', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 1, 'rankByModality': {'video': 1}}, {'id': '0199eacf-0d64-736a-8309-07cc254b849d', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3.1-audio', 'name': 'veo-3.1-audio', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 2, 'rankByModality': {'video': 2}}, {'id': '9bbbca46-b6c2-4919-83a8-87ef1c559c4e', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3-fast-audio', 'name': 'veo-3-fast-audio', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 3, 'rankByModality': {'video': 3}}, {'id': '01f39601-03ed-4d66-8afb-5039c90b27ea', 'organization': 'openai', 'provider': 'openai', 'publicName': 'sora-2-pro', 'name': 'sora-2-pro', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 4, 'rankByModality': {'video': 4}}, {'id': 'a071b843-0fc2-4fcf-b644-023509635452', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3-audio', 'name': 'veo-3-audio', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 5, 'rankByModality': {'video': 5}}, {'id': '043a03a9-a792-4045-9f6d-4bcd747dac43', 'organization': 'openai', 'provider': 'openai', 'publicName': 'sora-2', 'name': 'sora-2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 6, 'rankByModality': {'video': 6}}, {'id': '019a74c8-5bf5-7f37-b0c4-4d597ea6f831', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'wan2.5-t2v-preview', 'name': 'wan2.5-t2v-preview', 'capabilities': {'inputCapabilities': {'text': True, 'image': False}, 'outputCapabilities': {'video': True}}, 'rank': 7, 'rankByModality': {'video': 7}}, {'id': '1b677c7e-49dd-4045-9ce0-d1aedcb9bbbc', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3-fast', 'name': 'veo-3-fast', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 8, 'rankByModality': {'video': 8}}, {'id': '80caa6ac-05cd-4403-88e1-ef0164c8b1a8', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-3', 'name': 'veo-3', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 9, 'rankByModality': {'video': 9}}, {'id': '019aebd1-e566-7009-9190-f6178cd6f9ac', 'organization': 'kling', 'provider': 'fal', 'publicName': 'kling-2.6-pro', 'name': 'kling-2.6-pro-image-to-video-fal', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 10, 'rankByModality': {'video': 10}}, {'id': '019aebd1-edc0-71c9-906d-82656c3d8927', 'organization': 'kling', 'provider': 'fal', 'publicName': 'kling-2.6-pro', 'name': 'kling-2.6-pro-text-to-video-fal', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 10, 'rankByModality': {'video': 10}}, {'id': '4cd188c8-4671-45a0-8433-dd89ce4e16a5', 'organization': 'kling', 'provider': 'kling', 'publicName': 'kling-2.5-turbo-1080p', 'name': 'kling-2.5-turbo-1080p', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 11, 'rankByModality': {'video': 11}}, {'id': 'fec7074d-a3a0-4f83-a487-82700fcec84d', 'organization': 'luma-ai', 'provider': 'luma', 'publicName': 'ray-3', 'name': 'ray-3', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 12, 'rankByModality': {'video': 12}}, {'id': '019a1e21-7cb0-7778-9dd0-02ed4fb3563f', 'organization': 'minimax', 'provider': 'minimax', 'publicName': 'hailuo-2.3', 'name': 'hailuo-2.3', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': False}}, 'outputCapabilities': {'video': True}}, 'rank': 13, 'rankByModality': {'video': 13}}, {'id': '019a997a-92dd-7588-ac3c-bdca7e81af37', 'organization': 'kandinsky', 'provider': 'kandinsky', 'publicName': 'kandinsky-5.0-t2v-pro', 'name': 'kandinsky-5.0-t2v-pro', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 14, 'rankByModality': {'video': 14}}, {'id': '55069e04-a634-4d98-8765-95113b945f5e', 'organization': 'minimax', 'provider': 'fal', 'publicName': 'hailuo-02-pro', 'name': 'hailuo-02-pro-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 15, 'rankByModality': {'video': 15}}, {'id': '527e3f88-c13f-404c-92b4-0dcf7eeb61e6', 'organization': 'minimax', 'provider': 'fal', 'publicName': 'hailuo-02-pro', 'name': 'hailuo-02-pro-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 15, 'rankByModality': {'video': 15}}, {'id': 'e705b65f-82cd-40cb-9630-d9e6ca92d06f', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedance-v1-pro', 'name': 'seedance-v1-pro-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 16, 'rankByModality': {'video': 16}}, {'id': '4ddc4e52-2867-49b6-a603-5aab24a566ca', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedance-v1-pro', 'name': 'seedance-v1-pro-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rank': 16, 'rankByModality': {'video': 16}}, {'id': 'bf03b5bb-8b4e-4a36-a893-0b2809f1daec', 'organization': 'minimax', 'provider': 'fal', 'publicName': 'hailuo-02-standard', 'name': 'hailuo-02-standard-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 17, 'rankByModality': {'video': 17}}, {'id': 'e652c45e-8699-4392-94f0-7834e7464137', 'organization': 'minimax', 'provider': 'fal', 'publicName': 'hailuo-02-standard', 'name': 'hailuo-02-standard-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 17, 'rankByModality': {'video': 17}}, {'id': 'd63b03fb-8bc8-4ed8-9a50-6ccb683ac2b1', 'organization': 'kling', 'provider': 'fal', 'publicName': 'kling-v2.1-master', 'name': 'kling-v2.1-master-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 18, 'rankByModality': {'video': 18}}, {'id': 'efdb7e05-2091-4e88-af9e-4ea6168d2f85', 'organization': 'kling', 'provider': 'fal', 'publicName': 'kling-v2.1-master', 'name': 'kling-v2.1-master-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 18, 'rankByModality': {'video': 18}}, {'id': '08d8dcc6-2ab5-45ae-9bf1-353480f1f7ee', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'veo-2', 'name': 'veo-2', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rank': 19, 'rankByModality': {'video': 19}}, {'id': '3a91bb37-39fb-471c-8aa2-a89b98d280d0', 'organization': 'alibaba', 'provider': 'fal', 'publicName': 'wan-v2.2-a14b', 'name': 'wan-v2.2-a14b-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rank': 20, 'rankByModality': {'video': 20}}, {'id': '264e6e2f-b66a-4e27-a859-8145ff32d6f6', 'organization': 'alibaba', 'provider': 'fal', 'publicName': 'wan-v2.2-a14b', 'name': 'wan-v2.2-a14b-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 20, 'rankByModality': {'video': 20}}, {'id': '4c8dde6e-1b2c-45b9-91c3-413b2ceafffb', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedance-v1-lite', 'name': 'seedance-v1-lite-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rank': 21, 'rankByModality': {'video': 21}}, {'id': '13ce11ba-def2-4c80-a70b-b0b2c14d293e', 'organization': 'bytedance', 'provider': 'fal', 'publicName': 'seedance-v1-lite', 'name': 'seedance-v1-lite-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 21, 'rankByModality': {'video': 21}}, {'id': '019a997a-88a7-7e5c-9214-ed2a946eb739', 'organization': 'kandinsky', 'provider': 'kandinsky', 'publicName': 'kandinsky-5.0-t2v-lite', 'name': 'kandinsky-5.0-t2v-lite', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 22, 'rankByModality': {'video': 22}}, {'id': 'c3d0e5c8-f4b3-417a-8cb8-2ccf757d3869', 'organization': 'openai', 'provider': 'azureOpenAI', 'publicName': 'sora', 'name': 'sora', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 23, 'rankByModality': {'video': 23}}, {'id': '217158f9-793e-4ffe-a197-6de9448432fc', 'organization': 'luma-ai', 'provider': 'fal', 'publicName': 'ray2', 'name': 'ray2', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 24, 'rankByModality': {'video': 24}}, {'id': '5b3383a9-6bca-4f71-8210-78895c9d84d5', 'organization': 'luma-ai', 'provider': 'fal', 'publicName': 'ray2', 'name': 'ray2-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rank': 24, 'rankByModality': {'video': 24}}, {'id': '86de5aea-fc0c-4c36-b65a-7afc443a32d2', 'organization': 'pika', 'provider': 'fal', 'publicName': 'pika-v2.2', 'name': 'pika-v2.2-text-to-video', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 25, 'rankByModality': {'video': 25}}, {'id': 'f9b9f030-9ebc-4765-bf76-c64a82a72dfd', 'organization': 'pika', 'provider': 'fal', 'publicName': 'pika-v2.2', 'name': 'pika-v2.2-image-to-video', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rank': 25, 'rankByModality': {'video': 25}}, {'id': 'f4809219-14a8-47fe-9705-8685085513e7', 'organization': 'genmo', 'provider': 'fal', 'publicName': 'mochi-v1', 'name': 'mochi-v1', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rank': 26, 'rankByModality': {'video': 26}}, {'id': '019b28b8-5801-708c-9a7c-e77999d624d6', 'organization': 'tencent', 'provider': 'tencent', 'publicName': 'hunyuan-video-1.5', 'name': 'hunyuan-video-1.5-t2v-20251209', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '019b28b8-671b-7d8c-9d32-0cd757a60a6c', 'organization': 'tencent', 'provider': 'tencent', 'publicName': 'hunyuan-video-1.5', 'name': 'hunyuan-video-1.5-i2v-20251209', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '58060613-41dc-478b-97a0-6d9c4f0c722a', 'organization': 'minimax', 'provider': 'fal', 'publicName': 'hailuo-02-fast', 'name': 'hailuo-02-fast', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '019a36dd-5d6c-7f15-8598-4755f4c34e28', 'organization': 'minimax', 'provider': 'minimax', 'publicName': 'hailuo-2.3-fast', 'name': 'hailuo-2.3-fast', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': 'ea96cfc8-953a-4c3c-a229-1107c55b7479', 'organization': 'kling', 'provider': 'fal', 'publicName': 'kling-v2.1-standard', 'name': 'kling-v2.1-standard', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '0754baa1-ab91-42d0-ba74-522aa8e5b8e2', 'organization': 'runway', 'provider': 'replicate', 'publicName': 'runway-gen4-turbo', 'name': 'runway-gen4-turbo', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '52f5e9b7-c36b-41e4-8e35-bce8e8d5b12b', 'publicName': 'nimble-bean', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '019a3c95-c431-79fe-9298-09cce3f449f3', 'organization': 'alibaba', 'provider': 'alibaba', 'publicName': 'wan2.5-i2v-preview', 'name': 'wan2.5-i2v-preview', 'capabilities': {'inputCapabilities': {'text': True, 'image': {'requiresUpload': True}}, 'outputCapabilities': {'video': True}}, 'rankByModality': {'video': 9007199254740991}}, {'id': '019abdb7-6957-71c1-96a2-bfa79e8a094f', 'organization': 'google', 'provider': 'googleVertexGlobalSearch', 'publicName': 'gemini-3-pro-grounding', 'name': 'gemini-3-pro-grounding', 'capabilities': {'inputCapabilities': {'text': True, 'image': True}, 'outputCapabilities': {'search': True}}, 'rank': 1, 'rankByModality': {'search': 1}}, {'id': '019b1448-f74a-72de-b25d-8666618f8c5a', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.2-search', 'name': 'gpt-5.2-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 2, 'rankByModality': {'search': 2}}, {'id': '019abdb7-50a5-7c05-9308-4491d069578b', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.1-search', 'name': 'gpt-5.1-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 3, 'rankByModality': {'search': 3}}, {'id': '019af19c-0658-7566-9c60-112ae5bdb8db', 'organization': 'xai', 'provider': 'xaiResponsesSearch', 'publicName': 'grok-4-1-fast-search', 'name': 'grok-4-1-fast-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 4, 'rankByModality': {'search': 4}}, {'id': '9217ac2d-91bc-4391-aa07-b8f9e2cf11f2', 'organization': 'xai', 'provider': 'xaiResearchSearch', 'publicName': 'grok-4-fast-search', 'name': 'grok-4-fast-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 5, 'rankByModality': {'search': 5}}, {'id': '24145149-86c9-4690-b7c9-79c7db216e5c', 'organization': 'perplexity', 'provider': 'perplexity', 'publicName': 'ppl-sonar-reasoning-pro-high', 'name': 'ppl-sonar-reasoning-pro-high', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 6, 'rankByModality': {'search': 6}}, {'id': 'fbe08e9a-3805-4f9f-a085-7bc38e4b51d1', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'o3-search', 'name': 'o3-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 7, 'rankByModality': {'search': 7}}, {'id': 'b222be23-bd55-4b20-930b-a30cc84d3afd', 'organization': 'google', 'provider': 'googleVertex', 'publicName': 'gemini-2.5-pro-grounding', 'name': 'gemini-2.5-pro-grounding', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 8, 'rankByModality': {'search': 8}}, {'id': '86d767b0-2574-4e47-a256-a22bcace9f56', 'organization': 'xai', 'provider': 'xaiSearch', 'publicName': 'grok-4-search', 'name': 'grok-4-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 9, 'rankByModality': {'search': 9}}, {'id': 'd942b564-191c-41c5-ae22-400a930a2cfe', 'organization': 'anthropic', 'provider': 'anthropicSearch', 'publicName': 'claude-opus-4-1-search', 'name': 'claude-opus-4-1-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 10, 'rankByModality': {'search': 10}}, {'id': 'd14d9b23-1e46-4659-b157-a3804ba7e2ef', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5-search', 'name': 'gpt-5-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 11, 'rankByModality': {'search': 11}}, {'id': 'c8711485-d061-4a00-94d2-26c31b840a3d', 'organization': 'perplexity', 'provider': 'perplexity', 'publicName': 'ppl-sonar-pro-high', 'name': 'ppl-sonar-pro-high', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 12, 'rankByModality': {'search': 12}}, {'id': '25bcb878-749e-49f4-ac05-de84d964bcee', 'organization': 'anthropic', 'provider': 'anthropicSearch', 'publicName': 'claude-opus-4-search', 'name': 'claude-opus-4-search', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 13, 'rankByModality': {'search': 13}}, {'id': '0862885e-ef53-4d0d-b9c4-4c8f68f453ce', 'organization': 'diffbot', 'provider': 'diffbot', 'publicName': 'diffbot-small-xl', 'name': 'diffbot-small-xl', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rank': 14, 'rankByModality': {'search': 14}}, {'id': '019b2f68-97ae-75b1-9e2f-456470bd5332', 'organization': 'openai', 'provider': 'openaiResponses', 'publicName': 'gpt-5.1-search-sp', 'name': 'gpt-5.1-search-new-system-prompt-20251217', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rankByModality': {'search': 9007199254740991}}, {'id': '019b094b-ff6f-728b-8517-a735d43ab2ac', 'publicName': 'stellarblade', 'capabilities': {'inputCapabilities': {'text': True}, 'outputCapabilities': {'search': True}}, 'rankByModality': {'search': 9007199254740991}}]


text_models = {model["publicName"]: model["id"] for model in models if
               "text" in model["capabilities"]["outputCapabilities"]}
image_models = {model["publicName"]: model["id"] for model in models if
                "image" in model["capabilities"]["outputCapabilities"]}
vision_models = [model["publicName"] for model in models if "image" in model["capabilities"]["inputCapabilities"]]

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
    url = "https://lmarena.ai"
    share_url = None
    create_evaluation = "https://lmarena.ai/nextjs-api/stream/create-evaluation"
    post_to_evaluation = "https://lmarena.ai/nextjs-api/stream/post-to-evaluation/{id}"
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
    image_cache = True
    _next_actions = {
        "generateUploadUrl":"7020462b741e358317f3b5a1929766d8b9c241c7c6",
        "getSignedUrl":"60ff7bb683b22dd00024c9aee7664bbd39749e25c9",
        "updateTouConsent": "40efff1040868c07750a939a0d8120025f246dfe28",
        "createPointwiseFeedback": "605a0e3881424854b913fe1d76d222e50731b6037b",
        "createPairwiseFeedback":"600777eb84863d7e79d85d214130d3214fc744c80f",
        "getProxyImage": "60049198d4936e6b7acc63719b63b89284c58683e6"
    }
    @classmethod
    def get_models(cls, timeout: int = None, **kwargs) -> list[str]:
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
                response = curl_cffi.get(f"{cls.url}/?mode=direct", **args, timeout=timeout)
                if response.ok:
                    for line in response.text.splitlines():
                        if "initialModels" in line:
                            line = line.split("initialModels", maxsplit=1)[-1].split("initialModelAId")[0][3:-3]
                            line = line.encode("utf-8").decode("unicode_escape")
                            models = json.loads(line)
                            cls.text_models = {model["publicName"]: model["id"] for model in models if
                                               "text" in model["capabilities"]["outputCapabilities"]}
                            cls.image_models = {model["publicName"]: model["id"] for model in models if
                                                "image" in model["capabilities"]["outputCapabilities"]}
                            cls.vision_models = [model["publicName"] for model in models if
                                                 "image" in model["capabilities"]["inputCapabilities"]]
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
    async def get_models_async(cls) -> list[str]:
        if not cls._models_loaded:
            async with StreamSession() as session:
                async with session.get(f"{cls.url}/?mode=direct",) as response:
                    await cls.__load_actions(await response.text())
        return cls.models

    @classmethod
    async def get_args_from_nodriver(cls, proxy, force=True):
        cache_file = cls.get_cache_file()
        grecaptcha = []


        async def is_auth(page:nodriver.Tab):
            cookies = {c.name: c.value for c in await page.send(nodriver.cdp.network.get_cookies([cls.url]))}
            return any("arena-auth-prod" in cookie for cookie in cookies)

        async def clear_cookies_for_url(browser: nodriver.Browser, url: str):
            debug.log(f"Clearing cookies for {url}")
            host = urlparse(url).hostname
            if not host:
                raise ValueError(f"Bad url: {url}")

            tab = browser.main_tab  # any open tab is fine
            cookies = await browser.cookies.get_all()  # returns CDP cookies :contentReference[oaicite:2]{index=2}
            for c in cookies:
                dom = (c.domain or "").lstrip(".")
                if dom and (host == dom or host.endswith("." + dom)):
                    if c.name == "cf_clearance":
                        continue
                    await tab.send(
                        cdp.network.delete_cookies(
                            name=c.name,
                            domain=dom,  # exact domain :contentReference[oaicite:3]{index=3}
                            path=c.path,  # exact path :contentReference[oaicite:4]{index=4}
                            # partition_key=c.partition_key,  # if you use partitioned cookies
                        )
                    )

        async def callback(page: nodriver.Tab):
            if force:
                await clear_cookies_for_url(page.browser, cls.url)
                await page.reload()
            button = await page.find("Accept Cookies")
            if button:
                await button.click()
            else:
                debug.log("No 'Accept Cookies' button found, skipping.")
            await asyncio.sleep(1)
            textarea = await page.select('textarea[name="message"]')
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
            await asyncio.sleep(1)
            element = await page.select('[style="display: grid;"]')
            if element:
                await click_trunstile(page, 'document.querySelector(\'[style="display: grid;"]\')')
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
            captcha = await page.evaluate(
                """window.grecaptcha.enterprise.execute('6Led_uYrAAAAAKjxDIF58fgFtX3t8loNAK85bW9I',  { action: 'chat_submit' }  );""",
                await_promise=True)
            grecaptcha.append(captcha)
            html = await page.get_content()
            await cls.__load_actions(html)

        args = await get_args_from_nodriver(cls.url, proxy=proxy, callback=callback)

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
                                '6Led_uYrAAAAAKjxDIF58fgFtX3t8loNAK85bW9I',
                                { action: 'chat_submit' }
                            );
                            resolve(token);
                        } catch (e) {
                            console.error("[LMArena API] reCAPTCHA execute failed:", e);
                            resolve(null);
                        }
                    });
                });""",
                await_promise=True)
            if isinstance(captcha, str):
                grecaptcha.append(captcha)
            else:
                raise Exception(captcha)
            html = await page.get_content()
            await cls.__load_actions(html)

        args = await get_args_from_nodriver(
            cls.url, proxy=proxy, callback=callback, cookies=args.get("cookies", {}), user_data_dir="grecaptcha",
            browser_args=["--guest", "--disable-gpu", "--no-sandbox"])

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
                cls.text_models = {model["publicName"]: model["id"] for model in models if
                                   "text" in model["capabilities"]["outputCapabilities"]}
                cls.image_models = {model["publicName"]: model["id"] for model in models if
                                    "image" in model["capabilities"]["outputCapabilities"]}
                cls.vision_models = [model["publicName"] for model in models if
                                     "image" in model["capabilities"]["inputCapabilities"]]
                cls.models = list(cls.text_models) + list(cls.image_models)
                cls.default_model = list(cls.text_models.keys())[0]
                cls._models_loaded = True
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
                                # if js_id != 5217:
                                #     continue
                                js_url = f"{cls.url}/_next/{js}"
                                async with session.get(js_url) as js_response:
                                    js_text = await js_response.text()
                                    if "generateUploadUrl" in js_text:
                                        # updateTouConsent, createPointwiseFeedback, createPairwiseFeedback, generateUploadUrl, getSignedUrl, getProxyImage
                                        start_id = re.findall(r'\("([a-f0-9]{40,})".*?"(\w+)"\)', js_text)
                                        for v, k in start_id:
                                            cls._next_actions[k] = v
                                        break

                elif chunk_data.startswith(("[", "{")):
                    try:
                        data = json.loads(chunk_data)
                        pars_data(data)
                    except json.decoder.JSONDecodeError:
                        ...

    @classmethod
    async def prepare_images(cls, args, media: list[tuple]) -> list[dict[str, str]]:
        files = []
        if not media:
            return files
        url = "https://lmarena.ai/?chat-modality=image"
        async with StreamSession(**args, ) as session:

            for index, (_file, file_name) in enumerate(media):

                data_bytes = to_bytes(_file)
                # Check Cache
                hasher = hashlib.md5()
                hasher.update(data_bytes)
                image_hash = hasher.hexdigest()
                file = ImagesCache.get(image_hash)
                if cls.image_cache and file:
                    debug.log("Using cached image")
                    files.append(file)
                    continue

                extension, file_type = detect_file_type(data_bytes)
                file_name = file_name or f"file-{len(data_bytes)}{extension}"
                # async with session.post(
                #         url="https://lmarena.ai/?chat-modality=image",
                #         json=[],
                #         headers={
                #             "accept": "text/x-component",
                #             "content-type": "text/plain;charset=UTF-8",
                #             "next-action": cls._next_actions["updateTouConsent"],
                #             "Referer": url
                #
                #         }
                # ) as response:
                #     await raise_for_status(response)

                async with session.post(
                        url="https://lmarena.ai/?chat-modality=image",
                        json=[file_name, file_type],
                        headers={
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": cls._next_actions["generateUploadUrl"],
                            "Referer": url

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
                        url=url,
                        json=[key],
                        headers={
                            "accept": "text/x-component",
                            "content-type": "text/plain;charset=UTF-8",
                            "next-action": cls._next_actions["getSignedUrl"],
                            "Referer": url
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
        args = kwargs.get("lmarena_args", {})
        grecaptcha = kwargs.pop("grecaptcha", "")
        if not args and cache_file.exists():
            try:
                with cache_file.open("r") as f:
                    args = json.load(f)
            except json.JSONDecodeError:
                debug.log(f"Cache file {cache_file} is corrupted, removing it.")
                cache_file.unlink()
                args = None
        force = False
        for _ in range(2):
            if args:
                pass
            elif has_nodriver or cls.share_url is None:
                args, grecaptcha = await cls.get_args_from_nodriver(proxy)

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
                # change to async
                await cls.get_models_async()
            is_image_model = model in cls.image_models
            if not model:
                model = cls.default_model
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
            if model in cls.text_models:
                model_id = cls.text_models[model]
            elif model in cls.image_models:
                model_id = cls.image_models[model]
            else:
                raise ModelNotFoundError(f"Model '{model}' is not supported by LMArena provider.")

            if conversation and getattr(conversation, "evaluationSessionId", None):
                url = cls.post_to_evaluation.format(id=conversation.evaluationSessionId)
                evaluationSessionId = conversation.evaluationSessionId
            else:
                url = cls.create_evaluation
                evaluationSessionId = str(uuid7())
            userMessageId = str(uuid7())
            modelAMessageId = str(uuid7())
            if not grecaptcha and has_nodriver:
                debug.log("get grecaptcha")
                args, grecaptcha = await cls.get_grecaptcha(args, proxy)
            files = await cls.prepare_images(args, media)
            data = {
                "id": evaluationSessionId,
                "mode": "direct",
                "modelAId": model_id,
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
                            elif line.startswith("ag:"):
                                chunk = json.loads(line[3:])
                                yield Reasoning(chunk)
                            elif line.startswith("a2:") and line == 'a2:[{"type":"heartbeat"}]':
                                # 'a2:[{"type":"heartbeat"}]'
                                continue
                            elif line.startswith("a2:"):
                                chunk = json.loads(line[3:])
                                __images = [image.get("image") for image in chunk if image.get("image")]
                                if __images:
                                    yield ImageResponse(__images, prompt)
                            elif line.startswith("ad:"):
                                yield JsonConversation(evaluationSessionId=evaluationSessionId)
                                finish = json.loads(line[3:])
                                if "finishReason" in finish:
                                    yield FinishReason(finish["finishReason"])
                                if "usage" in finish:
                                    yield Usage(**finish["usage"])
                            elif line.startswith("a3:"):
                                raise RuntimeError(f"LMArena: {json.loads(line[3:])}")
                            else:
                                debug.log(f"LMArena: Unknown line prefix: {line[:2]}")
                break
            except (CloudflareError, MissingAuthError) as error:
                args = None
                debug.error(error)
                debug.log(f"{cls.__name__}: Cloudflare error")
                continue
            except (RateLimitError) as error:
                args = None
                force = True
                debug.error(error)
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
