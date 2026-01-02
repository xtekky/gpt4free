from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
from typing import Dict, Optional, Union, AsyncIterable

from g4f.image import to_bytes, detect_file_type

try:
    import curl_cffi

    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

try:
    import nodriver
    from nodriver import cdp
    from nodriver.cdp.fetch import RequestPaused, RequestStage
    from ...requests.nodriver_ import clear_cookies_for_url, set_cookies_for_browser, wait_for_ready_state, \
        RequestInterception, Request, get_cookies, get_args
    import nodriver.cdp.io as io

    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...typing import AsyncResult, Messages, MediaListType
from ...requests import raise_for_status, merge_cookies, \
    get_nodriver
from ...requests.aiohttp import StreamSession
from ...errors import ModelNotFoundError, CloudflareError, MissingAuthError, RateLimitError
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

text_models = {'gemini-3-pro': '019a98f7-afcd-779f-8dcb-856cc3b3f078',
               'grok-4.1-thinking': '019a9389-a9d3-77a8-afbb-4fe4dd3d8630',
               'gemini-3-flash': '019b0ad8-856e-74fc-871c-86ccbb2b1d35',
               'claude-opus-4-5-20251101-thinking-32k': '019ab8b2-9bcf-79b5-9fb5-149a7c67b7c0',
               'claude-opus-4-5-20251101': '019adbec-8396-71cc-87d5-b47f8431a6a6',
               'grok-4.1': '019a9389-a4d8-748d-9939-b4640198302e',
               'gemini-3-flash (thinking-minimal)': '019b1424-f775-76d9-8c1b-dbb9c54ac4fb',
               'gpt-5.1-high': '019a8548-a2b1-70ce-b1be-eba096d41f58',
               'gemini-2.5-pro-grounding-exp': '51a47cc6-5ef9-4ac7-a59c-4009230d7564',
               'gemini-2.5-pro': '0199f060-b306-7e1f-aeae-0ebb4e3f1122',
               'claude-sonnet-4-5-20250929-thinking-32k': 'b0ea1407-2f92-4515-b9cc-b22a6d6c14f2',
               'claude-opus-4-1-20250805-thinking-16k': 'f1a2eb6f-fc30-4806-9e00-1efd0d73cbc4',
               'claude-sonnet-4-5-20250929': '019a2d13-28a5-7205-908c-0a58de904617',
               'claude-opus-4-1-20250805': '96ae95fd-b70d-49c3-91cc-b58c7da1090b',
               'gpt-5.2-high': '019b1448-dafa-7f92-90c3-50e159c2263c',
               'chatgpt-4o-latest-20250326': '0199c1e0-3720-742d-91c8-787788b0a19b',
               'gpt-5.2': '019b1448-d548-78f4-8b98-788d72cbd057', 'gpt-5.1': '019a7ebf-0f3f-7518-8899-fca13e32d9dc',
               'gpt-5-high': '983bc566-b783-4d28-b24c-3c8b08eb1086',
               'o3-2025-04-16': 'cb0f1e24-e8e9-4745-aabc-b926ffde7475',
               'qwen3-max-preview': '812c93cc-5f88-4cff-b9ca-c11a26599b0e',
               'grok-4-1-fast-reasoning': '019aa41a-0a13-714a-beb1-be4a918a4b56',
               'ernie-5.0-preview-1103': '019a4ca9-720d-75f5-9012-883ce8ff61df',
               'kimi-k2-thinking-turbo': '019a59bc-8bb8-7933-92eb-fe143770c211',
               'glm-4.6': 'f595e6f1-6175-4880-a9eb-377e390819e4', 'gpt-5-chat': '4b11c78c-08c8-461c-938e-5fc97d56a40d',
               'qwen3-max-2025-09-23': '98ad8b8b-12cd-46cd-98de-99edde7e03eb',
               'claude-opus-4-20250514-thinking-16k': '3b5e9593-3dc0-4492-a3da-19784c4bde75',
               'qwen3-235b-a22b-instruct-2507': 'ee7cb86e-8601-4585-b1d0-7c7380f8f6f4',
               'deepseek-v3.2-thinking': '019adb32-bb7a-77eb-882f-b8e3aaa2b2fd',
               'grok-4-fast-chat': '71023e9b-7361-498a-b6db-f2d2a83883fd',
               'mistral-large-3': '019acbac-df7c-73dc-9716-ebe040daaa4e',
               'kimi-k2-0905-preview': 'b88e983b-9459-473d-8bf1-753932f1679a',
               'deepseek-v3.2': '019adb32-b716-7591-9a2f-c6882973e340',
               'kimi-k2-0711-preview': '7a3626fc-4e64-4c9e-821f-b449a4b43b6a',
               'qwen3-vl-235b-a22b-instruct': '716aa8ca-d729-427f-93ab-9579e4a13e98',
               'gpt-4.1-2025-04-14': '14e9311c-94d2-40c2-8c54-273947e208b0',
               'claude-opus-4-20250514': 'ee116d12-64d6-48a8-88e5-b2d06325cdd2',
               'mistral-medium-2508': '27035fb8-a25b-4ec9-8410-34be18328afd',
               'grok-4-0709': 'b9edb8e9-4e98-49e7-8aaf-ae67e9797a11', 'glm-4.5': 'd079ef40-3b20-4c58-ab5e-243738dbada5',
               'gemini-2.5-flash': '0199f059-3877-7cfe-bc80-e01b1a4a83de',
               'gemini-2.5-flash-preview-09-2025': 'fc700d46-c4c1-4fec-88b5-f086876ae0bb',
               'claude-haiku-4-5-20251001': '0199e8e9-01ed-73e0-96ba-cf43b286bf10',
               'grok-4-fast-reasoning': '19b3730a-0369-49ba-ad9c-09e7337937f0',
               'qwen3-next-80b-a3b-instruct': '351fe482-eb6c-4536-857b-909e16c0bf52',
               'longcat-flash-chat': '6fcbe051-f521-4dc7-8986-c429eb6191bf',
               'qwen3-235b-a22b-no-thinking': '1a400d9a-f61c-4bc2-89b4-a9b7e77dff12',
               'claude-sonnet-4-20250514-thinking-32k': '4653dded-a46b-442a-a8fe-9bb9730e2453',
               'qwen3-235b-a22b-thinking-2507': '16b8e53a-cc7b-4608-a29a-20d4dac77cf2',
               'qwen3-vl-235b-a22b-thinking': '03c511f5-0d35-4751-aae6-24f918b0d49e',
               'gpt-5-mini-high': '5fd3caa8-fe4c-41a5-a22c-0025b58f4b42',
               'deepseek-v3-0324': '2f5253e4-75be-473c-bcfc-baeb3df0f8ad',
               'o4-mini-2025-04-16': 'f1102bbf-34ca-468f-a9fc-14bcf63f315b',
               'hunyuan-vision-1.5-thinking': '6a3a1e04-050e-4cb4-9052-b9ac4bec0c38',
               'claude-sonnet-4-20250514': 'ac44dd10-0666-451c-b824-386ccfea7bcc',
               'claude-3-7-sonnet-20250219-thinking-32k': 'be98fcfd-345c-4ae1-9a82-a19123ebf1d2',
               'qwen3-coder-480b-a35b-instruct': 'af033cbd-ec6c-42cc-9afa-e227fc12efe8',
               'hunyuan-t1-20250711': 'ba8c2392-4c47-42af-bfee-c6c057615a91',
               'mistral-medium-2505': '27b9f8c6-3ee1-464a-9479-a8b3c2a48fd4',
               'qwen3-30b-a3b-instruct-2507': 'a8d1d310-e485-4c50-8f27-4bff18292a99',
               'gpt-4.1-mini-2025-04-14': '6a5437a7-c786-467b-b701-17b0bc8c8231',
               'gemini-2.5-flash-lite-preview-09-2025-no-thinking': '75555628-8c14-402a-8d6e-43c19cb40116',
               'gemini-2.5-flash-lite-preview-06-17-thinking': '04ec9a17-c597-49df-acf0-963da275c246',
               'qwen3-235b-a22b': '2595a594-fa54-4299-97cd-2d7380d21c80',
               'claude-3-5-sonnet-20241022': 'f44e280a-7914-43ca-a25d-ecfcc5d48d09',
               'claude-3-7-sonnet-20250219': 'c5a11495-081a-4dc6-8d9a-64a4fd6f7bbc',
               'glm-4.5-air': '7bfb254a-5d32-4ce2-b6dc-2c7faf1d5fe8',
               'qwen3-next-80b-a3b-thinking': '73cf8705-98c8-4b75-8d04-e3746e1c1565',
               'minimax-m1': '87e8d160-049e-4b4e-adc4-7f2511348539',
               'gemma-3-27b-it': '789e245f-eafe-4c72-b563-d135e93988fc',
               'grok-3-mini-high': '149619f1-f1d5-45fd-a53e-7d790f156f20',
               'gemini-2.0-flash-001': '7a55108b-b997-4cff-a72f-5aa83beee918',
               'grok-3-mini-beta': '7699c8d4-0742-42f9-a117-d10e84688dab',
               'mistral-small-2506': 'bbad1d17-6aa5-4321-949c-d11fb6289241',
               'intellect-3': '019aebfd-af0e-7f0c-8f0d-96c588e4cd3b',
               'gpt-oss-120b': '6ee9f901-17b5-4fbe-9cc2-13c16497c23b',
               'glm-4.5v': '9dab0475-a0cc-4524-84a2-3fd25aa8c768',
               'command-a-03-2025': '0f785ba1-efcb-472d-961e-69f7b251c7e3',
               'amazon-nova-experimental-chat-10-20': '019a4c75-256c-790b-9088-4694cc63c507',
               'o3-mini': 'c680645e-efac-4a81-b0af-da16902b2541',
               'ling-flash-2.0': '71f96ca9-4cf8-4be7-bac2-2231613930a6',
               'minimax-m2': '019a27e0-e7d8-7b0b-877c-a2106c6eb87d', 'step-3': '1ea13a81-93a7-4804-bcdd-693cd72e302d',
               'gpt-5-nano-high': '2dc249b3-98da-44b4-8d1e-6666346a8012',
               'nova-2-lite': '019ae300-83b7-7717-a1e0-31accd1ff6fa', 'qwq-32b': '885976d3-d178-48f5-a3f4-6e13e0718872',
               'llama-4-maverick-17b-128e-instruct': 'b5ad3ab7-fc56-4ecd-8921-bd56b55c1159',
               'nvidia-nemotron-3-nano-30b-a3b-bf16': '019b0aa7-334a-78e8-b2a8-885f31f4fc0c',
               'qwen3-30b-a3b': '9a066f6a-7205-4325-8d0b-d81cc4b049c0',
               'claude-3-5-haiku-20241022': 'f6fbf06c-532c-4c8a-89c7-f3ddcfb34bd1',
               'ring-flash-2.0': '11ad4114-c868-4fed-b6e7-d535dc9c62f8',
               'llama-3.3-70b-instruct': 'dcbd7897-5a37-4a34-93f1-76a24c7bb028',
               'gemma-3n-e4b-it': '896a3848-ae03-4651-963b-7d8f54b61ae8',
               'gpt-oss-20b': 'ec3beb4b-7229-4232-bab9-670ee52dd711', 'mercury': '019a6f77-e20d-7c1d-a7cd-8bd926e7395d',
               'olmo-3-32b-think': '019ac2ef-27e1-769f-8258-d131f79e28ef',
               'magistral-medium-2506': '6337f479-2fc8-4311-a76b-8c957765cd68',
               'mistral-small-3.1-24b-instruct-2503': '69f5d38a-45f5-4d3a-9320-b866a4035ed9',
               'ibm-granite-h-small': '4ddb69f5-391a-4f78-af92-7d7328c18ab1',
               'anonymous-1111': '019aa97e-7b68-7c21-add9-b17edc20ff02',
               'lucky-lark': '019b2357-c7c5-74f0-96d3-bd98d6d33fa0',
               'ghostfalcon-20251215': '019b2498-86aa-7221-9963-a62a24a4703e',
               'fiercefalcon-20251215': '019b2498-89df-743c-8136-b6846f5d6df6',
               'evo-logic': '019adcc2-c05e-7b1a-a950-92b1ca78087b',
               'raptor-1123': '019ab9ad-caac-78f4-8007-bf0e082771d0',
               'raptor-1124': '019ab9ad-d128-7858-8745-6da5936ccb5e',
               'micro-mango': '019ade1a-a918-79dd-9ef9-ba22f97c977d',
               'x1-1-preview-0915': '42015285-534d-4e6b-9a9a-9061c2f73e1c',
               'qwen3-vl-8b-thinking': '0199e3d1-a308-77b9-a650-41453e8ef2fb',
               'qwen3-vl-8b-instruct': '0199e3d1-a713-7de2-a5dd-a1583cad9532',
               'ernie-exp-251024': '019a184d-efba-7743-979a-324e949aa1c4',
               'step-3-mini-2511': '019b287d-9ffc-7ce4-95be-d93929cdd94f',
               'EB45-turbo-vl-0906': 'b4a681ed-df4e-476f-89c6-a992a5783e60',
               'stephen-v2': '39b185cb-aba9-4232-99ea-074883a5ccd4',
               'dark-dragon': '019acb94-2eca-79a2-bc6c-585bf303df54',
               'x1-turbo-0906': '8d1f38a1-51a6-4030-ae4b-e19fb503e4fa',
               'raptor-1202': '019ae523-87e2-702f-8416-9266b20630d6',
               'frame-flow': '019ae5e0-1ba6-701e-a482-a4b8bb23fe36',
               'grok-4-1-fast-non-reasoning': '019aa41a-01bf-726c-913c-8a65b8f4c879',
               'raptor-llm-1205': '019afca4-2085-78ee-847a-e221cdb50501',
               'anonymous-1215': '019b2d23-7f4f-7544-891e-bb9c434f485c',
               'beluga-1128-1': '019aec73-6e86-784c-885a-7fe129025f95',
               'phantom-1203-1': '019af0b1-5c1d-7c24-8260-621ed4b32b96',
               'beluga-1128-2': '019b0923-102a-7b40-b9ba-95110a235709',
               'phantom-mm-1125-1': '019af1f8-aa23-7188-8a73-e717ae24a7d8',
               'holo-scope': '019b00de-7b51-7d3e-8d27-01b44b156c5d',
               'integrated-info': '019b0442-ff8e-72ac-99c0-9fc71eaf6db2',
               'beluga-1203-1': '019b0582-32cb-7f4b-9402-1f833c3b2a9e',
               'master-node': '019b2a43-6750-7119-808e-09c32d66af6a',
               'amazon.nova-pro-v1:0': 'a14546b5-d78d-4cf6-bb61-ab5b8510a9d6',
               'beluga-1202-1': '019b092f-6ecf-7945-9e1a-43dbe448fc1b',
               'cogilux': '019b2eea-8cab-7525-9e70-2be764ff8ef6',
               'gpt-5-high-no-system-prompt': '0199c1d5-51b8-7ead-a0a8-3f59234682fa',
               'EB45-vision': '638fb8b8-1037-4ee5-bfba-333392575a5d',
               'olmo-3.1-32b-think': '019b33ae-2ee4-7de8-89f3-d0fef62fe698',
               'fire-bird': '019b0e9a-c833-7e18-8a31-31c5a2415284',
               'beluga-1211-2': '019b33cd-0c2b-7b75-8238-0396e5928c38',
               'anonymous-1212': '019b0fe9-a2a2-7a2b-ad88-abfc70c4eaa3',
               'lmarena-internal-test-only': 'c15b93ed-e87b-467f-8f9f-d830fd7aa54d',
               'not-a-new-model': 'f1a5a6ab-e1b1-4247-88ac-49395291c1e3',
               'raptor-1.8-1208': '019b0bd7-efa8-7024-a3be-a2ee84cf4b66',
               'beluga-1214-1': '019b33ce-c65a-7d39-bfe7-a358dcd07a39',
               'mimo-v2-flash': '019b352d-36a8-7dd3-bbec-2bc78852d6c0',
               'EB45-turbo': 'e9dd5a96-c066-48b0-869f-eb762030b5ed',
               'raptor-vision-1015': '0199f437-08bd-7907-81e7-098e2eab750a',
               'silentnova': '019b00f1-3d28-7616-b27f-f4ebc9c25780', 'glm-4.6v': '019b151a-7c3b-72a2-8811-0bf9317c2ef5',
               'gpt-5.2-no-system-prompt': '019b1448-ff14-7c98-a1ac-726fece799ec',
               'gpt-5.2-high-no-system-prompt': '019b1449-0313-7911-b836-419e2ed79b2e',
               'glm-4.6v-flash': '019b1536-49c0-73b2-8d45-403b8571568d',
               'leepwal': '9af435c8-1f53-4b78-a400-c1f5e9fe09b0', 'blackhawk': '019a33c6-e248-7091-94c0-af68f29390b8',
               'jet-force': '019b1644-8045-7af4-a02a-1bb21cd26344',
               'kernel-sense': '019b1699-ff5f-7f8d-a158-9b04cc99737e',
               'december-chatbot2': '019b1994-ac41-7f07-8f05-bbaeaf9aaea8',
               'qwen3-max-2025-09-26': 'ac31e980-8bf1-4637-adba-cf9ffa8b6343',
               'raptor': 'f23d6df4-4395-4404-897f-bdedc909e783', 'winter-wind': '019a6617-4f6a-77ab-af79-097f7aa7c86d',
               'MiMo-7B': 'ee3588cd-1fe1-484a-bcc9-f92065b8380c',
               'qwen3-omni-flash': '0199c9dc-e157-7458-bd49-5942363be215',
               'qwen3-max-thinking': '019a3aeb-371d-7474-a1b7-b8f4207a9a2c',
               'stephen-vision-csfix': 'e3c9ea42-5f42-496b-bc80-c7e8ee5653cc',
               'monster': '0304b4de-e544-48d4-8490-ad9123bc26e3',
               'ernie-exp-vl-251016': '0199eb70-db25-73ea-b944-f18fe0c3c0cd',
               'qwen-vl-max-2025-08-13': '6fe1ec40-3219-4c33-b3e7-0e65658b4194',
               'ling-1t': '0199edd2-88be-76b8-9aaa-5fc6b9c53503', 'monterey': 'd8444b25-e302-4d3b-8561-7ded087ee03c',
               'sunshine-ai': '019a3e64-eae6-7178-a9ae-2894c25ada46', 'gauss': '019a5658-ba6f-7185-b79d-51516ef94314',
               'qwen3-max-2025-10-20': '019a026f-30b2-7ffa-9714-7180577666e2',
               'minimax-m2-preview': '019a17b5-5e1e-7df6-9e1a-6c4338f8b6ff',
               'whisperfall': '019a7b93-af36-7f06-bb46-2b6b0548fc93', 'neon': '019a842c-7e62-725a-8a2a-f76f5ec786dd',
               'viper': '019a6c82-2ead-7475-8612-55dabba11c31',
               'ernie-5.0-preview-1120': '019a19ad-97fe-72fb-89ad-d02355152e55',
               'flying-octopus': '0199de70-e6ad-7276-9020-a7502bed99ad',
               'ernie-exp-251023': '019a19ad-8f50-7282-8cb9-5e49b9052aff',
               'ernie-exp-251027': '019a26b4-2a20-7864-b1f7-d471ec1f4e3f',
               'nightride-on': '48fe3167-5680-4903-9ab5-2f0b9dc05815',
               'ring-1t': '019a19ad-a47e-7828-aa1f-5c7f2c4f65fe',
               'ernie-exp-251025': '019a1dce-b0d7-75cc-8742-997a0639f061',
               'ernie-exp-251026': '019a26b4-2540-7a7d-96bf-b6b377e8d1ec',
               'nightride-on-v2': 'c822ec98-38e9-4e43-a434-982eb534824f',
               'raptor-llm-1017': '0199fdfb-8d7f-7435-a362-e68678c308c2',
               'raptor-llm-1024': '019a26b4-2f39-7a0e-818d-254d0f37a85a',
               'newton': '019a5658-b301-7c8e-acb2-707cad8d6177', 'aegis-core': '019a4ca3-54ee-7c78-9cbf-19584ddfb7be',
               'raptor-vision-1107': '019a7011-49dd-7e6e-94e1-ced43eac205c',
               'raptor-1110': '019a7904-198f-71dc-b671-b9e39b950a37',
               'rain-drop': '019a6617-53aa-7d9f-940d-94afc98aab1f', 'redwood': '6c98ce8c-41d1-42cd-b2e3-292c6519add5',
               'gpt-5-high-new-system-prompt': '19ad5f04-38c6-48ae-b826-f7d5bbfd79f7',
               'MiMo-VL-7B-RL-2508': '1c0259b5-dff7-48ce-bca1-b6957675463b',
               'raptor-1119': '019a9e80-e469-722e-8f9a-585860a5859d',
               'ling-1t-1031': '019aa3fb-ec59-7b49-af33-57d0981509e0'}
image_models = {'gpt-image-1.5': '019b2893-2508-7686-a63d-4861997565d2',
                'gemini-3-pro-image-preview-2k (nano-banana-pro)': '019abc10-e78d-7932-b725-7f1563ed8a12',
                'gemini-3-pro-image-preview (nano-banana-pro)': '019aa208-5c19-7162-ae3b-0a9ddbb1e16a',
                'flux-2-max': '019b2a10-15dd-78c6-b8ae-bafea83efd6e',
                'flux-2-flex': '019abed6-d96e-7a2b-bf69-198c28bef281',
                'gemini-2.5-flash-image-preview (nano-banana)': '0199ef2a-583f-7088-b704-b75fd169401d',
                'flux-2-pro': '019abcf4-5600-7a8b-864d-9b8ab7ab7328',
                'hunyuan-image-3.0': '7766a45c-1b6b-4fb8-9823-2557291e1ddd',
                'flux-2-dev': '019ae6a0-4773-77d5-8ffb-cc35813e063c',
                'seedream-4.5': '019abd43-b052-7eec-aa57-e895e45c9723',
                'wan2.5-t2i-preview': '019a5050-2875-78ed-ae3a-d9a51a438685',
                'wan2.5-preview': '019a5050-1695-7b5d-b045-ba01ff08ec28',
                'seedream-4-high-res-fal': '32974d8d-333c-4d2e-abf3-f258c0ac1310',
                'gpt-image-1-high-fidelity': '69f90b32-01dc-43e1-8c48-bf494f8f4f38',
                'gpt-image-1': '6e855f13-55d7-4127-8656-9168a9f4dcc0',
                'gpt-image-1-mini': '0199c238-f8ee-7f7d-afc1-7e28fcfd21cf',
                'mai-image-1': '1b407d5c-1806-477c-90a5-e5c5a114f3bc',
                'seedream-3': 'd8771262-8248-4372-90d5-eb41910db034',
                'flux-1-kontext-max': '0633b1ef-289f-49d4-a834-3d475a25e46b',
                'qwen-image-prompt-extend': '9fe82ee1-c84f-417f-b0e7-cab4ae4cf3f3',
                'flux-1-kontext-pro': '28a8f330-3554-448c-9f32-2c0a08ec6477',
                'imagen-3.0-generate-002': '51ad1d79-61e2-414c-99e3-faeb64bb6b1b',
                'ideogram-v3-quality': '73378be5-cdba-49e7-b3d0-027949871aa6',
                'photon': 'e7c9fa2d-6f5d-40eb-8305-0980b11c7cab', 'recraft-v3': 'b88d5814-1d20-49cc-9eb6-e362f5851661',
                'lucid-origin': '5a3b3520-c87d-481f-953c-1364687b6e8f',
                'gemini-2.0-flash-preview-image-generation': '69bbf7d4-9f44-447e-a868-abc4f7a31810',
                'dall-e-3': 'bb97bc68-131c-4ea4-a59e-03a6252de0d2',
                'flux-1-kontext-dev': 'eb90ae46-a73a-4f27-be8b-40f090592c9a',
                'tangerine': '019aafce-f609-727d-a998-0fa521fcb1d5',
                'vidu-q2-image': '019adb32-afa4-749e-9992-39653b52fe13',
                'reve-v1.1-fast': '019b207a-d836-7742-a312-b2b247b2366e',
                'imagen-4.0-fast-generate-001': 'f44fd4f8-af30-480f-8ce2-80b2bdfea55e',
                'gemini-3-pro-image-preview-4k (nano-banana-pro)': '019ae2a3-304d-773b-b333-daa050124a92',
                'imagen-4.0-ultra-generate-001': '019ae6da-6438-7077-9d2d-b311a35645f8',
                'imagen-4.0-generate-001': '019ae6da-6788-761a-8253-e0bb2bf2e3a9',
                'wan2.5-i2i-preview': '019aeb62-c6ea-788e-88f9-19b1b48325b5',
                'chatgpt-image-latest (20251216)': '019b2eac-06ad-7e71-a9ac-67a6765f2619',
                'hazel-small-2': '019b1093-93d0-7655-a29b-c00b1a4104d4',
                'hazel-gen-2': '019b00fe-89e8-7fda-9f6c-36d16813fb48',
                'seededit-3.0': 'e2969ebb-6450-4bc4-87c9-bbdcf95840da',
                'hunyuan-image-2.1': 'a9a26426-5377-4efa-bef9-de71e29ad943',
                'hazel-gen-4': '019b00ff-a0f5-71e9-b43c-d02661323dfc',
                'hidream-e1.1': '32bff2df-00e6-409b-ad3f-bfbad87cc49f',
                'reve-v1.1': '019b169a-09c6-79ea-9400-1566d793ade1',
                'qwen-image-edit': '995cf221-af30-466d-a809-8e0985f83649',
                'hunyuan-image-3.0-fal': '0199e980-d247-7dd3-9ca1-77092f126f05'}
vision_models = ['gemini-3-pro', 'gemini-3-flash', 'gemini-3-flash (thinking-minimal)', 'gpt-5.1-high',
                 'gemini-2.5-pro', 'gpt-5.2-high', 'chatgpt-4o-latest-20250326', 'gpt-5.2', 'gpt-5.1', 'gpt-5-high',
                 'o3-2025-04-16', 'gpt-5-chat', 'qwen3-vl-235b-a22b-instruct', 'gpt-4.1-2025-04-14',
                 'mistral-medium-2508', 'grok-4-0709', 'gemini-2.5-flash', 'gemini-2.5-flash-preview-09-2025',
                 'qwen3-vl-235b-a22b-thinking', 'gpt-5-mini-high', 'o4-mini-2025-04-16', 'hunyuan-vision-1.5-thinking',
                 'mistral-medium-2505', 'gpt-4.1-mini-2025-04-14', 'gemini-2.5-flash-lite-preview-09-2025-no-thinking',
                 'gemini-2.5-flash-lite-preview-06-17-thinking', 'gemma-3-27b-it', 'gemini-2.0-flash-001',
                 'mistral-small-2506', 'glm-4.5v', 'step-3', 'gpt-5-nano-high', 'llama-4-maverick-17b-128e-instruct',
                 'mistral-small-3.1-24b-instruct-2503', 'anonymous-1111', 'ghostfalcon-20251215',
                 'fiercefalcon-20251215', 'raptor-1123', 'raptor-1124', 'qwen3-vl-8b-thinking', 'qwen3-vl-8b-instruct',
                 'step-3-mini-2511', 'EB45-turbo-vl-0906', 'raptor-1202', 'phantom-mm-1125-1', 'amazon.nova-pro-v1:0',
                 'gpt-5-high-no-system-prompt', 'EB45-vision', 'raptor-1.8-1208', 'raptor-vision-1015', 'glm-4.6v',
                 'glm-4.6v-flash', 'jet-force', 'raptor', 'qwen3-omni-flash', 'stephen-vision-csfix',
                 'ernie-exp-vl-251016', 'qwen-vl-max-2025-08-13', 'ernie-5.0-preview-1120', 'nightride-on',
                 'nightride-on-v2', 'aegis-core', 'raptor-vision-1107', 'raptor-1110', 'gpt-5-high-new-system-prompt',
                 'MiMo-VL-7B-RL-2508', 'raptor-1119', 'gpt-image-1.5',
                 'gemini-3-pro-image-preview-2k (nano-banana-pro)', 'gemini-3-pro-image-preview (nano-banana-pro)',
                 'flux-2-max', 'flux-2-flex', 'gemini-2.5-flash-image-preview (nano-banana)', 'flux-2-pro',
                 'flux-2-dev', 'seedream-4.5', 'seedream-4-high-res-fal', 'gpt-image-1-high-fidelity', 'gpt-image-1',
                 'gpt-image-1-mini', 'flux-1-kontext-max', 'flux-1-kontext-pro',
                 'gemini-2.0-flash-preview-image-generation', 'flux-1-kontext-dev', 'vidu-q2-image', 'reve-v1.1-fast',
                 'gemini-3-pro-image-preview-4k (nano-banana-pro)', 'wan2.5-i2i-preview',
                 'chatgpt-image-latest (20251216)', 'hazel-small-2', 'seededit-3.0', 'hidream-e1.1', 'reve-v1.1',
                 'qwen-image-edit', 'veo-3.1-fast-audio', 'veo-3.1-audio', 'veo-3-fast-audio', 'veo-3-audio',
                 'wan2.5-t2v-preview', 'veo-3-fast', 'veo-3', 'kling-2.6-pro', 'kling-2.5-turbo-1080p', 'ray-3',
                 'hailuo-2.3', 'hailuo-02-pro', 'seedance-v1-pro', 'hailuo-02-standard', 'kling-v2.1-master', 'veo-2',
                 'wan-v2.2-a14b', 'seedance-v1-lite', 'ray2', 'pika-v2.2', 'hunyuan-video-1.5', 'hailuo-02-fast',
                 'hailuo-2.3-fast', 'kling-v2.1-standard', 'runway-gen4-turbo', 'nimble-bean', 'wan2.5-i2v-preview',
                 'gemini-3-pro-grounding']

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
    image_cache = False
    _next_actions = {
        "generateUploadUrl": "7020462b741e358317f3b5a1929766d8b9c241c7c6",
        "getSignedUrl": "60ff7bb683b22dd00024c9aee7664bbd39749e25c9",
        "updateTouConsent": "40efff1040868c07750a939a0d8120025f246dfe28",
        "createPointwiseFeedback": "605a0e3881424854b913fe1d76d222e50731b6037b",
        "createPairwiseFeedback": "600777eb84863d7e79d85d214130d3214fc744c80f",
        "getProxyImage": "60049198d4936e6b7acc63719b63b89284c58683e6"
    }
    lock = asyncio.Lock()
    nodriver = None
    stop_nodriver = None

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        timeout: Optional[int] = kwargs.get("timeout", None)
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
    async def _wait_auth(cls, page, prompt="Hello"):
        async def is_auth(page: nodriver.Tab):
            # cookies = {c.name: c.value for c in await page.send(nodriver.cdp.network.get_cookies([cls.url]))}
            # return any("arena-auth-prod" in cookie for cookie in cookies)
            return await page.evaluate('document.cookie.indexOf("arena-auth-prod-v1") >= 0')

        if not await is_auth(page):
            button = await page.find_element_by_text("Accept Cookies")
            if button:
                await button.click()
            else:
                debug.log("No 'Accept Cookies' button found, skipping.")
            await asyncio.sleep(1)
            textarea = await page.select('textarea[name="message"]')
            if textarea:
                await textarea.send_keys(prompt)
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
            element = await page.query_selector('[style="display: grid;"]')
            if element:
                await click_trunstile(page, 'document.querySelector(\'[style="display: grid;"]\')')
            if not await is_auth(page):
                debug.log("No authentication cookie found, trying to authenticate.")
                await page.select('#cf-turnstile', 300)
                debug.log("Found Element: 'cf-turnstile'")
                await asyncio.sleep(3)
                await click_trunstile(page)
            while not await is_auth(page):
                await asyncio.sleep(1)
            while not await page.evaluate('!!document.querySelector(\'textarea\')'):
                await asyncio.sleep(1)

        html = await page.get_content()
        await cls.__load_actions(html)

    @classmethod
    async def _get_captcha(cls, page):
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
        if not isinstance(captcha, str):
            raise Exception(captcha)
        return captcha

    @classmethod
    async def _prepare_data(cls, prompt, conversation, model, captcha, files):
        # prepare request
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
            "recaptchaV3Token": captcha,
            # "recaptchaV2Token": grecaptcha
        }
        return data, url, evaluationSessionId

    @classmethod
    async def _prepare_images(cls, page: nodriver.Tab, media: list[tuple]) -> list[dict[str, str]]:
        files = []
        if not media:
            return files
        url = "https://lmarena.ai/?chat-modality=image"

        for index, (_file, file_name) in enumerate(media):

            data_bytes: bytes = to_bytes(_file)
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
            data = json.dumps([file_name, file_type])
            headers = {
                "accept": "text/x-component",
                "content-type": "text/plain;charset=UTF-8",
                "next-action": cls._next_actions["generateUploadUrl"],
                "Referer": url

            }
            # Prepare your data and headers in Python
            headers_json = json.dumps(headers)
            data_json = json.dumps(data)

            # Use a cleaner script structure
            script = f"""
                (async () => {{
                    const response = await fetch("{url}", {{
                        method: "POST",
                        headers: {headers_json},
                        body: {data_json}
                    }});
                    return await response.text();
                }})()
                """
            text = await page.evaluate(script, await_promise=True)
            line = next(filter(lambda x: x.startswith("1:"), text.split("\n")), "")
            if not line:
                raise Exception("Failed to get upload URL")
            chunk = json.loads(line[2:])
            if not chunk.get("success"):
                raise Exception(f"Failed to get upload URL: {chunk}")
            uploadUrl = chunk.get("data", {}).get("uploadUrl")
            key = chunk.get("data", {}).get("key")
            if not uploadUrl:
                raise Exception("Failed to get upload URL")

            # Prepare your data and headers in Python
            headers_json = json.dumps({
                "content-type": file_type,
            })
            # data_json = json.dumps(data_bytes)
            base64_data = base64.b64encode(data_bytes).decode('utf-8')
            script = f"""
                (async () => {{
                    try {{
                        // Convert Base64 string back to a Blob/Buffer
                        const base64Data = "{base64_data}";
                        const byteCharacters = atob(base64Data);
                        const byteNumbers = new Array(byteCharacters.length);
                        for (let i = 0; i < byteCharacters.length; i++) {{
                            byteNumbers[i] = byteCharacters.charCodeAt(i);
                        }}
                        const byteArray = new Uint8Array(byteNumbers);
                        const response = await fetch("{uploadUrl}", {{
                            method: "PUT",
                            headers: {headers_json},
                            body: byteArray
                        }});

                        // Extract the body as text
                        const body = await response.text();

                        // Return a custom object with the info you need
                        return {{
                            "status": response.status,
                            // "statusText": response.statusText,
                            "url": response.url,
                            // "headers": Object.fromEntries(response.headers.entries()),
                            "data": body
                        }};
                    }} catch (error) {{
                        return {{ "error": error.message }};
                    }}
                }})()
                """
            put_response = await page.evaluate(script, await_promise=True)
            print(put_response[0][1]["value"])
            headers_json = json.dumps({
                "accept": "text/x-component",
                "content-type": "text/plain;charset=UTF-8",
                "next-action": cls._next_actions["getSignedUrl"],
                "Referer": url
            })
            data_json = json.dumps([key])
            script = f"""
                (async () => {{
                    const response = await fetch("{url}", {{
                        method: "POST",
                        headers: {headers_json},
                        body: {data_json}
                    }});
                    // Extract the body as text
                    const body = await response.text();

                    // Return a custom object with the info you need
                    return {{
                        "status": response.status,
                        // "statusText": response.statusText,
                        "url": response.url,
                        // "headers": Object.fromEntries(response.headers.entries()),
                        "data": body
                    }};
                }})()
                """
            print(script)

            post_response = await page.evaluate(script, await_promise=True)
            print(post_response)
            text = post_response[2][1]["value"]
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

            # debug.log(f"Uploaded image to: {image_url}")
            # ImagesCache[image_hash] = uploaded_file
            # files.append(uploaded_file)
        return files

    @classmethod
    async def create_async_browser(
            cls,
            model: str,
            messages: Messages,
            conversation: JsonConversation = None,
            media: MediaListType = None,
            proxy: str = None,
            timeout: int = None,
            **kwargs) -> AsyncResult:

        prompt = get_last_user_message(messages)

        async with cls.lock:
            try:
                if cls.nodriver is None:
                    cls.nodriver, cls.stop_nodriver = await get_nodriver(proxy=proxy)

                await clear_cookies_for_url(cls.nodriver, cls.url, ["cf_clearance"])
                if kwargs.get("cookies"):
                    await set_cookies_for_browser(cls.nodriver, kwargs.get("cookies"), cls.url)
                page: nodriver.Tab = await cls.nodriver.get(cls.url)
                await wait_for_ready_state(page, raise_error=False)
                while not await page.evaluate("!!document.querySelector('body:not(.no-js)')"):
                    await asyncio.sleep(1)

                # Check user login and load models and actions
                await cls._wait_auth(page, prompt=prompt)
                # Get grecaptcha
                grecaptcha = await cls._get_captcha(page)
                # TODO:
                args = await get_args(cls.nodriver, cls.url)
                files = await cls.prepare_images(page, media)
                data, url, evaluationSessionId = await cls._prepare_data(prompt, conversation, model, grecaptcha, files)
                yield JsonRequest.from_dict(data)

                script = f"""
                (async () => {{
                    const data = {data};
                    const response = await fetch("{url}", {{
                        method: "POST",
                        body: JSON.stringify(data)
                    }});
                    return response
                }})()
                """

                # Option 1 - Stream
                async with RequestInterception(page, url, RequestStage.RESPONSE) as response:
                    await page.evaluate(script)
                    intercepted_request: Request = await response.response_future
                    # 429
                    if intercepted_request.response_status_code != 200:
                        raise Exception(intercepted_request.response_error_reason)

                    response_body = ""
                    async for chunk_raw in intercepted_request.response_body_as_stream:
                        response_body += chunk_raw

                # Option 2 -
                # async with page.expect_response(url) as response:
                #     await page.evaluate(script)
                #     await response.value
                #     response_body, _ = await response.response_body
                #     if (await response.response).status != 200:
                #         raise Exception(response_body)

                async for chunk in cls._process_stream_response_(response_body.split("\n"), prompt, conversation):
                    yield chunk
            finally:
                if cls.stop_nodriver:
                    cls.stop_nodriver()

    @classmethod
    async def _process_stream_response_(
            cls,
            response_content: Union[list, AsyncIterable],
            prompt: str,
            conversation
    ) -> AsyncResult:
        # 1. Normalize the input into an async iterator
        async def get_chunks():
            if hasattr(response_content, '__aiter__'):
                async for chunk in response_content:
                    yield chunk
            else:
                for chunk in response_content:
                    yield chunk

        async for chunk in get_chunks():
            line = chunk
            if isinstance(line, bytes):
                line = line.decode()

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
                # yield JsonConversation(evaluationSessionId=conversation.evaluationSessionId)
                finish = json.loads(line[3:])
                if "finishReason" in finish:
                    yield FinishReason(finish["finishReason"])
                if "usage" in finish:
                    yield Usage(**finish["usage"])
            elif line.startswith("a3:"):
                raise RuntimeError(f"LMArena: {json.loads(line[3:])}")
            else:
                debug.log(f"LMArena: Unknown line prefix: {line[:2]}")

    ############################## create_async_generator ####################

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
                        raise Exception(f"Failed to get upload URL: {chunk}")
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
    async def get_models_async(cls) -> list[str]:
        if not cls._models_loaded:
            async with StreamSession() as session:
                async with session.get(f"{cls.url}/?mode=direct", ) as response:
                    await cls.__load_actions(await response.text())
        return cls.models

    @classmethod
    async def get_args_from_nodriver(cls, proxy, cookies=None):
        cache_file = cls.get_cache_file()
        try:
            if cls.nodriver is None:
                cls.nodriver, cls.stop_nodriver = await get_nodriver(proxy=proxy)
            await clear_cookies_for_url(cls.nodriver, cls.url, ["cf_clearance"])
            if cookies:
                await set_cookies_for_browser(cls.nodriver, cookies, cls.url)
            page: nodriver.Tab = await cls.nodriver.get(cls.url)
            await wait_for_ready_state(page, raise_error=False)
            while not await page.evaluate("!!document.querySelector('body:not(.no-js)')"):
                await asyncio.sleep(1)

            # Check user login and load models and actions
            await cls._wait_auth(page)
            captcha = await cls._get_captcha(page)
            args = await get_args(cls.nodriver, cls.url)

        finally:
            if cls.stop_nodriver:
                cls.stop_nodriver()

        args["impersonate"] = "chrome136"

        with cache_file.open("w") as f:
            json.dump(args, f)

        return args, captcha

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
                args = {}
        cookies = kwargs.get("cookies", {})
        for _ in range(2):
            if has_nodriver:
                args, grecaptcha = await cls.get_args_from_nodriver(proxy, cookies=cookies or args.get("cookies"))

            if not cls._models_loaded:
                # change to async
                await cls.get_models_async()

            files = await cls.prepare_images(args, media)
            data, url, evaluationSessionId = await cls._prepare_data(prompt, conversation, model, grecaptcha, files)
            yield JsonRequest.from_dict(data)
            try:
                async with StreamSession(**args, timeout=timeout or 5 * 60) as session:
                    async with session.post(
                            url,
                            json=data,
                            proxy=proxy,
                            # impersonate="chrome136"
                    ) as response:
                        await raise_for_status(response)
                        args["cookies"] = merge_cookies(args["cookies"], response)
                        async for chunk in cls._process_stream_response_(response.iter_lines(), prompt,
                                                                         conversation):
                            yield chunk

                break
            except (CloudflareError, MissingAuthError) as error:
                args = None
                cookies.clear()
                debug.error(error)
                debug.log(f"{cls.__name__}: Cloudflare error")
                continue
            except (RateLimitError) as error:
                args = None
                cookies.clear()
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
