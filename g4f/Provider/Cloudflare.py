from __future__ import annotations

import asyncio
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..errors import ModelNotFoundError
from .. import debug
from .helper import render_messages

def clean_name(name: str) -> str:
    return name.split("/")[-1].replace(
        "-instruct", "").replace(
        "-17b-16e", "").replace(
        "-chat", "").replace(
        "-fp8", "").replace(
        "-fast", "").replace(
        "-int8", "").replace(
        "-awq", "").replace(
        "-qvq", "").replace(
        "-r1", "").replace(
        "meta-llama-", "llama-").replace(
        "-it", "").replace(
        "qwen-", "qwen").replace(
        "qwen", "qwen-")

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    working = True
    use_nodriver = False
    active_by_default = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = 'llama-3.3-70b'
    model_aliases = {
        'deepseek-coder-6.7b': '@hf/thebloke/deepseek-coder-6.7b-instruct-awq',
        'deepseek-coder-6.7b-base': '@hf/thebloke/deepseek-coder-6.7b-base-awq',
        'deepseek-distill-qwen-32b': '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
        'deepseek-math-7b': '@cf/deepseek-ai/deepseek-math-7b-instruct',
        'discolm-german-7b-v1': '@cf/thebloke/discolm-german-7b-v1-awq',
        'falcon-7b': '@cf/tiiuae/falcon-7b-instruct',
        'gemma-2b-lora': '@cf/google/gemma-2b-it-lora',
        'gemma-3-12b': '@cf/google/gemma-3-12b-it',
        'gemma-4-26b-a4b-it': '@cf/google/gemma-4-26b-a4b-it',
        'gemma-7b': '@hf/google/gemma-7b-it',
        'gemma-7b-it-lora': '@cf/google/gemma-7b-it-lora',
        'gemma-sea-lion-v4-27b': '@cf/aisingapore/gemma-sea-lion-v4-27b-it',
        'glm-4.7-flash': '@cf/zai-org/glm-4.7-flash',
        'glm-5.2': '@cf/zai-org/glm-5.2',
        'gpt-oss-120b': '@cf/openai/gpt-oss-120b',
        'gpt-oss-20b': '@cf/openai/gpt-oss-20b',
        'granite-4.0-h-micro': '@cf/ibm-granite/granite-4.0-h-micro',
        'hermes-2-pro-mistral-7b': '@hf/nousresearch/hermes-2-pro-mistral-7b',
        'kimi-k2.6': '@cf/moonshotai/kimi-k2.6',
        'kimi-k2.7-code': '@cf/moonshotai/kimi-k2.7-code',
        'llama-2-13b': '@hf/thebloke/llama-2-13b-chat-awq',
        'llama-2-7b': '@cf/meta/llama-2-7b-chat-int8',
        'llama-2-7b-fp16': '@cf/meta/llama-2-7b-chat-fp16',
        'llama-2-7b-lora': '@cf/meta-llama/llama-2-7b-chat-hf-lora',
        'llama-3-8b': '@hf/meta-llama/meta-llama-3-8b-instruct',
        'llama-3.1-8b': '@cf/meta/llama-3.1-8b-instruct-fp8',
        'llama-3.2-11b-vision': '@cf/meta/llama-3.2-11b-vision-instruct',
        'llama-3.2-1b': '@cf/meta/llama-3.2-1b-instruct',
        'llama-3.2-3b': '@cf/meta/llama-3.2-3b-instruct',
        'llama-3.3-70b': '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
        'llama-4-scout': '@cf/meta/llama-4-scout-17b-16e-instruct',
        'llama-guard-3-8b': '@cf/meta/llama-guard-3-8b',
        'llamaguard-7b': '@hf/thebloke/llamaguard-7b-awq',
        'mistral-7b-v0.1': '@hf/thebloke/mistral-7b-instruct-v0.1-awq',
        'mistral-7b-v0.2': '@hf/mistral/mistral-7b-instruct-v0.2',
        'mistral-7b-v0.2-lora': '@cf/mistral/mistral-7b-instruct-v0.2-lora',
        'mistral-small-3.1-24b': '@cf/mistralai/mistral-small-3.1-24b-instruct',
        'nemotron-3-120b': '@cf/nvidia/nemotron-3-120b-a12b',
        'neural-7b-v3-1': '@hf/thebloke/neural-chat-7b-v3-1-awq',
        'openchat-3.5-0106': '@cf/openchat/openchat-3.5-0106',
        'openhermes-2.5-mistral-7b': '@hf/thebloke/openhermes-2.5-mistral-7b-awq',
        'phi-2': '@cf/microsoft/phi-2',
        'qwen-1.5-1.8b': '@cf/qwen/qwen1.5-1.8b-chat',
        'qwen-1.5-14b': '@cf/qwen/qwen1.5-14b-chat-awq',
        'qwen-1.5-7b': '@cf/qwen/qwen1.5-7b-chat-awq',
        'qwen-2.5-coder-32b': '@cf/qwen/qwen2.5-coder-32b-instruct',
        'qwen1.5-0.5b': '@cf/qwen/qwen1.5-0.5b-chat',
        'qwen3-30b-a3b': '@cf/qwen/qwen3-30b-a3b-fp8',
        'qwq-32b': '@cf/qwen/qwq-32b',
        'sqlcoder-7b-2': '@cf/defog/sqlcoder-7b-2',
        'starling-lm-7b-beta': '@hf/nexusflow/starling-lm-7b-beta',
        'tinyllama-1.1b-v1.0': '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
        'una-cybertron-7b-v2-bf16': '@cf/fblgit/una-cybertron-7b-v2-bf16',
        'zephyr-7b-beta': '@hf/thebloke/zephyr-7b-beta-awq'
    }
    models = list(model_aliases.keys())

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncResult:
        try:
            from ..requests.cdp import CDPSession
        except ImportError:
            raise RuntimeError("CDP module is required for Cloudflare provider. Please ensure g4f.requests.cdp is available.")

        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass

        debug.log("Cloudflare: Starting CDPSession...")
        session = CDPSession(headless=False)
        await session.start()
        
        try:
            await session.navigate(cls.url)
            
            # Wait for Cloudflare validation to pass and React to load
            await asyncio.sleep(5)
            
            for _ in range(30):
                title = await session.evaluate_js("document.title") or ""
                content = await session.evaluate_js("document.body.innerText") or ""
                if title and "Just a moment" not in title and "Attention Required" not in title and "cf-browser-verification" not in content:
                    break
                await asyncio.sleep(1)

            # Setup event queue for console logs
            q = asyncio.Queue()
            session.add_event_handler("Runtime.consoleAPICalled", q)

            # Render messages to the format expected by Cloudflare UI
            cf_messages = [{"role": msg["role"], "parts": [{"type": "text", "text": msg["content"]}], "id": f"msg_{i}"} for i, msg in enumerate(render_messages(messages))]
            
            # Inject JS to handle the WebSocket stream
            js_code = f"""
            async function runChat() {{
                const pk = crypto.randomUUID();
                const agentId = 'playground-' + Math.random().toString(36).substring(2, 15);
                const modelStr = {json.dumps(model)};
                
                const wsUrl = `wss://playground.ai.cloudflare.com/agents/playground/${{agentId}}?_pk=${{pk}}&model=${{encodeURIComponent(modelStr)}}`;
                const ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {{
                    ws.send(JSON.stringify({{"type":"cf_agent_stream_resume_request"}}));
                    ws.send(JSON.stringify({{"name":agentId,"agent":"playground","type":"cf_agent_identity"}}));
                    ws.send(JSON.stringify({{"state":{{"model":modelStr,"temperature":1,"stream":true,"system":"You are a helpful assistant.","useExternalProvider":false,"externalProvider":"openai","generativeUI":false}},"type":"cf_agent_state"}}));
                    ws.send(JSON.stringify({{"mcp":{{"prompts":[],"resources":[],"servers":{{}},"tools":[]}},"type":"cf_agent_mcp_servers"}}));
                    ws.send(JSON.stringify({{"type":"cf_agent_stream_resume_none"}}));
                    
                    setTimeout(() => {{
                        const req = {{
                            "id": "req_" + Math.random().toString(36).substring(2, 10),
                            "init": {{
                                "method": "POST",
                                "body": JSON.stringify({{
                                    "messages": {json.dumps(cf_messages)},
                                    "trigger": "submit-message"
                                }})
                            }},
                            "type": "cf_agent_use_chat_request"
                        }};
                        ws.send(JSON.stringify(req));
                    }}, 1000);
                }};
                
                ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.type === 'cf_agent_use_chat_response') {{
                            const body = JSON.parse(data.body);
                            if (body.type === 'text-delta') {{
                                console.log("CF_CHUNK: " + body.delta);
                            }} else if (body.type === 'finish-step' || body.type === 'finish') {{
                                console.log("CF_DONE");
                                ws.close();
                            }}
                        }}
                    }} catch (e) {{}}
                }};
                
                ws.onerror = () => console.log("CF_ERROR");
                ws.onclose = () => console.log("CF_DONE");
            }}
            runChat();
            """
            
            debug.log("Cloudflare: Injecting WebSocket streaming JS...")
            await session.evaluate_js(js_code)
            
            # Yield from the queue
            last_val = None
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    args = event.get("args", [])
                    if args and args[0].get("type") == "string":
                        val = args[0].get("value", "")
                        if val == last_val:
                            continue
                        last_val = val
                        if val.startswith("CF_CHUNK: "):
                            yield val[10:]
                        elif val == "CF_DONE":
                            break
                        elif val == "CF_ERROR":
                            raise RuntimeError("WebSocket error inside Cloudflare session")
                except asyncio.TimeoutError:
                    raise TimeoutError("Timeout waiting for Cloudflare response")
        finally:
            session.remove_event_handler("Runtime.consoleAPICalled", q)
            await session.close()
