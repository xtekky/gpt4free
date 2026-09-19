import asyncio, sys
sys.path.insert(0, "/home/heiner/Projects/gpt4free")


async def main():
    from g4f.requests import StreamSession
    from g4f.Provider.needs_auth import ChatGPTLightweight as M

    cookies, _ = M.ChatGPTLightweight._load_auth_state()
    ua = M.ChatGPTLightweight.user_agent
    async with StreamSession(impersonate="chrome", timeout=60, cookies=cookies) as s:
        async with s.get(
            "https://chatgpt.com/sentinel/20260810913b/sdk.js",
            headers={"user-agent": ua, "referer": "https://chatgpt.com/backend-api/sentinel/frame.html"},
        ) as r:
            js = await r.text()
    print("len:", len(js))
    with open("/tmp/sentinel_sdk.js", "w") as f:
        f.write(js)


asyncio.run(main())
