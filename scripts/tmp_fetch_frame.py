import asyncio, re, sys
sys.path.insert(0, "/home/heiner/Projects/gpt4free")


async def main():
    from g4f.requests import StreamSession
    from g4f.Provider.needs_auth import ChatGPTLightweight as M

    cookies, _ = M.ChatGPTLightweight._load_auth_state()
    ua = M.ChatGPTLightweight.user_agent
    async with StreamSession(impersonate="chrome", timeout=60, cookies=cookies) as s:
        async with s.get(
            "https://chatgpt.com/backend-api/sentinel/frame.html",
            headers={"user-agent": ua, "accept": "text/html", "referer": "https://chatgpt.com/"},
        ) as r:
            html = await r.text()
    print("status ok, len:", len(html))
    with open("/tmp/frame.html", "w") as f:
        f.write(html)
    for m in re.findall(r'<script[^>]*src="([^"]+)"', html):
        print("script:", m)
    inline = re.findall(r"<script(?![^>]*src)[^>]*>(.*?)</script>", html, re.S)
    for i, t in enumerate(inline):
        print(f"inline[{i}]: {len(t)} chars")
        with open(f"/tmp/frame_inline_{i}.js", "w") as f:
            f.write(t)


asyncio.run(main())
