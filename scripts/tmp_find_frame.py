import json, re

har = json.load(open("har_and_cookies/chatgpt.com.har"))
for e in har["log"]["entries"]:
    u = e["request"]["url"]
    if "frame" in u.lower() or ("sentinel" in u and u.endswith(".js")):
        print(u[:160])

for e in har["log"]["entries"]:
    if "frame.html" in e["request"]["url"]:
        c = e.get("response", {}).get("content", {})
        txt = c.get("text", "") or ""
        print("\nframe.html size:", c.get("size"), "has text:", len(txt))
        for m in re.findall(r'src="([^"]+)"', txt)[:10]:
            print("  script:", m)
        break
