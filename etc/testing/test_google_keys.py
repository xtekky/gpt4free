#!/usr/bin/env python3
"""Test Google API keys against the Gemini API.

Usage:
  # Via environment variable (comma-separated keys)
  export GOOGLE_API_KEYS="key1,key2,key3"
  python etc/testing/test_google_keys.py

  # Via command line
  python etc/testing/test_google_keys.py key1 key2 key3

Keys are never written to disk by this script.
"""

import json
import os
import sys
import urllib.request
import urllib.error

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "?key={key}"
)

# Minimal generation request to verify the key is authorized
GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.8-flash:generateContent?key={key}"
)

PAYLOAD = json.dumps({
    "contents": [{"parts": [{"text": "Say 'ok' and nothing else."}]}]
}).encode()


def list_models(key: str) -> tuple[bool, str]:
    """Check if the credential can list models. Returns (ok, detail)."""
    url = GEMINI_ENDPOINT.format(key=key)
    headers = {}
    if key.startswith("AQ."):
        # OAuth-style token: use Bearer auth instead of key param
        url = GEMINI_ENDPOINT.format(key="")
        headers["Authorization"] = f"Bearer {key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as resp:
            data = json.loads(resp.read())
            count = len(data.get("models", []))
            return True, f"listed {count} models"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)


def generate(key: str) -> tuple[bool, str]:
    """Check if the credential can generate content. Returns (ok, detail)."""
    url = GENERATE_URL.format(key=key)
    headers = {"Content-Type": "application/json"}
    if key.startswith("AQ."):
        url = GENERATE_URL.format(key="")
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=PAYLOAD, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return True, f"response: {text.strip()[:60]!r}"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        return False, f"HTTP {e.code}: {body}"
    except Exception as e:
        return False, str(e)


def mask(key: str) -> str:
    """Mask key for safe display."""
    if len(key) <= 10:
        return key[:4] + "..."
    return key[:8] + "..." + key[-4:]


def main():
    keys = []

    if len(sys.argv) > 1:
        keys = sys.argv[1:]
    elif os.environ.get("GOOGLE_API_KEYS"):
        keys = [k.strip() for k in os.environ["GOOGLE_API_KEYS"].split(",") if k.strip()]

    if not keys:
        print("No keys provided.")
        print("Usage: python etc/testing/test_google_keys.py key1 key2 ...")
        print("   or: GOOGLE_API_KEYS=key1,key2 python etc/testing/test_google_keys.py")
        sys.exit(1)

    print(f"Testing {len(keys)} key(s)...\n")
    results = []

    for key in keys:
        ok_list, detail_list = list_models(key)
        ok_gen, detail_gen = ("-", "-")
        if ok_list:
            ok_gen, detail_gen = generate(key)

        status = "VALID" if (ok_list and ok_gen) else ("LIST-ONLY" if ok_list else "INVALID")
        results.append((mask(key), status))

        print(f"Key {mask(key)}")
        print(f"  list_models : {'PASS' if ok_list else 'FAIL'} — {detail_list}")
        if ok_list:
            print(f"  generate    : {'PASS' if ok_gen else 'FAIL'} — {detail_gen}")
        else:
            print(f"  generate    : SKIP — {detail_gen}")
        print()
    print("=" * 50)
    print("Summary:")
    for k, s in results:
        print(f"  {k:30s} {s}")


if __name__ == "__main__":
    main()
