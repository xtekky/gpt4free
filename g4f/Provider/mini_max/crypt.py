from __future__ import annotations

import asyncio
import hashlib
import json
from urllib.parse import quote

from ...providers.response import JsonMixin
from ...requests import Tab

API_PATH = "/v4/api/chat/msg"

class CallbackResults(JsonMixin):
    def __init__(self):
        self.token: str = None
        self.path_and_query: str = None
        self.timestamp: int = None

def hash_function(base_string: str) -> str:
    """
    Mimics the hashFunction using MD5.
    """
    return hashlib.md5(base_string.encode()).hexdigest()

def generate_yy_header(has_search_params_path: str, body_to_yy: dict, time: int) -> str:
    """
    Python equivalent of the generateYYHeader function.
    """
    # print("Encoded Path:", quote(has_search_params_path, ""))
    # print("Stringified Body:", s)
    # print("Hashed Time:", hash_function(str(time)))

    encoded_path = quote(has_search_params_path, "")
    time_hash = hash_function(str(time))
    combined_string = f"{encoded_path}_{body_to_yy}{time_hash}ooui"

    # print("Combined String:", combined_string)
    # print("Hashed Combined String:", hash_function(combined_string))
    return hash_function(combined_string)

def get_body_to_yy(l):
    L = l["msgContent"].replace("\r\n", "").replace("\n", "").replace("\r", "")
    M = hash_function(l["characterID"]) + hash_function(L) + hash_function(l["chatID"])
    M += hash_function("")  # Mimics hashFunction(undefined) in JS

    # print("bodyToYY:", M)
    return M

def get_body_json(s):
    return json.dumps(s, ensure_ascii=True, sort_keys=True)

async def get_browser_callback(auth_result: CallbackResults):
    async def callback(page: Tab):
        while not auth_result.token:
            auth_result.token = await page.evaluate("localStorage.getItem('_token')")
            if not auth_result.token:
                await asyncio.sleep(1)
        (auth_result.path_and_query, auth_result.timestamp) = await page.evaluate("""
            const device_id = localStorage.getItem("USER_HARD_WARE_INFO");
            const uuid = localStorage.getItem("UNIQUE_USER_ID");
            const os_name = navigator.userAgentData?.platform || navigator.platform || "Unknown";
            const browser_name = (() => {
                const userAgent = navigator.userAgent.toLowerCase();
                if (userAgent.includes("chrome") && !userAgent.includes("edg")) return "chrome";
                if (userAgent.includes("edg")) return "edge";
                if (userAgent.includes("firefox")) return "firefox";
                if (userAgent.includes("safari") && !userAgent.includes("chrome")) return "safari";
                return "unknown";
            })();
            const cpu_core_num = navigator.hardwareConcurrency || 8;
            const browser_language = navigator.language || "unknown";
            const browser_platform = `${navigator.platform || "unknown"}`;
            const screen_width = window.screen.width || "unknown";
            const screen_height = window.screen.height || "unknown";
            const unix = Date.now(); // Current Unix timestamp in milliseconds
            const params = {
                device_platform: "web",
                biz_id: 2,
                app_id: 3001,
                version_code: 22201,
                lang: "en",
                uuid,
                device_id,
                os_name,
                browser_name,
                cpu_core_num,
                browser_language,
                browser_platform,
                screen_width,
                screen_height,
                unix
            };
            [new URLSearchParams(params).toString(), unix]
        """)
        auth_result.path_and_query = f"{API_PATH}?{auth_result.path_and_query}"
    return callback