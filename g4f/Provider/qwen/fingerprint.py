import random
import time
from typing import Dict, List, Any

# =========================
# DEFAULT TEMPLATE
# =========================

DEFAULT_TEMPLATE: Dict[str, Any] = {
    "deviceId": "84985177a19a010dea49",
    "sdkVersion": "websdk-2.3.15d",
    "initTimestamp": "1765348410850",
    "field3": "91",
    "field4": "1|15",
    "language": "zh-CN",
    "timezoneOffset": "-480",
    "colorDepth": "16705151|12791",
    "screenInfo": "1470|956|283|797|158|0|1470|956|1470|798|0|0",
    "field9": "5",
    "platform": "MacIntel",
    "field11": "10",
    "webglRenderer": (
        "ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version)"
        "|Google Inc. (Apple)"
    ),
    "field13": "30|30",
    "field14": "0",
    "field15": "28",
    "pluginCount": "5",
    "vendor": "Google Inc.",
    "field29": "8",
    "touchInfo": "-1|0|0|0|0",
    "field32": "11",
    "field35": "0",
    "mode": "P",
}

# =========================
# PRESETS
# =========================

SCREEN_PRESETS = {
    "1920x1080": "1920|1080|283|1080|158|0|1920|1080|1920|922|0|0",
    "2560x1440": "2560|1440|283|1440|158|0|2560|1440|2560|1282|0|0",
    "1470x956": "1470|956|283|797|158|0|1470|956|1470|798|0|0",
    "1440x900": "1440|900|283|900|158|0|1440|900|1440|742|0|0",
    "1536x864": "1536|864|283|864|158|0|1536|864|1536|706|0|0",
}

PLATFORM_PRESETS = {
    "macIntel": {
        "platform": "MacIntel",
        "webglRenderer": (
            "ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version)"
            "|Google Inc. (Apple)"
        ),
        "vendor": "Google Inc.",
    },
    "macM1": {
        "platform": "MacIntel",
        "webglRenderer": (
            "ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)"
            "|Google Inc. (Apple)"
        ),
        "vendor": "Google Inc.",
    },
    "win64": {
        "platform": "Win32",
        "webglRenderer": (
            "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 "
            "vs_5_0 ps_5_0, D3D11)|Google Inc. (NVIDIA)"
        ),
        "vendor": "Google Inc.",
    },
    "linux": {
        "platform": "Linux x86_64",
        "webglRenderer": (
            "ANGLE (Intel, Mesa Intel(R) UHD Graphics 630, OpenGL 4.6)"
            "|Google Inc. (Intel)"
        ),
        "vendor": "Google Inc.",
    },
}

LANGUAGE_PRESETS = {
    "zh-CN": {"language": "zh-CN", "timezoneOffset": "-480"},
    "zh-TW": {"language": "zh-TW", "timezoneOffset": "-480"},
    "en-US": {"language": "en-US", "timezoneOffset": "480"},
    "ja-JP": {"language": "ja-JP", "timezoneOffset": "-540"},
    "ko-KR": {"language": "ko-KR", "timezoneOffset": "-540"},
}

# =========================
# HELPERS
# =========================

def generate_device_id() -> str:
    """Generate a 20-character hex device ID"""
    return "".join(random.choice("0123456789abcdef") for _ in range(20))


def generate_hash() -> int:
    """Generate a 32-bit unsigned random hash"""
    return random.randint(0, 0xFFFFFFFF)


# =========================
# CORE LOGIC
# =========================

def generate_fingerprint(options: Dict[str, Any] = None) -> str:
    if options is None:
        options = {}

    config = DEFAULT_TEMPLATE.copy()

    # platform preset
    platform = options.get("platform")
    if platform in PLATFORM_PRESETS:
        config.update(PLATFORM_PRESETS[platform])

    # screen preset
    screen = options.get("screen")
    if screen in SCREEN_PRESETS:
        config["screenInfo"] = SCREEN_PRESETS[screen]

    # language preset
    locale = options.get("locale")
    if locale in LANGUAGE_PRESETS:
        config.update(LANGUAGE_PRESETS[locale])

    # custom overrides
    if "custom" in options and isinstance(options["custom"], dict):
        config.update(options["custom"])

    device_id = options.get("deviceId") or generate_device_id()
    current_timestamp = int(time.time() * 1000)

    plugin_hash = generate_hash()
    canvas_hash = generate_hash()
    ua_hash1 = generate_hash()
    ua_hash2 = generate_hash()
    url_hash = generate_hash()
    doc_hash = random.randint(10, 100)

    fields: List[Any] = [
        device_id,
        config["sdkVersion"],
        config["initTimestamp"],
        config["field3"],
        config["field4"],
        config["language"],
        config["timezoneOffset"],
        config["colorDepth"],
        config["screenInfo"],
        config["field9"],
        config["platform"],
        config["field11"],
        config["webglRenderer"],
        config["field13"],
        config["field14"],
        config["field15"],
        f'{config["pluginCount"]}|{plugin_hash}',
        canvas_hash,
        ua_hash1,
        "1",
        "0",
        "1",
        "0",
        config["mode"],
        "0",
        "0",
        "0",
        "416",
        config["vendor"],
        config["field29"],
        config["touchInfo"],
        ua_hash2,
        config["field32"],
        current_timestamp,
        url_hash,
        config["field35"],
        doc_hash,
    ]

    return "^".join(map(str, fields))


def generate_fingerprint_batch(count: int, options: Dict[str, Any] = None) -> List[str]:
    return [generate_fingerprint(options) for _ in range(count)]


def parse_fingerprint(fingerprint: str) -> Dict[str, Any]:
    fields = fingerprint.split("^")
    return {
        "deviceId": fields[0],
        "sdkVersion": fields[1],
        "initTimestamp": fields[2],
        "language": fields[5],
        "timezoneOffset": fields[6],
        "platform": fields[10],
        "webglRenderer": fields[12],
        "mode": fields[23],
        "vendor": fields[28],
        "timestamp": fields[33],
        "raw": fields,
    }
