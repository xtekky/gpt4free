import hashlib
import base64
import random
import json
import time
import uuid

from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List

from datetime import (
    datetime, 
    timedelta, 
    timezone
)

from .har_file import RequestConfig

cores       = [16, 24, 32]
screens     = [3000, 4000, 6000]
maxAttempts = 500000

navigator_keys = [
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "appVersion−5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "share−function share() { [native code] }",
    "webdriver−false",
    "managed−[object NavigatorManagedData]",
    "canShare−function canShare() { [native code] }",
    "vendor−Google Inc.",
    "vendor−Google Inc.",
    "mediaDevices−[object MediaDevices]",
    "vibrate−function vibrate() { [native code] }",
    "storageBuckets−[object StorageBucketManager]",
    "mediaCapabilities−[object MediaCapabilities]",
    "getGamepads−function getGamepads() { [native code] }",
    "bluetooth−[object Bluetooth]",
    "share−function share() { [native code] }",
    "cookieEnabled−true",
    "virtualKeyboard−[object VirtualKeyboard]",
    "product−Gecko",
    "mediaDevices−[object MediaDevices]",
    "canShare−function canShare() { [native code] }",
    "getGamepads−function getGamepads() { [native code] }",
    "product−Gecko",
    "xr−[object XRSystem]",
    "clipboard−[object Clipboard]",
    "storageBuckets−[object StorageBucketManager]",
    "unregisterProtocolHandler−function unregisterProtocolHandler() { [native code] }",
    "productSub−20030107",
    "login−[object NavigatorLogin]",
    "vendorSub−",
    "login−[object NavigatorLogin]",
    "userAgent−Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "userAgent−Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "mediaDevices−[object MediaDevices]",
    "locks−[object LockManager]",
    "webkitGetUserMedia−function webkitGetUserMedia() { [native code] }",
    "vendor−Google Inc.",
    "xr−[object XRSystem]",
    "mediaDevices−[object MediaDevices]",
    "virtualKeyboard−[object VirtualKeyboard]",
    "userAgent−Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "virtualKeyboard−[object VirtualKeyboard]",
    "appName−Netscape",
    "storageBuckets−[object StorageBucketManager]",
    "presentation−[object Presentation]",
    "onLine−true",
    "mimeTypes−[object MimeTypeArray]",
    "credentials−[object CredentialsContainer]",
    "presentation−[object Presentation]",
    "getGamepads−function getGamepads() { [native code] }",
    "vendorSub−",
    "virtualKeyboard−[object VirtualKeyboard]",
    "serviceWorker−[object ServiceWorkerContainer]",
    "xr−[object XRSystem]",
    "product−Gecko",
    "keyboard−[object Keyboard]",
    "gpu−[object GPU]",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "doNotTrack",
    "clearAppBadge−function clearAppBadge() { [native code] }",
    "presentation−[object Presentation]",
    "serial−[object Serial]",
    "locks−[object LockManager]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "locks−[object LockManager]",
    "requestMediaKeySystemAccess−function requestMediaKeySystemAccess() { [native code] }",
    "vendor−Google Inc.",
    "pdfViewerEnabled−true",
    "language−zh-CN",
    "setAppBadge−function setAppBadge() { [native code] }",
    "geolocation−[object Geolocation]",
    "userAgentData−[object NavigatorUAData]",
    "mediaCapabilities−[object MediaCapabilities]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "getUserMedia−function getUserMedia() { [native code] }",
    "mediaDevices−[object MediaDevices]",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "userAgent−Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "sendBeacon−function sendBeacon() { [native code] }",
    "hardwareConcurrency−32",
    "appVersion−5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "credentials−[object CredentialsContainer]",
    "storage−[object StorageManager]",
    "cookieEnabled−true",
    "pdfViewerEnabled−true",
    "windowControlsOverlay−[object WindowControlsOverlay]",
    "scheduling−[object Scheduling]",
    "pdfViewerEnabled−true",
    "hardwareConcurrency−32",
    "xr−[object XRSystem]",
    "userAgent−Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "webdriver−false",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "bluetooth−[object Bluetooth]"
]

window_keys   = [
    "0",
    "window",
    "self",
    "document",
    "name",
    "location",
    "customElements",
    "history",
    "navigation",
    "locationbar",
    "menubar",
    "personalbar",
    "scrollbars",
    "statusbar",
    "toolbar",
    "status",
    "closed",
    "frames",
    "length",
    "top",
    "opener",
    "parent",
    "frameElement",
    "navigator",
    "origin",
    "external",
    "screen",
    "innerWidth",
    "innerHeight",
    "scrollX",
    "pageXOffset",
    "scrollY",
    "pageYOffset",
    "visualViewport",
    "screenX",
    "screenY",
    "outerWidth",
    "outerHeight",
    "devicePixelRatio",
    "clientInformation",
    "screenLeft",
    "screenTop",
    "styleMedia",
    "onsearch",
    "isSecureContext",
    "trustedTypes",
    "performance",
    "onappinstalled",
    "onbeforeinstallprompt",
    "crypto",
    "indexedDB",
    "sessionStorage",
    "localStorage",
    "onbeforexrselect",
    "onabort",
    "onbeforeinput",
    "onbeforematch",
    "onbeforetoggle",
    "onblur",
    "oncancel",
    "oncanplay",
    "oncanplaythrough",
    "onchange",
    "onclick",
    "onclose",
    "oncontentvisibilityautostatechange",
    "oncontextlost",
    "oncontextmenu",
    "oncontextrestored",
    "oncuechange",
    "ondblclick",
    "ondrag",
    "ondragend",
    "ondragenter",
    "ondragleave",
    "ondragover",
    "ondragstart",
    "ondrop",
    "ondurationchange",
    "onemptied",
    "onended",
    "onerror",
    "onfocus",
    "onformdata",
    "oninput",
    "oninvalid",
    "onkeydown",
    "onkeypress",
    "onkeyup",
    "onload",
    "onloadeddata",
    "onloadedmetadata",
    "onloadstart",
    "onmousedown",
    "onmouseenter",
    "onmouseleave",
    "onmousemove",
    "onmouseout",
    "onmouseover",
    "onmouseup",
    "onmousewheel",
    "onpause",
    "onplay",
    "onplaying",
    "onprogress",
    "onratechange",
    "onreset",
    "onresize",
    "onscroll",
    "onsecuritypolicyviolation",
    "onseeked",
    "onseeking",
    "onselect",
    "onslotchange",
    "onstalled",
    "onsubmit",
    "onsuspend",
    "ontimeupdate",
    "ontoggle",
    "onvolumechange",
    "onwaiting",
    "onwebkitanimationend",
    "onwebkitanimationiteration",
    "onwebkitanimationstart",
    "onwebkittransitionend",
    "onwheel",
    "onauxclick",
    "ongotpointercapture",
    "onlostpointercapture",
    "onpointerdown",
    "onpointermove",
    "onpointerrawupdate",
    "onpointerup",
    "onpointercancel",
    "onpointerover",
    "onpointerout",
    "onpointerenter",
    "onpointerleave",
    "onselectstart",
    "onselectionchange",
    "onanimationend",
    "onanimationiteration",
    "onanimationstart",
    "ontransitionrun",
    "ontransitionstart",
    "ontransitionend",
    "ontransitioncancel",
    "onafterprint",
    "onbeforeprint",
    "onbeforeunload",
    "onhashchange",
    "onlanguagechange",
    "onmessage",
    "onmessageerror",
    "onoffline",
    "ononline",
    "onpagehide",
    "onpageshow",
    "onpopstate",
    "onrejectionhandled",
    "onstorage",
    "onunhandledrejection",
    "onunload",
    "crossOriginIsolated",
    "scheduler",
    "alert",
    "atob",
    "blur",
    "btoa",
    "cancelAnimationFrame",
    "cancelIdleCallback",
    "captureEvents",
    "clearInterval",
    "clearTimeout",
    "close",
    "confirm",
    "createImageBitmap",
    "fetch",
    "find",
    "focus",
    "getComputedStyle",
    "getSelection",
    "matchMedia",
    "moveBy",
    "moveTo",
    "open",
    "postMessage",
    "print",
    "prompt",
    "queueMicrotask",
    "releaseEvents",
    "reportError",
    "requestAnimationFrame",
    "requestIdleCallback",
    "resizeBy",
    "resizeTo",
    "scroll",
    "scrollBy",
    "scrollTo",
    "setInterval",
    "setTimeout",
    "stop",
    "structuredClone",
    "webkitCancelAnimationFrame",
    "webkitRequestAnimationFrame",
    "chrome",
    "g_opr",
    "opr",
    "ethereum",
    "caches",
    "cookieStore",
    "ondevicemotion",
    "ondeviceorientation",
    "ondeviceorientationabsolute",
    "launchQueue",
    "documentPictureInPicture",
    "getScreenDetails",
    "queryLocalFonts",
    "showDirectoryPicker",
    "showOpenFilePicker",
    "showSaveFilePicker",
    "originAgentCluster",
    "credentialless",
    "speechSynthesis",
    "onscrollend",
    "webkitRequestFileSystem",
    "webkitResolveLocalFileSystemURL",
    "__remixContext",
    "__oai_SSR_TTI",
    "__remixManifest",
    "__reactRouterVersion",
    "DD_RUM",
    "__REACT_INTL_CONTEXT__",
    "filterCSS",
    "filterXSS",
    "__SEGMENT_INSPECTOR__",
    "DD_LOGS",
    "regeneratorRuntime",
    "_g",
    "__remixRouteModules",
    "__remixRouter",
    "__STATSIG_SDK__",
    "__STATSIG_JS_SDK__",
    "__STATSIG_RERENDER_OVERRIDE__",
    "_oaiHandleSessionExpired"
]

def get_parse_time():
    now = datetime.now(timezone(timedelta(hours=-5)))
    return now.strftime("%a %b %d %Y %H:%M:%S") + " GMT+0200 (Central European Summer Time)"

def get_config(user_agent):

    core   = random.choice(cores)
    screen = random.choice(screens)
    
    # partially hardcoded config
    config = [
        core + screen,
        get_parse_time(),
        None,
        random.random(),
        user_agent,
        None,
        RequestConfig.data_build, #document.documentElement.getAttribute("data-build"),
        "en-US",
        "en-US,es-US,en,es",
        0,
        random.choice(navigator_keys),
        'location',
        random.choice(window_keys),
        time.perf_counter(),
        str(uuid.uuid4()),
        "",
        8,
        int(time.time()),
    ]
    
    return config


def get_answer_token(seed, diff, config):
    answer, solved = generate_answer(seed, diff, config)

    if solved:
        return "gAAAAAB" + answer
    else:
        raise Exception("Failed to solve 'gAAAAAB' challenge")

def generate_answer(seed, diff, config):
    diff_len            = len(diff)
    seed_encoded        = seed.encode()
    p1 = (json.dumps(config[:3], separators=(',', ':'), ensure_ascii=False)[:-1] + ',').encode()
    p2 = (',' + json.dumps(config[4:9], separators=(',', ':'), ensure_ascii=False)[1:-1] + ',').encode()
    p3 = (',' + json.dumps(config[10:], separators=(',', ':'), ensure_ascii=False)[1:]).encode()

    target_diff = bytes.fromhex(diff)

    for i in range(maxAttempts):
        d1   = str(i).encode()
        d2   = str(i >> 1).encode()
        
        string = (
            p1 
            + d1 
            + p2 
            + d2 
            + p3
        )
        
        base_encode = base64.b64encode(string)
        hash_value  = hashlib.new("sha3_512", seed_encoded + base_encode).digest()
        
        if hash_value[:diff_len] <= target_diff:
            return base_encode.decode(), True

    return 'wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D' + base64.b64encode(f'"{seed}"'.encode()).decode(), False

def get_requirements_token(config):
    require, solved = generate_answer(format(random.random()), "0fffff", config)

    if solved:
        return 'gAAAAAC' + require
    else:
        raise Exception("Failed to solve 'gAAAAAC' challenge")
    
    
### processing turnstile token

class OrderedMap:
    def __init__(self):
        self.map = OrderedDict()

    def add(self, key: str, value: Any):
        self.map[key] = value

    def to_json(self):
        return json.dumps(self.map)

    def __str__(self):
        return self.to_json()


TurnTokenList = List[List[Any]]
FloatMap      = Dict[float, Any]
StringMap     = Dict[str, Any]
FuncType      = Callable[..., Any]

start_time = time.time()

def get_turnstile_token(dx: str, p: str) -> str:
    decoded_bytes = base64.b64decode(dx)
    return process_turnstile_token(decoded_bytes.decode(), p)


def process_turnstile_token(dx: str, p: str) -> str:
    result = []
    p_length = len(p)
    if p_length != 0:
        for i, r in enumerate(dx):
            result.append(chr(ord(r) ^ ord(p[i % p_length])))
    else:
        result = list(dx)
    return "".join(result)


def is_slice(input_val: Any) -> bool:
    return isinstance(input_val, (list, tuple))


def is_float(input_val: Any) -> bool:
    return isinstance(input_val, float)


def is_string(input_val: Any) -> bool:
    return isinstance(input_val, str)


def to_str(input_val: Any) -> str:
    if input_val is None:
        return "undefined"
    elif is_float(input_val):
        return f"{input_val:.16g}"
    elif is_string(input_val):
        special_cases = {
            "window.Math": "[object Math]",
            "window.Reflect": "[object Reflect]",
            "window.performance": "[object Performance]",
            "window.localStorage": "[object Storage]",
            "window.Object": "function Object() { [native code] }",
            "window.Reflect.set": "function set() { [native code] }",
            "window.performance.now": "function () { [native code] }",
            "window.Object.create": "function create() { [native code] }",
            "window.Object.keys": "function keys() { [native code] }",
            "window.Math.random": "function random() { [native code] }",
        }
        return special_cases.get(input_val, input_val)
    elif isinstance(input_val, list) and all(
        isinstance(item, str) for item in input_val
    ):
        return ",".join(input_val)
    else:
        # print(f"Type of input is: {type(input_val)}")
        return str(input_val)


def get_func_map() -> FloatMap:
    process_map: FloatMap = defaultdict(lambda: None)

    def func_1(e: float, t: float):
        e_str = to_str(process_map[e])
        t_str = to_str(process_map[t])
        if e_str is not None and t_str is not None:
            res = process_turnstile_token(e_str, t_str)
            process_map[e] = res
        else:
            pass
            # print(f"Warning: Unable to process func_1 for e={e}, t={t}")

    def func_2(e: float, t: Any):
        process_map[e] = t

    def func_5(e: float, t: float):
        n = process_map[e]
        tres = process_map[t]
        if n is None:
            process_map[e] = tres
        elif is_slice(n):
            nt = n + [tres] if tres is not None else n
            process_map[e] = nt
        else:
            if is_string(n) or is_string(tres):
                res = to_str(n) + to_str(tres)
            elif is_float(n) and is_float(tres):
                res = n + tres
            else:
                res = "NaN"
            process_map[e] = res

    def func_6(e: float, t: float, n: float):
        tv = process_map[t]
        nv = process_map[n]
        if is_string(tv) and is_string(nv):
            res = f"{tv}.{nv}"
            if res == "window.document.location":
                process_map[e] = "https://chatgpt.com/"
            else:
                process_map[e] = res
        else:
            pass
            # print("func type 6 error")

    def func_24(e: float, t: float, n: float):
        tv = process_map[t]
        nv = process_map[n]
        if is_string(tv) and is_string(nv):
            process_map[e] = f"{tv}.{nv}"
        else:
            pass
            # print("func type 24 error")

    def func_7(e: float, *args):
        n = [process_map[arg] for arg in args]
        ev = process_map[e]
        if isinstance(ev, str):
            if ev == "window.Reflect.set":
                obj = n[0]
                key_str = str(n[1])
                val = n[2]
                obj.add(key_str, val)
        elif callable(ev):
            ev(*n)

    def func_17(e: float, t: float, *args):
        i = [process_map[arg] for arg in args]
        tv = process_map[t]
        res = None
        if isinstance(tv, str):
            if tv == "window.performance.now":
                current_time = time.time_ns()
                elapsed_ns = current_time - int(start_time * 1e9)
                res = (elapsed_ns + random.random()) / 1e6
            elif tv == "window.Object.create":
                res = OrderedMap()
            elif tv == "window.Object.keys":
                if isinstance(i[0], str) and i[0] == "window.localStorage":
                    res = [
                        "STATSIG_LOCAL_STORAGE_INTERNAL_STORE_V4",
                        "STATSIG_LOCAL_STORAGE_STABLE_ID",
                        "client-correlated-secret",
                        "oai/apps/capExpiresAt",
                        "oai-did",
                        "STATSIG_LOCAL_STORAGE_LOGGING_REQUEST",
                        "UiState.isNavigationCollapsed.1",
                    ]
            elif tv == "window.Math.random":
                res = random.random()
        elif callable(tv):
            res = tv(*i)
        process_map[e] = res

    def func_8(e: float, t: float):
        process_map[e] = process_map[t]

    def func_14(e: float, t: float):
        tv = process_map[t]
        if is_string(tv):
            try:
                token_list = json.loads(tv)
                process_map[e] = token_list
            except json.JSONDecodeError:
                # print(f"Warning: Unable to parse JSON for key {t}")
                process_map[e] = None
        else:
            # print(f"Warning: Value for key {t} is not a string")
            process_map[e] = None

    def func_15(e: float, t: float):
        tv = process_map[t]
        process_map[e] = json.dumps(tv)

    def func_18(e: float):
        ev = process_map[e]
        e_str = to_str(ev)
        decoded = base64.b64decode(e_str).decode()
        process_map[e] = decoded

    def func_19(e: float):
        ev = process_map[e]
        e_str = to_str(ev)
        encoded = base64.b64encode(e_str.encode()).decode()
        process_map[e] = encoded

    def func_20(e: float, t: float, n: float, *args):
        o = [process_map[arg] for arg in args]
        ev = process_map[e]
        tv = process_map[t]
        if ev == tv:
            nv = process_map[n]
            if callable(nv):
                nv(*o)
            else:
                pass
                # print("func type 20 error")

    def func_21(*args):
        pass

    def func_23(e: float, t: float, *args):
        i = list(args)
        ev = process_map[e]
        tv = process_map[t]
        if ev is not None and callable(tv):
            tv(*i)

    process_map.update(
        {
            1: func_1,
            2: func_2,
            5: func_5,
            6: func_6,
            7: func_7,
            8: func_8,
            10: "window",
            14: func_14,
            15: func_15,
            17: func_17,
            18: func_18,
            19: func_19,
            20: func_20,
            21: func_21,
            23: func_23,
            24: func_24,
        }
    )

    return process_map


def process_turnstile(dx: str, p: str) -> str:
    tokens = get_turnstile_token(dx, p)
    res = ""
    token_list = json.loads(tokens)
    process_map = get_func_map()

    def func_3(e: str):
        nonlocal res
        res = base64.b64encode(e.encode()).decode()

    process_map[3]  = func_3
    process_map[9]  = token_list
    process_map[16] = p

    for token in token_list:
        try:
            e = token[0]
            t = token[1:]
            f = process_map.get(e)
            if callable(f):
                f(*t)
            else:
                pass
                # print(f"Warning: No function found for key {e}")
        except Exception as exc:
            raise Exception(f"Error processing token {token}: {exc}")
            # print(f"Error processing token {token}: {exc}")

    return res