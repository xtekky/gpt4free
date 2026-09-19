"""
Faithful Python transcription of the ChatGPT sentinel turnstile VM.

Source: https://chatgpt.com/sentinel/<version>/sdk.js — loaded through the
https://chatgpt.com/backend-api/sentinel/frame.html iframe. The VM (`Et` in the
minified SDK) executes an opcode program that is base64-decoded and
XOR-ciphered with the requirements token (`p` sent to sentinel/req).

Key semantics (transcribed exactly from the SDK):

- The register map IS the dispatch table: executing `[code, *args]` looks up
  `regs[code]` and calls it with `args`. Handlers and data share one map, so
  programs rebind opcodes to randomized float slots (obfuscation) and can call
  handlers (xor, set, ...) through call ops.
- Op 2 stores raw values: floats later serve as XOR keys via str() coercion.
- Call ops 13/20/21/23 forward args RAW; ops 7/17 evaluate args via the map.
- The program register (9) is a queue; ops may replace it mid-run
  (self-modifying continuation) or swap it temporarily (op 22 / closures).

Opcode table (register : handler):
    0  re-enter: decode new dx blob with reg[16] key, replace program
    1  reg[n] = xor(str(reg[n]), str(reg[e]))
    2  reg[n] = raw value
    3  resolve(result)   -> final b64 token
    4  reject(reason)    -> abort
    5  append: array push / JS `+` concat
    6  reg[n] = reg[e][reg[r]]
    7  reg[n](*[reg[a] for a in args])   (evaluated)
    8  reg[n] = reg[e]
    9  program queue (data)
    10  window object (data)
    11  reg[n] = first document script src matching regex reg[e], else null
    12  reg[n] = the register map itself
    13  reg[n] = reg[e](*raw args) or str(error)
    14  reg[n] = JSON.parse(str(reg[e]))
    15  reg[n] = JSON.stringify(reg[e])
    16  requirements token (data; the XOR key)
    17  reg[n] = reg[e](*[reg[a] for a in args]) or str(error)  (evaluated)
    18  reg[n] = atob(str(reg[n]))
    19  reg[n] = btoa(str(reg[n]))
    20  if reg[n] === reg[e]: reg[r](*raw args)
    21  if |num(reg[n]) - num(reg[e])| > num(reg[r]): reg[o](*raw args)
    22  run inline sub-program (swap reg[9], drain, restore)
    23  if reg[n] is not undefined: reg[e](*raw args)
    24  reg[n] = reg[e][reg[r]] bound to reg[e]
    25  noop
    26  noop
    27  remove: array splice / numeric subtract
    28  noop
    29  reg[n] = reg[e] < reg[r]   (JS <)
    30  define async closure in reg[n]
    33  reg[n] = Number(reg[e]) * Number(reg[r])
    34  reg[n] = await reg[e]
    35  reg[n] = Number(reg[e]) / Number(reg[r])   (0 when dividing by 0)
"""

import base64
import json
import random
import re
import time

_JS_UNDEFINED = object()  # missing register (JS undefined)
_JS_NULL = object()       # explicit JS null


def _b64decode(s: str) -> bytes:
    return base64.b64decode(s + "=" * (-len(s) % 4))


def _xor_cipher(data: str, key: str) -> str:
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))


def _to_str(v) -> str:
    """JS ``"" + v`` coercion."""
    if v is None or v is _JS_UNDEFINED:
        return "undefined"
    if v is _JS_NULL:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def _to_num(v) -> float:
    """JS ``Number(v)`` coercion (NaN approximated as 0)."""
    if v is None or v is _JS_UNDEFINED or v is _JS_NULL or isinstance(v, (dict, list)):
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _js_strict_eq(a, b) -> bool:
    """JS ``===``."""
    if a is None or a is _JS_UNDEFINED:
        return b is None or b is _JS_UNDEFINED
    if b is None or b is _JS_UNDEFINED:
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) != isinstance(b, str):
        return False
    return a == b


def _js_add(a, b):
    """JS ``+``."""
    a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_num and b_num:
        return a + b
    return _to_str(a) + _to_str(b)


def _js_lt(a, b) -> bool:
    """JS ``<``."""
    if isinstance(a, str) and isinstance(b, str):
        return a < b
    try:
        if isinstance(a, str):
            return float(a) < _to_num(b)
        if isinstance(b, str):
            return _to_num(a) < float(b)
    except ValueError:
        return False
    return _to_num(a) < _to_num(b)


class TurnstileVM:
    def __init__(self, program: list, requirements_token: str, user_agent: str = "",
                 script_srcs: list | None = None):
        self.regs: dict = {}
        self.resolved = None
        self.rejected = None
        self.op_count = 0
        self.start = time.time()
        self.script_srcs = script_srcs or []
        self._setup(program, requirements_token, user_agent)

    # ------------------------------------------------------------------ setup

    def _setup(self, program: list, requirements_token: str, user_agent: str) -> None:
        r = self.regs
        r[0] = self._op0_reenter
        r[1] = self._op1_xor
        r[2] = self._op2_set
        r[3] = self._op3_resolve
        r[4] = self._op4_reject
        r[5] = self._op5_append
        r[6] = self._op6_index
        r[7] = self._op7_call
        r[8] = self._op8_copy
        r[10] = self._make_window(user_agent)
        r[11] = self._op11_find_script
        r[12] = self._op12_expose_map
        r[13] = self._op13_call_catch
        r[14] = self._op14_json_parse
        r[15] = self._op15_json_stringify
        r[16] = requirements_token
        r[17] = self._op17_async_call
        r[18] = self._op18_b64decode
        r[19] = self._op19_b64encode
        r[20] = self._op20_cond_call
        r[21] = self._op21_cond_delta
        r[22] = self._op22_subprogram
        r[23] = self._op23_call_if_defined
        r[24] = self._op24_bind
        r[25] = self._noop
        r[26] = self._noop
        r[27] = self._op27_remove
        r[28] = self._noop
        r[29] = self._op29_less_than
        r[30] = self._op30_closure
        r[33] = self._op33_multiply
        r[34] = self._op34_await
        r[35] = self._op35_divide
        r[9] = list(program)

    def _make_window(self, user_agent: str) -> dict:
        vm = self

        class Performance:
            @staticmethod
            def now() -> float:
                return (time.time() - vm.start) * 1000.0 + random.random()

        return {
            "document": {"location": "https://chatgpt.com/", "visibilityState": "visible"},
            "navigator": {
                "userAgent": user_agent,
                "language": "en-US",
                "languages": ["en-US"],
                "hardwareConcurrency": 8,
            },
            "localStorage": {
                "STATSIG_LOCAL_STORAGE_INTERNAL_STORE_V4": "{}",
                "STATSIG_LOCAL_STORAGE_STABLE_ID": "",
                "client-correlated-secret": "",
                "oai/apps/capExpiresAt": "",
                "oai-did": "",
                "STATSIG_LOCAL_STORAGE_LOGGING_REQUEST": "",
                "UiState.isNavigationCollapsed.1": "",
            },
            "screen": {"width": 412, "height": 915, "availWidth": 412, "availHeight": 915},
            "history": {"length": 2},
            "performance": Performance,
            "Math": {"random": lambda: random.random()},
            "Object": {"create": lambda _proto=None: {}},
            "Reflect": {"set": lambda obj, k, v: obj.__setitem__(k, v) if isinstance(obj, dict) else None},
        }

    # --------------------------------------------------------------- helpers

    def _get(self, k):
        return self.regs.get(k, _JS_UNDEFINED)

    def _set(self, k, v) -> None:
        self.regs[k] = v

    def _call_handler(self, handler, args):
        if callable(handler):
            return handler(*args)
        return None

    def _noop(self, *args) -> None:
        pass

    # --------------------------------------------------------------- handlers

    def _op0_reenter(self, dx) -> None:
        """Decode a new challenge blob with the current reg[16] key and
        replace the program queue (nested continuation)."""
        raw = _b64decode(_to_str(dx))
        key = _to_str(self._get(16))
        program = json.loads(_xor_cipher(raw.decode("utf-8", "replace"), key))
        if isinstance(program, list):
            self.regs[9] = program

    def _op1_xor(self, n, e) -> None:
        key = _to_str(self._get(e))
        self._set(n, _xor_cipher(_to_str(self._get(n)), key))

    def _op2_set(self, n, v) -> None:
        self._set(n, v)

    def _op3_resolve(self, t) -> None:
        if self.resolved is None and self.rejected is None:
            self.resolved = _to_str(t)

    def _op4_reject(self, t) -> None:
        if self.resolved is None and self.rejected is None:
            self.rejected = _to_str(t)

    def _op5_append(self, n, e) -> None:
        cur, val = self._get(n), self._get(e)
        if isinstance(cur, list):
            cur.append(val)
        else:
            self._set(n, _js_add(cur, val))

    def _op6_index(self, n, e, r) -> None:
        obj, key = self._get(e), self._get(r)
        if isinstance(obj, dict):
            self._set(n, obj.get(_to_str(key), _JS_UNDEFINED))
        elif isinstance(obj, (list, str)):
            try:
                self._set(n, obj[int(_to_num(key))])
            except (IndexError, TypeError, ValueError):
                self._set(n, _JS_UNDEFINED)
        else:
            self._set(n, _JS_UNDEFINED)

    def _op7_call(self, n, *args) -> None:
        fn = self._get(n)
        self._call_handler(fn, [self._get(a) for a in args])

    def _op8_copy(self, n, e) -> None:
        self._set(n, self._get(e))

    def _op11_find_script(self, n, e) -> None:
        pattern = self._get(e)
        if isinstance(pattern, str):
            for src in self.script_srcs:
                m = re.search(pattern, src)
                if m:
                    self._set(n, m.group(0))
                    return
        self._set(n, _JS_NULL)

    def _op12_expose_map(self, n) -> None:
        self._set(n, self.regs)

    def _op13_call_catch(self, n, e, *raw_args) -> None:
        fn = self._get(e)
        try:
            self._set(n, self._call_handler(fn, list(raw_args)))
        except Exception as exc:  # noqa: BLE001
            self._set(n, str(exc))

    def _op14_json_parse(self, n, e) -> None:
        try:
            self._set(n, json.loads(_to_str(self._get(e))))
        except (json.JSONDecodeError, ValueError):
            self._set(n, _JS_UNDEFINED)

    def _op15_json_stringify(self, n, e) -> None:
        self._set(n, json.dumps(self._get(e), default=_to_str))

    def _op17_async_call(self, n, e, *args) -> None:
        fn = self._get(e)
        try:
            self._set(n, self._call_handler(fn, [self._get(a) for a in args]))
        except Exception as exc:  # noqa: BLE001
            self._set(n, str(exc))

    def _op18_b64decode(self, n) -> None:
        try:
            self._set(n, _b64decode(_to_str(self._get(n))).decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            self._set(n, "")

    def _op19_b64encode(self, n) -> None:
        self._set(n, base64.b64encode(_to_str(self._get(n)).encode()).decode())

    def _op20_cond_call(self, n, e, r, *raw_args) -> None:
        if _js_strict_eq(self._get(n), self._get(e)):
            self._call_handler(self._get(r), list(raw_args))

    def _op21_cond_delta(self, n, e, r, o, *raw_args) -> None:
        if abs(_to_num(self._get(n)) - _to_num(self._get(e))) > _to_num(self._get(r)):
            self._call_handler(self._get(o), list(raw_args))

    def _op22_subprogram(self, n, sub) -> None:
        saved = self.regs.get(9)
        self.regs[9] = list(sub) if isinstance(sub, list) else []
        self.run()
        self._set(n, "undefined")
        self.regs[9] = saved if isinstance(saved, list) else []

    def _op23_call_if_defined(self, n, e, *raw_args) -> None:
        if self._get(n) is not _JS_UNDEFINED:
            self._call_handler(self._get(e), list(raw_args))

    def _op24_bind(self, n, e, r) -> None:
        obj, key = self._get(e), self._get(r)
        if isinstance(obj, dict):
            self._set(n, obj.get(_to_str(key), _JS_UNDEFINED))
        else:
            self._set(n, f"{_to_str(obj)}.{_to_str(key)}")

    def _op27_remove(self, n, e) -> None:
        cur, val = self._get(n), self._get(e)
        if isinstance(cur, list):
            try:
                cur.remove(val)
            except ValueError:
                pass
        else:
            self._set(n, _to_num(cur) - _to_num(val))

    def _op29_less_than(self, n, e, r) -> None:
        self._set(n, _js_lt(self._get(e), self._get(r)))

    def _op30_closure(self, n, e, r, *rest) -> None:
        """Define a closure: reg[n] = function(*call_args) that binds args to
        slots, swaps the program for its body, drains it, and yields reg[e]."""
        has_bind = isinstance(rest[0], list) if rest else False
        bind_slots = rest[0] if has_bind else []
        body = (rest[1:] if has_bind else rest) or []

        def closure(*call_args):
            saved = self.regs.get(9)
            for slot, arg in zip(bind_slots, call_args):
                self._set(slot, arg)
            self.regs[9] = list(body)
            self.run()
            result = self._get(e)
            self.regs[9] = saved if isinstance(saved, list) else []
            return result

        self._set(n, closure)

    def _op33_multiply(self, n, e, r) -> None:
        self._set(n, _to_num(self._get(e)) * _to_num(self._get(r)))

    def _op34_await(self, n, e) -> None:
        self._set(n, self._get(e))

    def _op35_divide(self, n, e, r) -> None:
        divisor = _to_num(self._get(r))
        self._set(n, 0.0 if divisor == 0 else _to_num(self._get(e)) / divisor)

    # ------------------------------------------------------------- main loop

    def run(self) -> None:
        """Drain the program queue (reg[9]). Ops may replace the queue
        mid-run (self-modifying continuation)."""
        while self.resolved is None and self.rejected is None:
            queue = self.regs.get(9)
            if not isinstance(queue, list) or not queue:
                break
            op = queue.pop(0)
            if not isinstance(op, list) or not op:
                continue
            code, args = op[0], op[1:]
            self.op_count += 1
            handler = self.regs.get(code, _JS_UNDEFINED)
            if not callable(handler):
                continue
            try:
                handler(*args)
            except Exception:  # noqa: BLE001
                # JS: an op throw rejects the whole VM
                self.rejected = self.rejected or f"op {code} failed"
                return


def process_turnstile_new(dx: str, requirements_token: str, user_agent: str = "",
                          script_srcs: list | None = None) -> str:
    """Solve a sentinel turnstile challenge.

    Args:
        dx: base64 challenge blob from sentinel/req `turnstile.dx`.
        requirements_token: the `p` value sent with sentinel/req (XOR key).
        user_agent: UA string for navigator probes.
        script_srcs: document script srcs for op 11 regex probes.

    Returns:
        The turnstile token (b64), or "" on failure.
    """
    try:
        raw = _b64decode(dx)
        xored = _xor_cipher(raw.decode("utf-8", "replace"), requirements_token)
        program = json.loads(xored)
        if not isinstance(program, list):
            return ""
        vm = TurnstileVM(program, requirements_token, user_agent, script_srcs)
        vm.run()
        if vm.resolved:
            return base64.b64encode(vm.resolved.encode()).decode()
        return ""
    except Exception:  # noqa: BLE001
        return ""
