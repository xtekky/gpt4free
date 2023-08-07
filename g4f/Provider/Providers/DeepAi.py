import json
import os
import requests
import js2py
from ...typing import sha256, Dict, get_type_hints


url = "https://api.deepai.org/"
model = ['gpt-3.5-turbo']
supports_stream = True
needs_auth = False
working = True

token_js = """
var agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
var a, b, c, d, e, h, f, l, g, k, m, n, r, x, C, E, N, F, T, O, P, w, D, G, Q, R, W, I, aa, fa, na, oa, ha, ba, X, ia, ja, ka, J, la, K, L, ca, S, U, M, ma, B, da, V, Y;
h = Math.round(1E11 * Math.random()) + "";
f = function () {
    for (var p = [], q = 0; 64 > q;) p[q] = 0 | 4294967296 * Math.sin(++q % Math.PI);
    
    return function (t) {
        var v, y, H, ea = [v = 1732584193, y = 4023233417, ~v, ~y],
            Z = [],
            A = unescape(encodeURI(t)) + "\u0080",
            z = A.length;
        t = --z / 4 + 2 | 15;
        for (Z[--t] = 8 * z; ~z;) Z[z >> 2] |= A.charCodeAt(z) << 8 * z--;
        for (q = A = 0; q < t; q += 16) {
            for (z = ea; 64 > A; z = [H = z[3], v + ((H = z[0] + [v & y | ~v & H, H & v | ~H & y, v ^ y ^ H, y ^ (v | ~H)][z = A >> 4] + p[A] + ~~Z[q | [A, 5 * A + 1, 3 * A + 5, 7 * A][z] & 15]) << (z = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21][4 * z + A++ % 4]) | H >>> -z), v, y]) v = z[1] | 0, y = z[2];
            for (A = 4; A;) ea[--A] += z[A]
        }
        for (t = ""; 32 > A;) t += (ea[A >> 3] >> 4 * (1 ^ A++) & 15).toString(16);
        return t.split("").reverse().join("")
    }
}();

"tryit-" + h + "-" + f(agent + f(agent + f(agent + h + "x")));
"""

uuid4_js = """
function uuidv4() {
    for (var a = [], b = 0; 36 > b; b++) a[b] = "0123456789abcdef".substr(Math.floor(16 * Math.random()), 1);
    a[14] = "4";
    a[19] = "0123456789abcdef".substr(a[19] & 3 | 8, 1);
    a[8] = a[13] = a[18] = a[23] = "-";
    return a.join("")
}
uuidv4();"""

def create_session():
    url = "https://api.deepai.org/save_chat_session"

    payload = {'uuid': js2py.eval_js(uuid4_js), "title":"", "chat_style": "chat", "messages": '[]'}
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def _create_completion(model: str, messages:list, stream: bool = True, **kwargs):
    create_session()
    url = "https://api.deepai.org/make_me_a_pizza"

    payload = {'chas_style': "chat", "chatHistory": json.dumps(messages)}
    api_key = js2py.eval_js(token_js)
    headers = {"api-key": api_key, "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

    response = requests.request("POST", url, headers=headers, data=payload, stream=True)
    for chunk in response.iter_content(chunk_size=None):
        response.raise_for_status()
        yield chunk.decode()

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join(
        [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
