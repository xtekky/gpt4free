import json, js2py, requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class DeepAi(BaseProvider):
    url: str              = "https://deepai.org"
    working               = True
    supports_stream       = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
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

        payload = {"chas_style": "chat", "chatHistory": json.dumps(messages)}
        api_key = js2py.eval_js(token_js)
        headers = {
            "api-key": api_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        }

        response = requests.post("https://api.deepai.org/make_me_a_pizza", 
                                 headers=headers, data=payload, stream=True)
        
        for chunk in response.iter_content(chunk_size=None):
            response.raise_for_status()
            yield chunk.decode()
