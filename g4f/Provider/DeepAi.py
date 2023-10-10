from __future__ import annotations

import json
import js2py
import random
import hashlib
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class DeepAi(AsyncGeneratorProvider):
    url                 = "https://deepai.org"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        
        token_js = """
var agent = '""" + agent + """'
var a, b, c, d, e, h, f, l, g, k, m, n, r, x, C, E, N, F, T, O, P, w, D, G, Q, R, W, I, aa, fa, na, oa, ha, ba, X, ia, ja, ka, J, la, K, L, ca, S, U, M, ma, B, da, V, Y;
h = Math.round(1E11 * Math.random()) + "";
f = function() {
                    for (var p = [], r = 0; 64 > r;) p[r] = 0 | 4294967296 * Math.sin(++r % Math.PI);
                    return function(z) {
                        var B, G, H, ca = [B = 1732584193, G = 4023233417, ~B, ~G],
                            X = [],
                            x = unescape(encodeURI(z)) + "\u0080",
                            v = x.length;
                        z = --v / 4 + 2 | 15;
                        for (X[--z] = 8 * v; ~v;) X[v >> 2] |= x.charCodeAt(v) << 8 * v--;
                        for (r = x = 0; r < z; r += 16) {
                            for (v = ca; 64 > x; v = [H = v[3], B + ((H = v[0] + [B & G | ~B & H, H & B | ~H & G, B ^ G ^ H, G ^ (B | ~H)][v = x >> 4] + p[x] + ~~X[r | [x, 5 * x + 1, 3 * x + 5, 7 * x][v] & 15]) << (v = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21][4 * v + x++ % 4]) | H >>> -v), B, G]) B = v[1] | 0, G = v[2];
                            for (x = 4; x;) ca[--x] += v[x]
                        }
                        for (z = ""; 32 > x;) z += (ca[x >> 3] >> 4 * (1 ^ x++) & 15).toString(16);
                        return z.split("").reverse().join("")
                    }
                }();

"tryit-" + h + "-" + f(agent + f(agent + f(agent + h + "x")));
"""

        payload = {"chat_style": "chat", "chatHistory": json.dumps(messages)}
        api_key = js2py.eval_js(token_js)
        headers = {
            "api-key": api_key,
            "User-Agent": agent,
            **kwargs.get("headers", {})
        }
        async with ClientSession(
            headers=headers
        ) as session:
            fill = "ing_is"
            fill = f"ack{fill}_a_crim"
            async with session.post(f"https://api.deepai.org/h{fill}e", proxy=proxy, data=payload) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    if stream:
                        try:
                            yield stream.decode("utf-8")
                        except UnicodeDecodeError:
                            yield stream.decode("unicode-escape")


def get_api_key(user_agent: str):
    e = str(round(1E11 * random.random()))

    def hash(data: str):    
        return hashlib.md5(data.encode()).hexdigest()[::-1]

    return f"tryit-{e}-" + hash(user_agent + hash(user_agent + hash(user_agent + e + "x")))