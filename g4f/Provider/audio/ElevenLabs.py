from __future__ import annotations

import os
import json
import base64

try:
    import zendriver as nodriver

    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...typing import AsyncResult, Messages
from ...providers.response import AudioResponse
from ...image.copy_images import get_filename, get_media_dir, ensure_media_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_message
from ...cookies import get_cookies
from ...requests import get_nodriver, get_cookie_params_from_dict


class ElevenLabs(AsyncGeneratorProvider, ProviderModelMixin):
    """Anonymous ElevenLabs TTS via browser hcaptcha token.

    Requires:
        pip install "g4f[slim]"  (zendriver; included in slim/all extras)
        hcaptcha accessibility cookie — log in once at
        https://dashboard.hcaptcha.com/signup?type=accessibility in your
        browser (read automatically via browser_cookie3), or pass
        hc_accessibility="..." kwarg. Cookie valid ~24h.

    Usage:
        client.chat.completions.create(
            model="elevenlabs-tts",
            messages=[{"role": "user", "content": "Olá"}],
            audio={"voice": "tS45q0QcrDHqHoaWdCDR", "language": "pt"},
        )
    """

    label = "ElevenLabs TTS"
    url = "https://elevenlabs.io"
    working = has_nodriver
    use_nodriver = has_nodriver
    needs_auth = True  # hcaptcha accessibility cookie

    model_id = "elevenlabs-tts"
    default_model = "eleven_v3"
    default_voice = "tS45q0QcrDHqHoaWdCDR"
    default_format = "mp3_44100_128"
    default_language = "pt"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        audio: dict = {},
        proxy: str = None,
        **kwargs,
    ) -> AsyncResult:
        prompt = get_last_message(messages, prompt)
        if not prompt:
            raise ValueError("Prompt is empty.")

        voice = audio.get("voice", cls.default_voice)
        model_id = audio.get("model", cls.default_model)
        output_format = audio.get("format", cls.default_format)
        language = audio.get("language", cls.default_language)

        # hcaptcha accessibility cookie — kwarg, then browser cookie store
        cookie_val = kwargs.get("hc_accessibility")
        if not cookie_val:
            cookie_val = get_cookies(
                ".hcaptcha.com", raise_requirements_error=False
            ).get("hc_accessibility")
        if not cookie_val:
            raise RuntimeError(
                "hCaptcha accessibility cookie required: log in at "
                "https://dashboard.hcaptcha.com/signup?type=accessibility "
                'in your browser, or pass hc_accessibility="...".'
            )

        browser, stop_browser = await get_nodriver(proxy=proxy)
        try:
            cookie_params = get_cookie_params_from_dict(
                {"hc_accessibility": cookie_val}, domain=".hcaptcha.com"
            )
            await browser.cookies.set_all(cookie_params)

            page = await browser.get(cls.url)
            # Wait until DOM body exists before injecting elements
            await page.evaluate(
                """
                document.body || new Promise(r => {
                    document.addEventListener('DOMContentLoaded', r, {once: true});
                })
            """,
                await_promise=True,
            )

            chunks = await page.evaluate(
                _build_js(prompt, voice, model_id, output_format, language),
                await_promise=True,
            )
        finally:
            await stop_browser()

        if not chunks:
            raise RuntimeError("No audio chunks received from ElevenLabs.")

        audio_bytes = b"".join(base64.b64decode(c) for c in chunks)
        # ponytail: whole response buffered; streaming per-chunk yield if memory matters
        filename = get_filename([cls.__name__], prompt, ".mp3", prompt)
        target_path = os.path.join(get_media_dir(), filename)
        ensure_media_dir()
        with open(target_path, "wb") as f:
            f.write(audio_bytes)

        yield AudioResponse(f"/media/{filename}", text=prompt)


def _build_js(
    prompt: str, voice: str, model_id: str, output_format: str, language: str
) -> str:
    """Render invisible hcaptcha, fetch elevenlabs API with token, collect SSE audio chunks."""
    _text = json.dumps(prompt)
    _voice = json.dumps(voice)
    _model = json.dumps(model_id)
    _format = json.dumps(output_format)
    _lang = json.dumps(language)

    return f"""(async () => {{
    const c = document.createElement('div');
    c.id = 'hc-' + Math.random().toString(36).slice(2);
    c.style.display = 'none';
    document.body.appendChild(c);
    if (typeof hcaptcha === 'undefined') {{
        await new Promise((r, e) => {{
            const s = document.createElement('script');
            s.src = 'https://js.hcaptcha.com/1/api.js';
            s.onload = r; s.onerror = e;
            document.head.appendChild(s);
        }});
        await new Promise(r => setTimeout(r, 1500));
    }}
    const token = await new Promise((res, rej) => {{
        const w = hcaptcha.render(c.id, {{
            sitekey: '8e58fe8c-1a48-4f94-88ae-8e90b586a192',
            size: 'invisible',
            callback: (t) => res(t),
            'error-callback': (e) => rej(new Error('hcaptcha: ' + e))
        }});
        hcaptcha.execute(w);
    }});
    const r = await fetch(
        'https://api.elevenlabs.io/v1/text-to-speech/' + {_voice} + '/stream/with-timestamps/anonymous?output_format=' + {_format},
        {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
                text: {_text},
                model_id: {_model},
                voice_settings: {{speed: 1}},
                hcaptcha_token: token,
                language_code: {_lang}
            }})
        }}
    );
    if (!r.ok) throw new Error(r.status + ' ' + await r.text());
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = '', chunks = [];
    while (true) {{
        const {{done, value}} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {{stream: true}});
        const lines = buf.split('\\n');
        buf = lines.pop() || '';
        for (const ln of lines) {{
            const t = ln.trim();
            if (!t) continue;
            // Accept both plain NDJSON and SSE "data:" prefixed lines
            const d = t.startsWith('data:') ? t.slice(5).trim() : t;
            try {{
                const p = JSON.parse(d);
                if (p.audio_base64) chunks.push(p.audio_base64);
            }} catch (e) {{}}
        }}
    }}
    if (!chunks.length) throw new Error('No audio chunks');
    return chunks;
}})()"""
