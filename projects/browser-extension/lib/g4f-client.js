/**
 * g4f extension — minimal client for the local gpt4free FastAPI server.
 *
 * Endpoints used (see g4f/api/__init__.py):
 *   GET  /v1/models                -> { object: "list", data: [{id, image, vision, provider, ...}] }
 *   GET  /v1/providers             -> [{id, url, label, ...}]
 *   POST /v1/chat/completions      -> OpenAI-compatible (stream: SSE `data:` lines)
 *   POST /v1/images/generations    -> { data: [{url | b64_json}], ... }
 *
 * Auth: `Authorization: Bearer <key>` (also accepted: `g4f-api-key: <key>`).
 */

import { normalizeBaseUrl } from "./constants.js";

function authHeaders(apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) {
    headers["Authorization"] = "Bearer " + apiKey;
  }
  return headers;
}

async function parseError(res) {
  let message = `HTTP ${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.error?.message) message = body.error.message;
    else if (typeof body?.detail === "string") message = body.detail;
    else if (body?.error) message = JSON.stringify(body.error);
  } catch {
    /* keep default */
  }
  if (res.status === 401 || res.status === 403) {
    message += " — check the API key in extension settings.";
  } else if (res.status === 0 || res.type === "opaque") {
    message = "Cannot reach the g4f server. Is it running?";
  }
  return new Error(message);
}

/** GET /v1/models */
export async function listModels(settings) {
  const provider = settings.provider || undefined;
  const base = normalizeBaseUrl(settings.serverUrl);
  let url = base + "/v1/models";
  if (provider) {
    url = base + "/api/" + encodeURIComponent(provider) +
      "/models";
  }
  const res = await fetch(url, {
    headers: authHeaders(settings.apiKey),
  });
  if (!res.ok) throw await parseError(res);
  const data = await res.json();
  return (data?.data || []).map((m) => ({
    id: m.id,
    label: m.label || m.id,
    image: !!m.image,
    vision: !!m.vision,
    provider: !!m.provider,
    owned_by: m.owned_by || "",
  }));
}

/** GET /v1/providers */
export async function listProviders(settings) {
  const base = normalizeBaseUrl(settings.serverUrl);
  const res = await fetch(base + "/v1/providers", {
    headers: authHeaders(settings.apiKey),
  });
  if (!res.ok) throw await parseError(res);
  const data = await res.json();
  return (Array.isArray(data) ? data : data?.data || []).map((p) => ({
    id: p.id,
    label: p.label || p.id,
    url: p.url || "",
  }));
}

/** Lightweight reachability check. */
export async function checkHealth(settings) {
  const base = normalizeBaseUrl(settings.serverUrl);
  try {
    const res = await fetch(base + "/v1/models", {
      headers: authHeaders(settings.apiKey),
      signal: AbortSignal.timeout(5000),
    });
    return { ok: res.ok, status: res.status };
  } catch (e) {
    return { ok: false, status: 0, error: String(e?.message || e) };
  }
}

/**
 * POST /v1/chat/completions (streaming).
 *
 * @param {object} settings   Extension settings.
 * @param {Array}  messages   [{role, content}]
 * @param {object} [opts]     { model, provider, temperature, maxTokens, signal, onChunk(text), onDone(fullText) }
 * @returns {Promise<string>} Full assistant text.
 */
export async function chatCompletion(settings, messages, opts = {}) {
  const base = normalizeBaseUrl(settings.serverUrl);
  const provider = opts.provider || settings.provider || undefined;
  const body = {
    model: opts.model || settings.model || undefined,
    provider,
    messages,
    stream: opts.stream ?? settings.stream,
  };
  const temperature = opts.temperature ?? settings.temperature;
  if (temperature !== null && temperature !== undefined && temperature !== "") {
    body.temperature = Number(temperature);
  }
  const maxTokens = opts.maxTokens ?? settings.maxTokens;
  if (maxTokens) body.max_tokens = Number(maxTokens);

  // When a provider is selected, use the provider-scoped route
  // /api/{provider}/{model}/chat/completions (falls back to /v1).
  let url = base + "/v1/chat/completions";
  if (provider) {
    url = base + "/api/" + encodeURIComponent(provider) +
      "/chat/completions";
    delete body.provider;
  }

  const res = await fetch(url, {
    method: "POST",
    headers: authHeaders(settings.apiKey),
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!res.ok) throw await parseError(res);

  const ctype = res.headers.get("content-type") || "";
  const isSSE = ctype.includes("text/event-stream") || body.stream;

  if (!isSSE) {
    const data = await res.json();
    if (data?.error) throw new Error(data.error.message || "Server error");
    const text = data?.choices?.[0]?.message?.content ?? "";
    opts.onChunk?.(text);
    opts.onDone?.(text);
    return text;
  }

  // --- SSE streaming ---
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let json;
        try {
          json = JSON.parse(payload);
        } catch {
          continue;
        }
        if (json?.error) {
          // Server-side provider errors arrive inside the stream (HTTP 200).
          throw new Error(json.error.message || JSON.stringify(json.error));
        }
        const delta = json?.choices?.[0]?.delta;
        const piece =
          delta?.content ??
          json?.choices?.[0]?.message?.content ??
          json?.choices?.[0]?.text ??
          "";
        if (piece) {
          full += piece;
          opts.onChunk?.(piece, full);
        }
      }
    }
  } catch (e) {
    // Release the connection on any mid-stream failure (network drop,
    // server error, abort) so the socket is not leaked.
    try { await reader.cancel(); } catch { /* already closed */ }
    throw e;
  }
  opts.onDone?.(full);
  return full;
}

/**
 * POST /v1/images/generations
 *
 * @returns {Promise<string[]>} Image URLs (data URIs or http URLs).
 */
export async function generateImage(settings, prompt, opts = {}) {
  const base = normalizeBaseUrl(settings.serverUrl);
  const body = {
    prompt,
    n: opts.n ?? settings.imageCount ?? 1,
    response_format: "url",
  };
  if (opts.model || settings.imageModel) body.model = opts.model || settings.imageModel;
  if (opts.provider || settings.provider) body.provider = opts.provider || settings.provider;

  const res = await fetch(base + "/v1/images/generations", {
    method: "POST",
    headers: authHeaders(settings.apiKey),
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!res.ok) throw await parseError(res);
  const data = await res.json();
  if (data?.error) throw new Error(data.error.message || "Image generation failed");
  return (data?.data || [])
    .map((d) => d.url || (d.b64_json ? "data:image/png;base64," + d.b64_json : null))
    .filter(Boolean);
}
