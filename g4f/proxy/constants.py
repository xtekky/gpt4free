
# Headers dropped by the /api/https:// CORS proxy: hop-by-hop headers, identity
# leaks and headers recomputed by the outgoing client / streaming response.
_PROXY_DROP_REQUEST_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding", "cookie", "origin", "referer",
    "x-workspace-secret", "g4f-api-key", "x-secret",
    "pragma", "user-agent", "cache-control", "accept", "accept-language",
    "sec-fetch-site", "sec-ch-ua", "sec-ch-ua-platform", "sec-ch-ua-mobile", "sec-fetch-mode", "sec-fetch-dest"
}
_PROXY_DROP_RESPONSE_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "content-encoding", "set-cookie",
    "access-control-allow-origin", "access-control-allow-methods",
    "access-control-allow-headers", "access-control-expose-headers",
    "access-control-allow-credentials",
}
# The proxy only forwards JSON: restrict methods and body size accordingly.
_PROXY_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "OPTIONS"}
_PROXY_MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
_PROXY_JSON_CONTENT_TYPES = ("application/json", "application/problem+json")

__all__ = [
    "_PROXY_DROP_REQUEST_HEADERS",
    "_PROXY_DROP_RESPONSE_HEADERS",
    "_PROXY_ALLOWED_METHODS",
    "_PROXY_MAX_BODY_SIZE",
    "_PROXY_JSON_CONTENT_TYPES",
]