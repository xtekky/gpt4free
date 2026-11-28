import gzip
from flask import Flask, request

# Small payloads (error pages, tiny JSON) are not worth the CPU or the extra
# bytes of a gzip envelope.
_MIN_GZIP_SIZE = 512
_COMPRESSIBLE_TYPES = (
    "text/",
    "application/javascript",
    "application/json",
    "application/manifest+json",
    "image/svg+xml",
)


def create_app(compress: bool = True) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def compress_response(response):
        if not compress or request.method == "HEAD":
            return response
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return response
        if "Content-Encoding" in response.headers:
            return response
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith(_COMPRESSIBLE_TYPES):
            return response
        # Streaming responses (SSE chat completions, proxied media) have no
        # Content-Length. Reading ``response.data`` would consume the whole
        # generator and defeat incremental delivery, so pass them through.
        content_length = response.headers.get("Content-Length")
        if content_length is None:
            return response
        try:
            if int(content_length) < _MIN_GZIP_SIZE:
                return response
        except (TypeError, ValueError):
            return response

        response.direct_passthrough = False
        response.data = gzip.compress(response.data, compresslevel=6)
        response.headers["Content-Encoding"] = "gzip"
        response.headers.add("Vary", "Accept-Encoding")
        response.headers["Content-Length"] = len(response.data)
        return response

    return app
