import gzip
from flask import Flask, request

def create_app(compress: bool = True) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def compress_response(response):
        if not compress:
            return response
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
        if response.status_code < 200 or response.status_code >= 300:
            return response
        if 'Content-Encoding' in response.headers:
            return response
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith(('text/', 'application/javascript', 'application/json')):
            return response

        response.direct_passthrough = False
        response.data = gzip.compress(response.data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Vary'] = 'Accept-Encoding'
        response.headers['Content-Length'] = len(response.data)
        return response

    return app