from __future__ import annotations

import json
import flask
import os
import logging
import asyncio
import shutil
import random
import datetime
import tempfile
from flask import Flask, Response, redirect, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import NotFound
from typing import Generator
from pathlib import Path
from urllib.parse import quote_plus
from hashlib import sha256

from ...client.service import convert_to_provider
from ...providers.asyncio import to_sync_generator
from ...providers.response import FinishReason
from ...client.helper import filter_markdown
from ...tools.files import supports_filename, get_streaming, get_bucket_dir, get_buckets
from ...tools.run_tools import iter_run_tools
from ...errors import ProviderNotFoundError
from ...image import is_allowed_extension
from ...cookies import get_cookies_dir
from ...image.copy_images import secure_filename, get_source_url, images_dir
from ... import ChatCompletion
from ... import models
from .api import Api

logger = logging.getLogger(__name__)

def safe_iter_generator(generator: Generator) -> Generator:
    start = next(generator)
    def iter_generator():
        yield start
        yield from generator
    return iter_generator()

class Backend_Api(Api):    
    """
    Handles various endpoints in a Flask application for backend operations.

    This class provides methods to interact with models, providers, and to handle
    various functionalities like conversations, error handling, and version management.

    Attributes:
        app (Flask): A Flask application instance.
        routes (dict): A dictionary mapping API endpoints to their respective handlers.
    """
    def __init__(self, app: Flask) -> None:
        """
        Initialize the backend API with the given Flask application.

        Args:
            app (Flask): Flask application instance to attach routes to.
        """
        self.app: Flask = app
        self.chat_cache = {}

        if app.demo:
            @app.route('/', methods=['GET'])
            def home():
                client_id = os.environ.get("OAUTH_CLIENT_ID", "ed074164-4f8d-4fb2-8bec-44952707965e")
                backend_url = os.environ.get("G4F_BACKEND_URL", "")
                return render_template('demo.html', backend_url=backend_url, client_id=client_id)
        else:
            @app.route('/', methods=['GET'])
            def home():
                return render_template('home.html')

        @app.route('/qrcode', methods=['GET'])
        @app.route('/qrcode/<conversation_id>', methods=['GET'])
        def qrcode(conversation_id: str = ""):
            share_url = os.environ.get("G4F_SHARE_URL", "")
            return render_template('qrcode.html', conversation_id=conversation_id, share_url=share_url)

        @app.route('/backend-api/v2/models', methods=['GET'])
        def jsonify_models(**kwargs):
            response = get_demo_models() if app.demo else self.get_models(**kwargs)
            return jsonify(response)

        @app.route('/backend-api/v2/models/<provider>', methods=['GET'])
        def jsonify_provider_models(**kwargs):
            response = self.get_provider_models(**kwargs)
            return jsonify(response)

        @app.route('/backend-api/v2/providers', methods=['GET'])
        def jsonify_providers(**kwargs):
            response = self.get_providers(**kwargs)
            if isinstance(response, list):
                return jsonify(response)
            return response

        def get_demo_models():
            return [{
                "name": model.name,
                "image": isinstance(model, models.ImageModel),
                "vision": isinstance(model, models.VisionModel),
                "providers": [
                    getattr(provider, "parent", provider.__name__)
                    for provider in providers
                ],
                "demo": True
            }
            for model, providers in models.demo_models.values()]

        def handle_conversation():
            """
            Handles conversation requests and streams responses back.

            Returns:
                Response: A Flask response object for streaming.
            """
            if "json" in request.form:
                json_data = json.loads(request.form['json'])
            else:
                json_data = request.json
            if "files" in request.files:
                media = []
                for file in request.files.getlist('files'):
                    if file.filename != '' and is_allowed_extension(file.filename):
                        newfile = tempfile.TemporaryFile()
                        shutil.copyfileobj(file.stream, newfile)
                        media.append((newfile, file.filename))
                json_data['media'] = media

            if app.demo and not json_data.get("provider"):
                model = json_data.get("model")
                if model != "default" and model in models.demo_models:
                    json_data["provider"] = random.choice(models.demo_models[model][1])
                else:
                    json_data["provider"] = models.HuggingFace
            kwargs = self._prepare_conversation_kwargs(json_data)
            return self.app.response_class(
                self._create_response_stream(
                    kwargs,
                    json_data.get("conversation_id"),
                    json_data.get("provider"),
                    json_data.get("download_media", True),
                ),
                mimetype='text/event-stream'
            )

        @app.route('/backend-api/v2/conversation', methods=['POST'])
        def _handle_conversation():
            return handle_conversation()

        @app.route('/backend-api/v2/usage', methods=['POST'])
        def add_usage():
            cache_dir = Path(get_cookies_dir()) / ".usage"
            cache_file = cache_dir / f"{datetime.date.today()}.jsonl"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_file.open("a" if cache_file.exists() else "w") as f:
                f.write(f"{json.dumps(request.json)}\n")
            return {}

        @app.route('/backend-api/v2/log', methods=['POST'])
        def add_log():
            cache_dir = Path(get_cookies_dir()) / ".logging"
            cache_file = cache_dir / f"{datetime.date.today()}.jsonl"
            cache_dir.mkdir(parents=True, exist_ok=True)
            data = {"origin": request.headers.get("origin"), **request.json}
            with cache_file.open("a" if cache_file.exists() else "w") as f:
                f.write(f"{json.dumps(data)}\n")
            return {}

        @app.route('/backend-api/v2/memory/<user_id>', methods=['POST'])
        def add_memory(user_id: str):
            api_key = request.headers.get("x_api_key")
            json_data = request.json
            from mem0 import MemoryClient
            client = MemoryClient(api_key=api_key)
            client.add(
                [{"role": item["role"], "content": item["content"]} for item in json_data.get("items")],
                user_id=user_id,
                metadata={"conversation_id": json_data.get("id")}
            )
            return {"count": len(json_data.get("items"))}

        @app.route('/backend-api/v2/memory/<user_id>', methods=['GET'])
        def read_memory(user_id: str):
            api_key = request.headers.get("x_api_key")
            from mem0 import MemoryClient
            client = MemoryClient(api_key=api_key)
            if request.args.get("search"):
                return client.search(
                    request.args.get("search"),
                    user_id=user_id,
                    filters=json.loads(request.args.get("filters", "null")),
                    metadata=json.loads(request.args.get("metadata", "null"))
                )
            return client.get_all(
                user_id=user_id,
                page=request.args.get("page", 1),
                page_size=request.args.get("page_size", 100),
                filters=json.loads(request.args.get("filters", "null")),
            )

        self.routes = {
            '/backend-api/v2/version': {
                'function': self.get_version,
                'methods': ['GET']
            },
            '/backend-api/v2/synthesize/<provider>': {
                'function': self.handle_synthesize,
                'methods': ['GET']
            },
            '/images/<path:name>': {
                'function': self.serve_images,
                'methods': ['GET']
            },
            '/media/<path:name>': {
                'function': self.serve_images,
                'methods': ['GET']
            }
        }

        @app.route('/backend-api/v2/create', methods=['GET', 'POST'])
        def create():
            try:
                tool_calls = [{
                    "function": {
                        "name": "bucket_tool"
                    },
                    "type": "function"
                }]
                web_search = request.args.get("web_search")
                if web_search:
                    tool_calls.append({
                        "function": {
                            "name": "search_tool",
                            "arguments": {"query": web_search, "instructions": "", "max_words": 1000} if web_search != "true" else {}
                        },
                        "type": "function"
                    })
                do_filter_markdown = request.args.get("filter_markdown")
                cache_id = request.args.get('cache')
                parameters = {
                    "model": request.args.get("model"),
                    "messages": [{"role": "user", "content": request.args.get("prompt")}],
                    "provider": request.args.get("provider", None),
                    "stream": not do_filter_markdown and not cache_id,
                    "ignore_stream": not request.args.get("stream"),
                    "tool_calls": tool_calls,
                }
                if cache_id:
                    cache_id = sha256(cache_id.encode() + json.dumps(parameters, sort_keys=True).encode()).hexdigest()
                    cache_dir = Path(get_cookies_dir()) / ".scrape_cache" / "create"
                    cache_file = cache_dir / f"{quote_plus(request.args.get('prompt').strip()[:20])}.{cache_id}.txt"
                    if cache_file.exists():
                        with cache_file.open("r") as f:
                            response = f.read()
                    else:
                        response = iter_run_tools(ChatCompletion.create, **parameters)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        copy_response = [chunk for chunk in response]
                        with cache_file.open("w") as f:
                            for chunk in copy_response:
                                f.write(str(chunk))
                        response = copy_response
                else:
                    response = iter_run_tools(ChatCompletion.create, **parameters)

                if do_filter_markdown:
                    return Response(filter_markdown("".join([str(chunk) for chunk in response]), do_filter_markdown), mimetype='text/plain')
                def cast_str():
                    for chunk in response:
                        if isinstance(chunk, FinishReason):
                            yield f"[{chunk.reason}]" if chunk.reason != "stop" else ""
                        elif not isinstance(chunk, Exception):
                            yield str(chunk)
                return Response(cast_str(), mimetype='text/plain')
            except Exception as e:
                logger.exception(e)
                return jsonify({"error": {"message": f"{type(e).__name__}: {e}"}}), 500

        @app.route('/backend-api/v2/files/<bucket_id>', methods=['GET', 'DELETE'])
        def manage_files(bucket_id: str):
            bucket_id = secure_filename(bucket_id)
            bucket_dir = get_bucket_dir(bucket_id)

            if not os.path.isdir(bucket_dir):
                return jsonify({"error": {"message": "Bucket directory not found"}}), 404

            if request.method == 'DELETE':
                try:
                    shutil.rmtree(bucket_dir)
                    return jsonify({"message": "Bucket deleted successfully"}), 200
                except OSError as e:
                    return jsonify({"error": {"message": f"Error deleting bucket: {str(e)}"}}), 500
                except Exception as e:
                    return jsonify({"error": {"message": str(e)}}), 500

            delete_files = request.args.get('delete_files', True)
            refine_chunks_with_spacy = request.args.get('refine_chunks_with_spacy', False)
            event_stream = 'text/event-stream' in request.headers.get('Accept', '')
            mimetype = "text/event-stream" if event_stream else "text/plain";
            return Response(get_streaming(bucket_dir, delete_files, refine_chunks_with_spacy, event_stream), mimetype=mimetype)

        @self.app.route('/backend-api/v2/files/<bucket_id>', methods=['POST'])
        def upload_files(bucket_id: str):
            bucket_id = secure_filename(bucket_id)
            bucket_dir = get_bucket_dir(bucket_id)
            media_dir = os.path.join(bucket_dir, "media")
            os.makedirs(bucket_dir, exist_ok=True)
            filenames = []
            media = []
            for file in request.files.getlist('files'):
                try:
                    filename = secure_filename(file.filename)
                    if is_allowed_extension(filename):
                        os.makedirs(media_dir, exist_ok=True)
                        newfile = os.path.join(media_dir, filename)
                        media.append(filename)
                    elif supports_filename(filename):
                        newfile = os.path.join(bucket_dir, filename)
                        filenames.append(filename)
                    else:
                        continue
                    with open(newfile, 'wb') as f:
                        shutil.copyfileobj(file.stream, f)
                finally:
                    file.stream.close()
            with open(os.path.join(bucket_dir, "files.txt"), 'w') as f:
                [f.write(f"{filename}\n") for filename in filenames]
            return {"bucket_id": bucket_id, "files": filenames, "media": media}

        @app.route('/files/<bucket_id>/media/<filename>', methods=['GET'])
        def get_media(bucket_id, filename, dirname: str = None):
            media_dir = get_bucket_dir(dirname, bucket_id, "media")
            try:
                return send_from_directory(os.path.abspath(media_dir), filename)
            except NotFound:
                source_url = get_source_url(request.query_string.decode())
                if source_url is not None:
                    return redirect(source_url)
                raise

        @app.route('/search/<search>', methods=['GET'])
        def find_media(search: str):
            search = [secure_filename(chunk.lower()) for chunk in search.split("+")]
            if not os.access(images_dir, os.R_OK):
                return jsonify({"error": {"message": "Not found"}}), 404
            match_files = {}
            for root, _, files in os.walk(images_dir):
                for file in files:
                    mime_type = is_allowed_extension(file)
                    if mime_type is not None:
                        mime_type = secure_filename(mime_type)
                        for tag in search:
                            if tag in mime_type:
                                match_files[file] = match_files.get(file, 0) + 1
                                break
                    for tag in search:
                        if tag in file.lower():
                            match_files[file] = match_files.get(file, 0) + 1
            match_files = [file for file, count in match_files.items() if count >= request.args.get("min", len(search))]
            if int(request.args.get("skip", 0)) >= len(match_files):
                return jsonify({"error": {"message": "Not found"}}), 404
            if (request.args.get("random", False)):
                return redirect(f"/media/{random.choice(match_files)}"), 302
            return redirect(f"/media/{match_files[int(request.args.get('skip', 0))]}", 302)

        @app.route('/backend-api/v2/upload_cookies', methods=['POST'])
        def upload_cookies():
            file = None
            if "file" in request.files:
                file = request.files['file']
                if file.filename == '':
                    return 'No selected file', 400
            if file and file.filename.endswith(".json") or file.filename.endswith(".har"):
                filename = secure_filename(file.filename)
                file.save(os.path.join(get_cookies_dir(), filename))
                return "File saved", 200
            return 'Not supported file', 400

        @self.app.route('/backend-api/v2/chat/<share_id>', methods=['GET'])
        def get_chat(share_id: str) -> str:
            share_id = secure_filename(share_id)
            if self.chat_cache.get(share_id, 0) == int(request.headers.get("if-none-match", 0)):
                return jsonify({"error": {"message": "Not modified"}}), 304
            file = get_bucket_dir(share_id, "chat.json")
            if not os.path.isfile(file):
                return jsonify({"error": {"message": "Not found"}}), 404
            with open(file, 'r') as f:
                chat_data = json.load(f)
                if chat_data.get("updated", 0) == int(request.headers.get("if-none-match", 0)):
                    return jsonify({"error": {"message": "Not modified"}}), 304
                self.chat_cache[share_id] = chat_data.get("updated", 0)
                return jsonify(chat_data), 200

        @self.app.route('/backend-api/v2/chat/<share_id>', methods=['POST'])
        def upload_chat(share_id: str) -> dict:
            chat_data = {**request.json}
            updated = chat_data.get("updated", 0)
            cache_value = self.chat_cache.get(share_id, 0)
            if updated == cache_value:
                return jsonify({"error": {"message": "invalid date"}}), 400
            share_id = secure_filename(share_id)
            bucket_dir = get_bucket_dir(share_id)
            os.makedirs(bucket_dir, exist_ok=True)
            with open(os.path.join(bucket_dir, "chat.json"), 'w') as f:
                json.dump(chat_data, f)
            self.chat_cache[share_id] = updated
            return {"share_id": share_id}

    def handle_synthesize(self, provider: str):
        try:
            provider_handler = convert_to_provider(provider)
        except ProviderNotFoundError:
            return "Provider not found", 404
        if not hasattr(provider_handler, "synthesize"):
            return "Provider doesn't support synthesize", 500
        response_data = provider_handler.synthesize({**request.args})
        if asyncio.iscoroutinefunction(provider_handler.synthesize):
            response_data = asyncio.run(response_data)
        else:
            if hasattr(response_data, "__aiter__"):
                response_data = to_sync_generator(response_data)
            response_data = safe_iter_generator(response_data)
        content_type = getattr(provider_handler, "synthesize_content_type", "application/octet-stream")
        response = flask.Response(response_data, content_type=content_type)
        response.headers['Cache-Control'] = "max-age=604800"
        return response

    def get_provider_models(self, provider: str):
        api_key = request.headers.get("x_api_key")
        api_base = request.headers.get("x_api_base")
        models = super().get_provider_models(provider, api_key, api_base)
        if models is None:
            return "Provider not found", 404
        return models

    def _format_json(self, response_type: str, content = None, **kwargs) -> str:
        """
        Formats and returns a JSON response.

        Args:
            response_type (str): The type of the response.
            content: The content to be included in the response.

        Returns:
            str: A JSON formatted string.
        """
        return json.dumps(super()._format_json(response_type, content, **kwargs)) + "\n"
