from __future__ import annotations

import json
import os.path
from typing import Iterator
from uuid import uuid4
from functools import partial
import webview
import platformdirs
from plyer import camera
from plyer import filechooser

app_storage_path = platformdirs.user_pictures_dir
user_select_image = partial(
    filechooser.open_file,
    path=platformdirs.user_pictures_dir(),
    filters=[["Image", "*.jpg", "*.jpeg", "*.png", "*.webp", "*.svg"]],
)

from .api import Api

class JsApi(Api):

    def get_conversation(self, options: dict, message_id: str = None, scroll: bool = None) -> Iterator:
        window = webview.windows[0]
        if hasattr(self, "image") and self.image is not None:
            options["image"] = open(self.image, "rb")
        for message in self._create_response_stream(
            self._prepare_conversation_kwargs(options),
            options.get("conversation_id"),
            options.get('provider')
        ):
            if window.evaluate_js(
                f"""
                    is_stopped() ? true :
                    this.add_message_chunk({
                        json.dumps(message)
                    }, {
                        json.dumps(message_id)
                    }, {
                        json.dumps(options.get('provider'))
                    }, {
                        'true' if scroll else 'false'
                    }); is_stopped();
                """):
                break
        self.image = None
        self.set_selected(None)

    def choose_image(self):
        user_select_image(
            on_selection=self.on_image_selection
        )

    def take_picture(self):
        filename = os.path.join(app_storage_path(), f"chat-{uuid4()}.png")
        camera.take_picture(filename=filename, on_complete=self.on_camera)

    def on_image_selection(self, filename):
        filename = filename[0] if isinstance(filename, list) and filename else filename
        if filename and os.path.exists(filename):
            self.image = filename
        else:
            self.image = None
        self.set_selected(None if self.image is None else "image")

    def on_camera(self, filename):
        if filename and os.path.exists(filename):
            self.image = filename
        else:
            self.image = None
        self.set_selected(None if self.image is None else "camera")

    def set_selected(self, input_id: str = None):
        window = webview.windows[0]
        if window is not None:
            window.evaluate_js(
                f"document.querySelector(`.image-label.selected`)?.classList.remove(`selected`);"
            )
            if input_id is not None and input_id in ("image", "camera"):
                window.evaluate_js(
                    f'document.querySelector(`label[for="{input_id}"]`)?.classList.add(`selected`);'
                )

    def get_version(self):
        return super().get_version()

    def get_models(self):
        return super().get_models()

    def get_providers(self):
        return super().get_providers()

    def get_provider_models(self, provider: str, **kwargs):
        return super().get_provider_models(provider, **kwargs)