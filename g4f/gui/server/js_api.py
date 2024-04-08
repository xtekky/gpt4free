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

try:
    from android.runnable import run_on_ui_thread
    import android.permissions
    from android.permissions import Permission
    from android.permissions import _RequestPermissionsManager
    _RequestPermissionsManager.register_callback()
    from .android_gallery import user_select_image
    has_android = True
except:
    run_on_ui_thread = lambda a : a
    has_android = False

from .api import Api

class JsApi(Api):
    def get_conversation(self, options: dict, **kwargs) -> Iterator:
        window = webview.windows[0]
        if hasattr(self, "image") and self.image is not None:
            kwargs["image"] = open(self.image, "rb")
        for message in self._create_response_stream(
            self._prepare_conversation_kwargs(options, kwargs),
            options.get("conversation_id"),
            options.get('provider')
        ):
            if not window.evaluate_js(f"if (!this.abort) this.add_message_chunk({json.dumps(message)}); !this.abort && !this.error;"):
                break
        self.image = None
        self.set_selected(None)

    @run_on_ui_thread
    def choose_image(self):
        self.request_permissions()
        user_select_image(
            on_selection=self.on_image_selection
        )

    @run_on_ui_thread
    def take_picture(self):
        self.request_permissions()
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

    def request_permissions(self):
        if has_android:
            android.permissions.request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE
            ])