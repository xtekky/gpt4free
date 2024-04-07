from __future__ import annotations

from .types import Client, ImageProvider

from ..Provider.BingCreateImages import BingCreateImages
from ..Provider.needs_auth import Gemini, OpenaiChat
from ..Provider.You import You

class ImageModels():
    gemini = Gemini
    openai = OpenaiChat
    you = You

    def __init__(self, client: Client) -> None:
        self.client = client
        self.default = BingCreateImages(proxy=self.client.get_proxy())

    def get(self, name: str, default: ImageProvider = None) -> ImageProvider:
        return getattr(self, name) if hasattr(self, name) else default or self.default
