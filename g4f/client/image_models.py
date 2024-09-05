from __future__ import annotations

from .types import Client, ImageProvider

from ..models import ModelUtils

class ImageModels():
    def __init__(self, client):
        self.client = client
        self.models = ModelUtils.convert

    def get(self, name, default=None):
        model = self.models.get(name)
        if model and model.best_provider:
            return model.best_provider
        return default
