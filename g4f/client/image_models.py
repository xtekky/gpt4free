from __future__ import annotations

from ..models import ModelUtils
from ..Provider import ProviderUtils

class ImageModels():
    def __init__(self, client):
        self.client = client

    def get(self, name, default=None):
        if name in ModelUtils.convert:
            return ModelUtils.convert[name].best_provider
        if name in ProviderUtils.convert:
            return ProviderUtils.convert[name]
        return default