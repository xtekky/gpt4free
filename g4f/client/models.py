from __future__ import annotations

from ..models import ModelUtils, ImageModel, VisionModel
from ..Provider import ProviderUtils
from ..providers.types import ProviderType

class ClientModels():
    def __init__(self, client, provider: ProviderType = None, media_provider: ProviderType = None):
        self.client = client
        self.provider = provider
        self.media_provider = media_provider

    def get(self, name, default=None) -> ProviderType:
        if name in ModelUtils.convert:
            return ModelUtils.convert[name].best_provider
        if name in ProviderUtils.convert:
            return ProviderUtils.convert[name]
        return default

    def get_all(self, api_key: str = None, **kwargs) -> list[str]:
        if self.provider is None:
            return []
        if api_key is None:
            api_key = self.client.api_key
        return self.provider.get_models(
            **kwargs,
            **{} if api_key is None else {"api_key": api_key}
        )

    def get_vision(self, **kwargs) -> list[str]:
        if self.provider is None:
            return [model_id for model_id, model in ModelUtils.convert.items() if isinstance(model, VisionModel)]
        self.get_all(**kwargs)
        if hasattr(self.provider, "vision_models"):
            return self.provider.vision_models
        return []

    def get_media(self, api_key: str = None, **kwargs) -> list[str]:
        if self.media_provider is None:
            return []
        if api_key is None:
            api_key = self.client.api_key
        return self.media_provider.get_models(
            **kwargs,
            **{} if api_key is None else {"api_key": api_key}
        )

    def get_image(self, **kwargs) -> list[str]:
        if self.media_provider is None:
            return [model_id for model_id, model in ModelUtils.convert.items() if isinstance(model, ImageModel)]
        self.get_media(**kwargs)
        if hasattr(self.media_provider, "image_models"):
            return self.media_provider.image_models
        return []

    def get_video(self, **kwargs) -> list[str]:
        if self.media_provider is None:
            return []
        self.get_media(**kwargs)
        if hasattr(self.media_provider, "video_models"):
            return self.media_provider.video_models
        return []