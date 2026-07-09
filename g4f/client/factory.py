from __future__ import annotations

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Type

from ..providers.types import ProviderType, BaseProvider
from ..errors import ProviderNotFoundError
from ..Provider import ProviderUtils
from ..Provider.template import OpenaiTemplate
from ..cookies import get_cookies_dir
from ..config import AppConfig

def create_custom_provider(
    base_url: str,
    api_key: str = None,
    name: str = None,
    working: bool = True,
    default_model: str = "",
    models: List[str] = None,
    **kwargs
) -> ProviderType:
    """
    Create a custom provider class based on OpenaiTemplate.
    
    Args:
        base_url: The base URL for the API (e.g., "https://api.example.com/v1")
        api_key: Optional API key for authentication
        name: Optional name for the provider (defaults to derived from base_url)
        working: Whether the provider is working (default: True)
        default_model: Default model to use
        models: List of available models
        **kwargs: Additional attributes to set on the provider class
    
    Returns:
        A custom provider class that extends OpenaiTemplate
    """
    if name is None:
        # Derive name from base_url
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        name = parsed.netloc.replace(".", "_").replace("-", "_").title().replace("_", "")
        if not name:
            name = "CustomProvider"
    
    # Create a new class that extends OpenaiTemplate
    class_attrs = {
        "url": base_url,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "working": working,
        "default_model": default_model,
        "models": models or [],
        **kwargs
    }
    
    CustomProvider = type(name, (OpenaiTemplate,), class_attrs)
    return CustomProvider


class AbstractClientFactory:
    # Registry of live/custom providers
    _live_providers_url = "https://g4f.dev/dist/js/providers.json"
    _live_providers: Dict[str, Dict] = {}
    
    @classmethod
    def create_provider(
        cls,
        name: str,
        provider: Union[Type[BaseProvider], str],
        base_url: str = None,
        api_key: str = None,
        **kwargs
    ) -> Type[BaseProvider]:
        """
        Register a live/custom provider that can be used by name.
        
        Args:
            name: Name to register the provider under
            provider: Either a provider class or "custom" to create a custom provider
            base_url: Base URL for custom providers
            api_key: API key for custom providers
            **kwargs: Additional arguments for custom provider creation
            
        Returns:
            The registered provider class
        """
        if not isinstance(provider, str):
            return provider
        elif provider.startswith("custom:"):
            base_url = f"https://g4f.space/custom/{provider[7:]}"
            if not api_key and not cls.is_provider_api_key(AppConfig.g4f_api_key):
                api_key = AppConfig.g4f_api_key
            provider = create_custom_provider(base_url, api_key, name=name, **kwargs)            
        elif provider in ProviderUtils.convert:
            provider = ProviderUtils.convert[provider]
        else:
            if not cls._live_providers:
                path = Path(get_cookies_dir()) / "models" / datetime.today().strftime('%Y-%m-%d') / f"providers.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        cls._live_providers = json.load(f)
                if not cls._live_providers:
                    cls._live_providers = requests.get(cls._live_providers_url).json()
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(cls._live_providers, f, indent=4)
            if provider in cls._live_providers.get("providers", {}):
                config = cls._live_providers["providers"][provider]
                if "provider" in config and config.get("provider") in ProviderUtils.convert:
                    return ProviderUtils.convert[config.get("provider")]
                if not api_key and not cls.is_provider_api_key(AppConfig.g4f_api_key) and config.get("backupUrl"):
                    api_key = AppConfig.g4f_api_key
                return create_custom_provider(
                    base_url=config.get("baseUrl") if cls.is_provider_api_key(api_key) or config.get("backupUrl") is None else config.get("backupUrl", config.get("baseUrl")),
                    api_key=api_key,
                    name=provider,
                    default_model=cls._live_providers["defaultModels"].get(provider, ""),
                )
            else:
                try:
                    provider = ProviderUtils.get_by_label(provider)
                except ValueError as e:
                    from g4f.mcp.pa_provider import get_pa_registry
                    registry = get_pa_registry()
                    if provider:
                        if provider.startswith("pa:"):
                            provider = provider[3:]
                        provider_cls = registry.get_provider_class(provider)
                        if provider_cls:
                            return provider_cls
                    raise ProviderNotFoundError(str(e)) from e
        return provider

    @classmethod
    def is_provider_api_key(cls, api_key: str) -> bool:
        return api_key and not api_key.startswith("g4f_") and not api_key.startswith("gfs_")
