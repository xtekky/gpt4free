from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider, RotatedProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider
from .. import debug

__map_paths__ = {
    "AIBadgr": "g4f.Provider.needs_auth.AIBadgr",
    "Anthropic": "g4f.Provider.needs_auth.Anthropic",
    "Antigravity": "g4f.Provider.needs_auth.Antigravity",
    "ApiAirforce": "g4f.Provider.needs_auth.ApiAirforce",
    "AsyncGeneratorProvider": "g4f.providers.base_provider",
    "AsyncProvider": "g4f.providers.base_provider",
    "Azure": "g4f.Provider.needs_auth.Azure",
    "BackendApi": "g4f.Provider.template.BackendApi",
    "BaseProvider": "g4f.providers.types",
    "BingCreateImages": "g4f.Provider.needs_auth.BingCreateImages",
    "BlackForestLabs_Flux1Dev": "g4f.Provider.hf_space.BlackForestLabs_Flux1Dev",
    "BlackForestLabs_Flux1KontextDev": "g4f.Provider.hf_space.BlackForestLabs_Flux1KontextDev",
    "BlackboxPro": "g4f.Provider.needs_auth.BlackboxPro",
    "CablyAI": "g4f.Provider.needs_auth.CablyAI",
    "CachedSearch": "g4f.Provider.search.CachedSearch",
    "Cerebras": "g4f.Provider.needs_auth.Cerebras",
    "Claude": "g4f.Provider.needs_auth.Claude",
    "Cloudflare": "g4f.Provider.Cloudflare",
    "Cohere": "g4f.Provider.needs_auth.Cohere",
    "CohereForAI_C4AI_Command": "g4f.Provider.hf_space.CohereForAI_C4AI_Command",
    "Copilot": "g4f.Provider.Copilot",
    "CopilotAccount": "g4f.Provider.needs_auth.CopilotAccount",
    "CopilotApp": "g4f.Provider.CopilotApp",
    "CopilotSession": "g4f.Provider.CopilotSession",
    "CreateImagesProvider": "g4f.providers.create_images",
    "Custom": "g4f.Provider.needs_auth.Custom",
    "DeepInfra": "g4f.Provider.deepinfra",
    "DeepSeek": "g4f.Provider.needs_auth.DeepSeek",
    "DeepSeekAPI": "g4f.Provider.needs_auth.DeepSeekAPI",
    "EasyChat": "g4f.Provider.EasyChat",
    "EdgeTTS": "g4f.Provider.audio.EdgeTTS",
    "Feature": "g4f.Provider.needs_auth.Custom",
    "Felo": "g4f.Provider.Felo",
    "FenayAI": "g4f.Provider.needs_auth.FenayAI",
    "GLM": "g4f.Provider.GLM",
    "Gemini": "g4f.Provider.needs_auth.Gemini",
    "GeminiCLI": "g4f.Provider.needs_auth.GeminiCLI",
    "GeminiPro": "g4f.Provider.needs_auth.GeminiPro",
    "GigaChat": "g4f.Provider.needs_auth.GigaChat",
    "GithubCopilot": "g4f.Provider.github.GithubCopilot",
    "GithubCopilotAPI": "g4f.Provider.needs_auth.GithubCopilotAPI",
    "GlhfChat": "g4f.Provider.needs_auth.GlhfChat",
    "GoogleSearch": "g4f.Provider.search.GoogleSearch",
    "GradientNetwork": "g4f.Provider.GradientNetwork",
    "Grok": "g4f.Provider.needs_auth.Grok",
    "Groq": "g4f.Provider.needs_auth.Groq",
    "HailuoAI": "g4f.Provider.needs_auth.mini_max.HailuoAI",
    "HuggingChat": "g4f.Provider.needs_auth.hf.HuggingChat",
    "HuggingFace": "g4f.Provider.needs_auth.hf",
    "HuggingFaceAPI": "g4f.Provider.needs_auth.hf.HuggingFaceAPI",
    "HuggingFaceInference": "g4f.Provider.needs_auth.hf.HuggingFaceInference",
    "HuggingFaceMedia": "g4f.Provider.needs_auth.hf.HuggingFaceMedia",
    "HuggingSpace": "g4f.Provider.hf_space",
    "IterListProvider": "g4f.providers.retry_provider",
    "LMArena": "g4f.Provider.needs_auth.LMArena",
    "Local": "g4f.Provider.local.Local",
    "MarkItDown": "g4f.Provider.audio.MarkItDown",
    "MetaAI": "g4f.Provider.needs_auth.MetaAI",
    "MetaAIAccount": "g4f.Provider.needs_auth.MetaAIAccount",
    "MicrosoftDesigner": "g4f.Provider.needs_auth.MicrosoftDesigner",
    "MiniMax": "g4f.Provider.needs_auth.mini_max.MiniMax",
    "Nvidia": "g4f.Provider.needs_auth.Nvidia",
    "Ollama": "g4f.Provider.local.Ollama",
    "OllamaSwarm": "g4f.Provider.OllamaSwarm",
    "OpenAIFM": "g4f.Provider.audio.OpenAIFM",
    "OpenRouter": "g4f.Provider.needs_auth.OpenRouter",
    "OpenRouterFree": "g4f.Provider.needs_auth.OpenRouter",
    "OpenaiAPI": "g4f.Provider.needs_auth.OpenaiAPI",
    "OpenaiAccount": "g4f.Provider.needs_auth.OpenaiAccount",
    "OpenaiChat": "g4f.Provider.needs_auth.OpenaiChat",
    "OpenaiTemplate": "g4f.Provider.template.OpenaiTemplate",
    "OperaAria": "g4f.Provider.OperaAria",
    "Perplexity": "g4f.Provider.Perplexity",
    "PerplexityApi": "g4f.Provider.needs_auth.PerplexityApi",
    "PhindAi": "g4f.Provider.PhindAi",
    "Pi": "g4f.Provider.needs_auth.Pi",
    "PollinationsAI": "g4f.Provider.PollinationsAI",
    "PollinationsAudio": "g4f.Provider.audio.PollinationsAudio",
    "PollinationsImage": "g4f.Provider.PollinationsImage",
    "PuterJS": "g4f.Provider.needs_auth.PuterJS",
    "Qwen": "g4f.Provider.Qwen",
    "QwenCode": "g4f.Provider.qwen.QwenCode",
    "Reka": "g4f.Provider.needs_auth.Reka",
    "Replicate": "g4f.Provider.needs_auth.Replicate",
    "RetryProvider": "g4f.providers.retry_provider",
    "RotatedProvider": "g4f.providers.retry_provider",
    "SearXNG": "g4f.Provider.search.SearXNG",
    "StabilityAI_SD35Large": "g4f.Provider.hf_space.StabilityAI_SD35Large",
    "TeachAnything": "g4f.Provider.TeachAnything",
    "ThebApi": "g4f.Provider.needs_auth.ThebApi",
    "Together": "g4f.Provider.needs_auth.Together",
    "Video": "g4f.Provider.needs_auth.Video",
    "WeWordle": "g4f.Provider.WeWordle",
    "WhiteRabbitNeo": "g4f.Provider.needs_auth.WhiteRabbitNeo",
    "You": "g4f.Provider.needs_auth.You",
    "YouTube": "g4f.Provider.search.YouTube",
    "Yqcloud": "g4f.Provider.Yqcloud",
    "Yupp": "g4f.Provider.Yupp",
    "gTTS": "g4f.Provider.audio.gTTS",
    "xAI": "g4f.Provider.needs_auth.xAI",
    "AnyProvider": "g4f.providers.any_provider",
}


__all__ = [
    "BaseProvider",
    "ProviderType",
    "RetryProvider",
    "IterListProvider",
    "RotatedProvider",
    "AsyncProvider",
    "AsyncGeneratorProvider",
    "CreateImagesProvider",
    "ProviderUtils",
    "__providers__",
    "__map__",
] + list(__map_paths__.keys())

_loaded_providers = {}

def __getattr__(name: str):
    if name in __map_paths__:
        module_path = __map_paths__[name]
        if not isinstance(module_path, str):
            return module_path
        if name not in _loaded_providers:
            import sys
            import importlib
            try:
                module = importlib.import_module(module_path)
                _loaded_providers[name] = getattr(module, name)
                setattr(sys.modules["g4f.Provider"], name, _loaded_providers[name])
            except ImportError as e:
                debug.error(f"Failed to load provider {name}: {e}")
                raise AttributeError(f"Provider {name} could not be loaded") from e
        return _loaded_providers[name]
    if name == "__providers__":
        # Load all providers if specifically requested
        providers_list = []
        for provider_name in __map_paths__.keys():
            try:
                providers_list.append(__getattr__(provider_name))
            except AttributeError:
                pass
        return providers_list

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
def __dir__():
    return __all__

class _ConvertDict(dict):
    def __contains__(self, item):
        return item in __map_paths__
    def __getitem__(self, item):
        if item in __map_paths__:
            if not isinstance(__map_paths__[item], str):
                return __map_paths__[item]
            return __getattr__(item)
        raise KeyError(item)
    def values(self):
        return __getattr__("__providers__")
    def keys(self):
        return __map_paths__.keys()
    def items(self):
        return [(k, self[k]) for k in __map_paths__.keys()]
    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

__map__ = _ConvertDict()

class ProviderUtils:
    convert = __map__

    @classmethod
    def get_by_label(cls, label: str) -> ProviderType:
        if not label:
            raise ValueError("Label must be provided")
            
        # Check explicit map
        if label in __map__:
            return __map__[label]
            
        # Fallback to search
        for provider_name in __map_paths__.keys():
            if provider_name.lower().startswith(label.lower()):
                provider = __map__[provider_name]
                if provider.working:
                    return provider
                    
        raise ValueError(f"Provider with label '{label}' not found")

import sys
import types

class LazyProviderModule(types.ModuleType):
    def __getattribute__(self, name):
        if name.startswith('__'):
            return super().__getattribute__(name)
        
        map_paths = super().__getattribute__('__map_paths__')
        if name in map_paths:
            return super().__getattribute__('__getattr__')(name)
            
        return super().__getattribute__(name)

sys.modules[__name__].__class__ = LazyProviderModule

