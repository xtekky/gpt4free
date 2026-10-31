from __future__ import annotations

import types, importlib

from ..providers.types import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider, RotatedProvider, ProviderCircuitBreaker
from ..providers.base_provider import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images import CreateImagesProvider
from .. import debug

__others__ = [
    "AnyProvider",
    "BaseProvider",
    "ProviderType",
    "RetryProvider",
    "IterListProvider",
    "RotatedProvider",
    "ProviderCircuitBreaker",
    "AsyncProvider",
    "AsyncGeneratorProvider",
    "CreateImagesProvider",
    "ProviderUtils",
    "ProviderLoader",
]

class ProviderLoader:
    names = [
        "Antigravity",
        "AgentTools",
        "Airforce",
        "BingCreateImages",
        "BraveSearch",
        "BlackForestLabs_Flux1Dev",
        "BlackForestLabs_Flux1KontextDev",
        "BlackboxPro",
        "CachedSearch",
        "Cerebras",
        "Claude",
        "Cloudflare",
        "CohereForAI_C4AI_Command",
        "Copilot",
        "CopilotApp",
        "DeepInfra",
        "DeepSeek",
        "EdgeTTS",
        "ElevenLabs",
        "G4FSpace",
        "GLM",
        "Gemini",
        "GeminiCLI",
        "GeminiPro",
        "GithubCopilot",
        "GoogleAiMode",
        "GoogleSearch",
        "Grok",
        "Groq",
        "HailuoAI",
        "HuggingChat",
        "HuggingFace",
        "HuggingFaceMedia",
        "HuggingSpace",
        "Arena",
        "Local",
        "MarkItDown",
        "MetaAI",
        "MetaAIAccount",
        "MicrosoftDesigner",
        "Nvidia",
        "RelayRouter",
        "KiloCode",
        "LLM7",
        "Ollama",
        "OmniRoute",
        "OpenAIFM",
        "OpenRouterFree",
        "ChatGPT",
        "OperaAria",
        "Perplexity",
        "Pi",
        "Pollinations",
        "PollinationsAudio",
        "PollinationsImage",
        "Puter",
        "Qwen",
        "QwenCode",
        "SearXNG",
        "StabilityAI_SD35Large",
        "TeachAnything",
        "WhiteRabbitNeo",
        "You",
        "YouTube",
        "Yqcloud",
        "gTTS",
    ]
    extra = [
        "APIRoute",
        "OpenaiTemplate",
        "OrcaRouter",
        "AIBadgr",
        "Anthropic",
        "GigaChat",
        "GithubCopilotAPI",
        "CheaperInference",
        "MiniMax",
        "OpenaiAPI",
        "Cohere",
        "OpenRouter",
        "PerplexityApi",
        "PhindAi",
        "Replicate",
        "ThebApi",
        "Together",
        "xAI",
    ]
    loaded = {}
    ignored = []

    @classmethod
    def from_name(cls, name: str) -> ProviderType:
        if not name or not isinstance(name, str):
            return None
        if name in cls.loaded:
            return cls.loaded[name]
        norm_name = name.lower().replace("-", "").replace("_", "")
        if norm_name in cls.loaded:
            return cls.loaded[norm_name]
        try:
            provider = cls._load(name)
            cls.loaded[name] = provider
            cls.loaded[norm_name] = provider
            return provider
        except ImportError:
            if norm_name in ("agent", "agenttools", "agent_tools"):
                actual_name = "AgentTools"
            else:
                lower_map = {p.lower().replace("-", "").replace("_", ""): p for p in cls.names}
                actual_name = lower_map.get(norm_name)
            if actual_name and actual_name != name:
                provider = cls._load(actual_name)
                cls.loaded[name] = provider
                cls.loaded[norm_name] = provider
                return provider
            raise ImportError(f"Provider not found: {name}")

    @classmethod
    def _load(cls, name: str) -> ProviderType:
        debug.log(f"Loading provider: {name}")
        if name in ["AnyProvider", "Default", "default"]:
            from g4f.providers.any_provider import AnyProvider

            return AnyProvider
        elif name == "G4FSpace":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "default"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://g4f.dev"
            cls.loaded[name].active_by_default = True
            return cls.loaded[name]
        try:
            if name in ["HuggingFace", "HuggingFaceMedia", "HuggingChat"]:
                module = "needs_auth.hf"
            elif name in ["HuggingSpace"]:
                module = "hf_space"
            elif name in ["QwenCode"]:
                module = "qwen.QwenCode"
            elif name in ["GithubCopilot"]:
                module = "github.GithubCopilot"
            elif name in ["MiniMax", "HailuoAI"]:
                module = "needs_auth.mini_max"
            elif name in ["OpenaiChat", "ChatGPT"]:
                name = "ChatGPT"
                module = "ChatGPT"
            else:
                module = name
            module = importlib.import_module(f"g4f.Provider.{module}")
            if hasattr(module, name):
                return getattr(module, name)
        except ImportError:
            pass
        for submodule in ["needs_auth", "search", "local", "audio", "extra", "hf_space"]:
            try:
                module = importlib.import_module(f"g4f.Provider.{submodule}.{name}")
                if hasattr(module, name):
                    return getattr(module, name)
            except ImportError:
                pass
        if name == "GLM":
            from g4f.Provider.glm import GLM

            return GLM
        elif name == "RelayRouter":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "custom:srv_mt1wbaxgf9c946af0c58", 
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://relayrouter.org"
            cls.loaded[name].active_by_default = True
            return cls.loaded[name]
        elif name == "KiloCode":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "https://api.kilo.ai/api/gateway"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://kilo.ai"
            cls.loaded[name].active_by_default = True
            cls.loaded[name].default_model = "kilo-auto/free"
            return cls.loaded[name]
        elif name == "LLM7":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "https://api.llm7.io/v1"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://llm7.io"
            cls.loaded[name].active_by_default = True
            cls.loaded[name].default_model = "default"
            cls.loaded[name].models = ["default"]
            cls.loaded[name].add_user = False
            return cls.loaded[name]
        elif name == "OrcaRouter":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                name, "https://api.orcarouter.ai/v1"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://orcarouter.ai"
            return cls.loaded[name]
        elif name in ("AgentTools", "agent-tools", "agent_tools", "agent"):
            from g4f.Provider.AgentTools import AgentTools

            return AgentTools
        elif name == "OpenaiTemplate":
            from g4f.Provider.template.OpenaiTemplate import OpenaiTemplate

            return OpenaiTemplate
        else:
            norm_name = name.lower().replace("-", "").replace("_", "")
            if norm_name == "agent":
                return cls.from_name("AgentTools")
            lower_map = {p.lower().replace("-", "").replace("_", ""): p for p in cls.names}
            if norm_name in lower_map and lower_map[norm_name] != name:
                return cls.from_name(lower_map[norm_name])
            raise ImportError(f"Provider '{name}' not found")

__all__ = __others__ + ProviderLoader.names + ProviderLoader.extra

def __getattr__(name: str):
    if name == "__providers__":
        # Load all providers if specifically requested
        providers_list = []
        for provider_name in ProviderLoader.names:
            if provider_name in ProviderLoader.ignored:
                continue
            try:
                providers_list.append(ProviderLoader.from_name(provider_name))
            except ImportError:
                pass
        return providers_list
    value = globals().get(name)
    # Importing a provider submodule (e.g. g4f.Provider.ChatGPT) sets the
    # module object as a package attribute, shadowing the provider class of
    # the same name.  Prefer the class from ProviderLoader so callers always
    # get a provider, and fall back to the module only if it is not a provider.
    if value is not None and not isinstance(value, types.ModuleType):
        return value
    try:
        return ProviderLoader.from_name(name)
    except ImportError as e:
        if value is not None:
            return value
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from e


def __dir__():
    return __all__


class _ConvertDict(dict):
    def _normalize(self, name: str) -> str:
        return name.lower().replace("-", "").replace("_", "")

    def __contains__(self, item):
        if not isinstance(item, str):
            return False
        try:
            return ProviderLoader.from_name(item) is not None
        except Exception:
            return False

    def __getitem__(self, item):
        try:
            return ProviderLoader.from_name(item)
        except Exception as e:
            raise KeyError(f"Provider '{item}' not found") from e

    def keys(self):
        return ProviderLoader.names

    def items(self):
        return [(k, self[k]) for k in ProviderLoader.names]

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
        try:
            return ProviderLoader.from_name(label)
        except ImportError:
            pass

        # Fallback to search
        for provider_name in ProviderLoader.names:
            if provider_name.lower().startswith(label.lower()):
                try:
                    provider = ProviderLoader.from_name(provider_name)
                    if provider.working:
                        return provider
                except ImportError:
                    pass

        raise ValueError(f"Provider with label '{label}' not found")


import sys
import types


class LazyProviderModule(types.ModuleType):
    def __getattribute__(self, name):
        if name.startswith("__"):
            return super().__getattribute__(name)

        try:
            return __getattr__(name)
        except AttributeError:
            pass

        return super().__getattribute__(name)


sys.modules[__name__].__class__ = LazyProviderModule
