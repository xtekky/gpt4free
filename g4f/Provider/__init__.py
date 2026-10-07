from __future__ import annotations

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
        "OpenaiTemplate",
        "OrcaRouter",
        "OpenCode",
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

        if name == "AnyProvider":
            from g4f.providers.any_provider import AnyProvider

            return AnyProvider
        elif name == "AIBadgr":
            from g4f.Provider.extra.AIBadgr import AIBadgr

            return AIBadgr
        elif name == "Anthropic":
            from g4f.Provider.needs_auth.Anthropic import Anthropic

            return Anthropic
        elif name == "Antigravity":
            from g4f.Provider.needs_auth.Antigravity import Antigravity

            return Antigravity
        elif name == "Airforce" or name == "ApiAirforce":
            from g4f.Provider.needs_auth.Airforce import Airforce

            return Airforce
        elif name == "BingCreateImages":
            from g4f.Provider.needs_auth.BingCreateImages import BingCreateImages

            return BingCreateImages
        elif name == "BraveSearch":
            from g4f.Provider.BraveSearch import BraveSearch

            return BraveSearch
        elif name == "BlackForestLabs_Flux1Dev":
            from g4f.Provider.hf_space.BlackForestLabs_Flux1Dev import (
                BlackForestLabs_Flux1Dev,
            )

            return BlackForestLabs_Flux1Dev
        elif name == "BlackForestLabs_Flux1KontextDev":
            from g4f.Provider.hf_space.BlackForestLabs_Flux1KontextDev import (
                BlackForestLabs_Flux1KontextDev,
            )

            return BlackForestLabs_Flux1KontextDev
        elif name == "BlackboxPro":
            from g4f.Provider.needs_auth.BlackboxPro import BlackboxPro

            return BlackboxPro
        elif name == "CachedSearch":
            from g4f.Provider.search.CachedSearch import CachedSearch

            return CachedSearch
        elif name == "Cerebras":
            from g4f.Provider.needs_auth.Cerebras import Cerebras

            return Cerebras
        elif name == "CheaperInference":
            from g4f.Provider.extra.CheaperInference import CheaperInference

            return CheaperInference
        elif name == "Claude":
            from g4f.Provider.needs_auth.Claude import Claude

            return Claude
        elif name == "Cloudflare":
            from g4f.Provider.Cloudflare import Cloudflare

            return Cloudflare
        elif name == "Cohere":
            from g4f.Provider.needs_auth.Cohere import Cohere

            return Cohere
        elif name == "CohereForAI_C4AI_Command":
            from g4f.Provider.hf_space.CohereForAI_C4AI_Command import (
                CohereForAI_C4AI_Command,
            )

            return CohereForAI_C4AI_Command
        elif name == "Copilot" or name == "CopilotSession":
            from g4f.Provider.Copilot import Copilot

            return Copilot
        elif name == "CopilotAccount":
            from g4f.Provider.needs_auth.CopilotAccount import CopilotAccount

            return CopilotAccount
        elif name == "CopilotApp":
            from g4f.Provider.CopilotApp import CopilotApp

            return CopilotApp
        elif name == "CopilotSession":
            from g4f.Provider.CopilotSession import CopilotSession

            return CopilotSession
        elif name == "Custom":
            from g4f.Provider.needs_auth.Custom import Custom

            return Custom
        elif name == "DeepInfra":
            from .DeepInfra import DeepInfra

            return DeepInfra
        elif name == "DeepSeek" or name == "DeepSeekAPI":
            from g4f.Provider.needs_auth.DeepSeek import DeepSeek

            return DeepSeek
        elif name == "Default":
            from g4f.providers.any_provider import DefaultProvider

            return DefaultProvider
        elif name == "EdgeTTS":
            from g4f.Provider.audio.EdgeTTS import EdgeTTS

            return EdgeTTS
        elif name == "ElevenLabs":
            from g4f.Provider.audio.ElevenLabs import ElevenLabs

            return ElevenLabs
        elif name == "G4FSpace":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "default"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://g4f.dev"
            cls.loaded[name].active_by_default = True
            return cls.loaded[name]
        elif name == "GLM":
            from g4f.Provider.glm import GLM

            return GLM
        elif name == "Gemini":
            from g4f.Provider.needs_auth.Gemini import Gemini

            return Gemini
        elif name == "GeminiCLI":
            from g4f.Provider.needs_auth.GeminiCLI import GeminiCLI

            return GeminiCLI
        elif name == "GeminiPro":
            from g4f.Provider.needs_auth.GeminiPro import GeminiPro

            return GeminiPro
        elif name == "GigaChat":
            from g4f.Provider.needs_auth.GigaChat import GigaChat

            return GigaChat
        elif name == "GithubCopilot":
            from g4f.Provider.github.GithubCopilot import GithubCopilot

            return GithubCopilot
        elif name == "GithubCopilotAPI":
            from g4f.Provider.extra.GithubCopilotAPI import GithubCopilotAPI

            return GithubCopilotAPI
        elif name == "GoogleAiMode":
            from g4f.Provider.search.GoogleAiMode import GoogleAiMode

            return GoogleAiMode
        elif name == "GoogleSearch":
            from g4f.Provider.search.GoogleSearch import GoogleSearch

            return GoogleSearch
        elif name == "Grok":
            from g4f.Provider.needs_auth.Grok import Grok

            return Grok
        elif name == "Groq":
            from g4f.Provider.needs_auth.Groq import Groq

            return Groq
        elif name == "HailuoAI":
            from g4f.Provider.needs_auth.mini_max.HailuoAI import HailuoAI

            return HailuoAI
        elif name == "HuggingChat":
            from g4f.Provider.needs_auth.hf.HuggingChat import HuggingChat

            return HuggingChat
        elif name == "HuggingFace" or name == "HuggingFaceAPI":
            from g4f.Provider.needs_auth.hf import HuggingFace

            return HuggingFace
        elif name == "HuggingFaceMedia":
            from g4f.Provider.needs_auth.hf.HuggingFaceMedia import HuggingFaceMedia

            return HuggingFaceMedia
        elif name == "HuggingSpace":
            from g4f.Provider.hf_space import HuggingSpace

            return HuggingSpace
        elif name == "Arena" or name == "LMArena":
            from g4f.Provider.needs_auth.Arena import Arena

            return Arena
        elif name == "Local":
            from g4f.Provider.local import Local

            return Local
        elif name == "MarkItDown":
            from g4f.Provider.audio.MarkItDown import MarkItDown

            return MarkItDown
        elif name == "MetaAI":
            from g4f.Provider.needs_auth.MetaAI import MetaAI

            return MetaAI
        elif name == "MetaAIAccount":
            from g4f.Provider.needs_auth.MetaAIAccount import MetaAIAccount

            return MetaAIAccount
        elif name == "MicrosoftDesigner":
            from g4f.Provider.needs_auth.MicrosoftDesigner import MicrosoftDesigner

            return MicrosoftDesigner
        elif name == "MiniMax":
            from g4f.Provider.needs_auth.mini_max.MiniMax import MiniMax

            return MiniMax
        elif name == "Nvidia":
            from g4f.Provider.needs_auth.Nvidia import Nvidia

            return Nvidia
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
        elif name == "Ollama":
            from g4f.Provider.local.Ollama import Ollama

            return Ollama
        elif name == "OpenAIFM":
            from g4f.Provider.audio.OpenAIFM import OpenAIFM

            return OpenAIFM
        elif name == "OpenCode":
            import time
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, "https://opencode.ai/zen/v1"
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://opencode.ai"
            cls.loaded[name].active_by_default = True
            cls.loaded[name].default_model = "space-bunny-free"
            cls.loaded[name].headers = {
                "Content-Type": "application/json",
                "User-Agent": "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
                "x-opencode-client": "cli",
                "x-opencode-project": "global",
                "x-opencode-session": f"ses_{int(time.time())}",
                "x-opencode-request": f"msg_{int(time.time())}",
            }
            return cls.loaded[name]

        elif name == "OpenRouter":
            from g4f.Provider.needs_auth.OpenRouter import OpenRouter

            return OpenRouter
        elif name == "OpenRouterFree":
            from g4f.Provider.needs_auth.OpenRouter import OpenRouterFree

            return OpenRouterFree
        elif name == "OrcaRouter":
            from ..client.factory import AbstractClientFactory
            cls.loaded[name] = AbstractClientFactory.create_provider(
                None, name.lower()
            )
            cls.loaded[name].__name__ = name
            cls.loaded[name].url = "https://orcarouter.ai"
            cls.loaded[name].active_by_default = True
            cls.loaded[name].supports_native_tools = True
            return cls.loaded[name]
        elif name in ("AgentTools", "agent-tools", "agent_tools", "agent"):
            from g4f.Provider.AgentTools import AgentTools

            return AgentTools
        elif name == "OpenaiAPI":
            from g4f.Provider.extra.OpenaiAPI import OpenaiAPI

            return OpenaiAPI
        elif name == "OpenaiAccount":
            from g4f.Provider.needs_auth.OpenaiAccount import OpenaiAccount

            return OpenaiAccount
        elif name == "ChatGPT" or name == "OpenaiChat":
            from g4f.Provider.ChatGPT import ChatGPT

            return ChatGPT
        elif name == "OpenaiTemplate":
            from g4f.Provider.template.OpenaiTemplate import OpenaiTemplate

            return OpenaiTemplate
        elif name == "OperaAria":
            from g4f.Provider.OperaAria import OperaAria

            return OperaAria
        elif name == "Perplexity":
            from g4f.Provider.Perplexity import Perplexity

            return Perplexity
        elif name == "PerplexityApi":
            from g4f.Provider.extra.PerplexityApi import PerplexityApi

            return PerplexityApi
        elif name == "PhindAi":
            from g4f.Provider.PhindAi import PhindAi

            return PhindAi
        elif name == "Pi":
            from g4f.Provider.needs_auth.Pi import Pi

            return Pi
        elif name == "Pollinations" or name == "PollinationsAI":
            from g4f.Provider.Pollinations import Pollinations

            return Pollinations
        elif name == "PollinationsAudio":
            from g4f.Provider.audio.PollinationsAudio import PollinationsAudio

            return PollinationsAudio
        elif name == "PollinationsImage":
            from g4f.Provider.PollinationsImage import PollinationsImage

            return PollinationsImage
        elif name == "Puter" or name == "PuterJS":
            from g4f.Provider.needs_auth.Puter import Puter

            return Puter
        elif name == "Qwen":
            from g4f.Provider.Qwen import Qwen

            return Qwen
        elif name == "QwenCode":
            from g4f.Provider.qwen.QwenCode import QwenCode

            return QwenCode
        elif name == "Reka":
            from g4f.Provider.needs_auth.Reka import Reka

            return Reka
        elif name == "Replicate":
            from g4f.Provider.extra.Replicate import Replicate

            return Replicate
        elif name == "SearXNG":
            from g4f.Provider.search.SearXNG import SearXNG

            return SearXNG
        elif name == "StabilityAI_SD35Large":
            from g4f.Provider.hf_space.StabilityAI_SD35Large import StabilityAI_SD35Large

            return StabilityAI_SD35Large
        elif name == "TeachAnything":
            from g4f.Provider.TeachAnything import TeachAnything

            return TeachAnything
        elif name == "ThebApi":
            from g4f.Provider.extra.ThebApi import ThebApi

            return ThebApi
        elif name == "Together":
            from g4f.Provider.extra.Together import Together

            return Together
        elif name == "WhiteRabbitNeo":
            from g4f.Provider.needs_auth.WhiteRabbitNeo import WhiteRabbitNeo

            return WhiteRabbitNeo
        elif name == "You":
            from g4f.Provider.needs_auth.You import You

            return You
        elif name == "YouTube":
            from g4f.Provider.search.YouTube import YouTube

            return YouTube
        elif name == "Yqcloud":
            from g4f.Provider.Yqcloud import Yqcloud

            return Yqcloud
        elif name == "gTTS":
            from g4f.Provider.audio.gTTS import gTTS

            return gTTS
        elif name == "xAI":
            from g4f.Provider.extra.xAI import xAI

            return xAI
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
    if name in globals().keys():
        return globals()[name]
    try:
        return ProviderLoader.from_name(name)
    except ImportError as e:
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
