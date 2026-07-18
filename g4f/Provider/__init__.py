from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider, RotatedProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

def _resolve_provider(name: str) -> ProviderType:
    if name == "AnyProvider":
        from g4f.providers.any_provider import AnyProvider; return AnyProvider
    elif name == "AIBadgr":
        from g4f.Provider.needs_auth.AIBadgr import AIBadgr; return AIBadgr
    elif name == "Anthropic":
        from g4f.Provider.needs_auth.Anthropic import Anthropic; return Anthropic
    elif name == "Antigravity":
        from g4f.Provider.needs_auth.Antigravity import Antigravity; return Antigravity
    elif name == "Airforce" or name == "ApiAirforce":
        from g4f.Provider.needs_auth.Airforce import Airforce; return Airforce
    elif name == "BingCreateImages":
        from g4f.Provider.needs_auth.BingCreateImages import BingCreateImages; return BingCreateImages
    elif name == "BlackForestLabs_Flux1Dev":
        from g4f.Provider.hf_space.BlackForestLabs_Flux1Dev import BlackForestLabs_Flux1Dev; return BlackForestLabs_Flux1Dev
    elif name == "BlackForestLabs_Flux1KontextDev":
        from g4f.Provider.hf_space.BlackForestLabs_Flux1KontextDev import BlackForestLabs_Flux1KontextDev; return BlackForestLabs_Flux1KontextDev
    elif name == "BlackboxPro":
        from g4f.Provider.needs_auth.BlackboxPro import BlackboxPro; return BlackboxPro
    elif name == "CablyAI":
        from g4f.Provider.needs_auth.CablyAI import CablyAI; return CablyAI
    elif name == "CachedSearch":
        from g4f.Provider.search.CachedSearch import CachedSearch; return CachedSearch
    elif name == "Cerebras":
        from g4f.Provider.needs_auth.Cerebras import Cerebras; return Cerebras
    elif name == "Claude":
        from g4f.Provider.needs_auth.Claude import Claude; return Claude
    elif name == "Cloudflare":
        from g4f.Provider.Cloudflare import Cloudflare; return Cloudflare
    elif name == "Cohere":
        from g4f.Provider.needs_auth.Cohere import Cohere; return Cohere
    elif name == "CohereForAI_C4AI_Command":
        from g4f.Provider.hf_space.CohereForAI_C4AI_Command import CohereForAI_C4AI_Command; return CohereForAI_C4AI_Command
    elif name == "Copilot":
        from g4f.Provider.Copilot import Copilot; return Copilot
    elif name == "CopilotAccount":
        from g4f.Provider.needs_auth.CopilotAccount import CopilotAccount; return CopilotAccount
    elif name == "CopilotApp":
        from g4f.Provider.CopilotApp import CopilotApp; return CopilotApp
    elif name == "CopilotSession":
        from g4f.Provider.CopilotSession import CopilotSession; return CopilotSession
    elif name == "Custom":
        from g4f.Provider.needs_auth.Custom import Custom; return Custom
    elif name == "DeepInfra":
        from .DeepInfra import DeepInfra; return DeepInfra
    elif name == "DeepSeek" or name == "DeepSeekAPI":
        from g4f.Provider.needs_auth.DeepSeek import DeepSeek; return DeepSeek
    elif name == "EdgeTTS":
        from g4f.Provider.audio.EdgeTTS import EdgeTTS; return EdgeTTS
    elif name == "FenayAI":
        from g4f.Provider.needs_auth.FenayAI import FenayAI; return FenayAI
    elif name == "GLM":
        from g4f.Provider.glm import GLM; return GLM
    elif name == "Gemini":
        from g4f.Provider.needs_auth.Gemini import Gemini; return Gemini
    elif name == "GeminiCLI":
        from g4f.Provider.needs_auth.GeminiCLI import GeminiCLI; return GeminiCLI
    elif name == "GeminiPro":
        from g4f.Provider.needs_auth.GeminiPro import GeminiPro; return GeminiPro
    elif name == "GigaChat":
        from g4f.Provider.needs_auth.GigaChat import GigaChat; return GigaChat
    elif name == "GithubCopilot":
        from g4f.Provider.github.GithubCopilot import GithubCopilot; return GithubCopilot
    elif name == "GithubCopilotAPI":
        from g4f.Provider.needs_auth.GithubCopilotAPI import GithubCopilotAPI; return GithubCopilotAPI
    elif name == "GlhfChat":
        from g4f.Provider.needs_auth.GlhfChat import GlhfChat; return GlhfChat
    elif name == "GoogleSearch":
        from g4f.Provider.search.GoogleSearch import GoogleSearch; return GoogleSearch
    elif name == "Grok":
        from g4f.Provider.needs_auth.Grok import Grok; return Grok
    elif name == "Groq":
        from g4f.Provider.needs_auth.Groq import Groq; return Groq
    elif name == "HailuoAI":
        from g4f.Provider.needs_auth.mini_max.HailuoAI import HailuoAI; return HailuoAI
    elif name == "HuggingChat":
        from g4f.Provider.needs_auth.hf.HuggingChat import HuggingChat; return HuggingChat
    elif name == "HuggingFace":
        from g4f.Provider.needs_auth.hf import HuggingFace; return HuggingFace
    elif name == "HuggingFaceAPI":
        from g4f.Provider.needs_auth.hf.HuggingFaceAPI import HuggingFaceAPI; return HuggingFaceAPI
    elif name == "HuggingFaceInference":
        from g4f.Provider.needs_auth.hf.HuggingFaceInference import HuggingFaceInference; return HuggingFaceInference
    elif name == "HuggingFaceMedia":
        from g4f.Provider.needs_auth.hf.HuggingFaceMedia import HuggingFaceMedia; return HuggingFaceMedia
    elif name == "HuggingSpace":
        from g4f.Provider.hf_space import HuggingSpace; return HuggingSpace
    elif name == "LMArena":
        from g4f.Provider.needs_auth.LMArena import LMArena; return LMArena
    elif name == "Local":
        from g4f.Provider.local import Local; return Local
    elif name == "MarkItDown":
        from g4f.Provider.audio.MarkItDown import MarkItDown; return MarkItDown
    elif name == "MetaAI":
        from g4f.Provider.needs_auth.MetaAI import MetaAI; return MetaAI
    elif name == "MetaAIAccount":
        from g4f.Provider.needs_auth.MetaAIAccount import MetaAIAccount; return MetaAIAccount
    elif name == "MicrosoftDesigner":
        from g4f.Provider.needs_auth.MicrosoftDesigner import MicrosoftDesigner; return MicrosoftDesigner
    elif name == "MiniMax":
        from g4f.Provider.needs_auth.mini_max.MiniMax import MiniMax; return MiniMax
    elif name == "Nvidia":
        from g4f.Provider.needs_auth.Nvidia import Nvidia; return Nvidia
    elif name == "Ollama":
        from g4f.Provider.local.Ollama import Ollama; return Ollama
    elif name == "OpenAIFM":
        from g4f.Provider.audio.OpenAIFM import OpenAIFM; return OpenAIFM
    elif name == "OpenRouter":
        from g4f.Provider.needs_auth.OpenRouter import OpenRouter; return OpenRouter
    elif name == "OpenRouterFree":
        from g4f.Provider.needs_auth.OpenRouter import OpenRouter; return OpenRouter
    elif name == "OpenaiAPI":
        from g4f.Provider.needs_auth.OpenaiAPI import OpenaiAPI; return OpenaiAPI
    elif name == "OpenaiAccount":
        from g4f.Provider.needs_auth.OpenaiAccount import OpenaiAccount; return OpenaiAccount
    elif name == "OpenaiChat":
        from g4f.Provider.needs_auth.OpenaiChat import OpenaiChat; return OpenaiChat
    elif name == "OpenaiTemplate":
        from g4f.Provider.template.OpenaiTemplate import OpenaiTemplate; return OpenaiTemplate
    elif name == "OperaAria":
        from g4f.Provider.OperaAria import OperaAria; return OperaAria
    elif name == "Perplexity":
        from g4f.Provider.Perplexity import Perplexity; return Perplexity
    elif name == "PerplexityApi":
        from g4f.Provider.needs_auth.PerplexityApi import PerplexityApi; return PerplexityApi
    elif name == "PhindAi":
        from g4f.Provider.PhindAi import PhindAi; return PhindAi
    elif name == "Pi":
        from g4f.Provider.needs_auth.Pi import Pi; return Pi
    elif name == "Pollinations" or name == "PollinationsAI":
        from g4f.Provider.Pollinations import Pollinations; return Pollinations
    elif name == "PollinationsAudio":
        from g4f.Provider.audio.PollinationsAudio import PollinationsAudio; return PollinationsAudio
    elif name == "PollinationsImage":
        from g4f.Provider.PollinationsImage import PollinationsImage; return PollinationsImage
    elif name == "Puter" or name == "PuterJS":
        from g4f.Provider.needs_auth.Puter import Puter; return Puter
    elif name == "Qwen":
        from g4f.Provider.Qwen import Qwen; return Qwen
    elif name == "QwenCode":
        from g4f.Provider.qwen.QwenCode import QwenCode; return QwenCode
    elif name == "Reka":
        from g4f.Provider.needs_auth.Reka import Reka; return Reka
    elif name == "Replicate":
        from g4f.Provider.needs_auth.Replicate import Replicate; return Replicate
    elif name == "SearXNG":
        from g4f.Provider.search.SearXNG import SearXNG; return SearXNG
    elif name == "StabilityAI_SD35Large":
        from g4f.Provider.hf_space.StabilityAI_SD35Large import StabilityAI_SD35Large; return StabilityAI_SD35Large
    elif name == "TeachAnything":
        from g4f.Provider.TeachAnything import TeachAnything; return TeachAnything
    elif name == "ThebApi":
        from g4f.Provider.needs_auth.ThebApi import ThebApi; return ThebApi
    elif name == "Together":
        from g4f.Provider.needs_auth.Together import Together; return Together
    elif name == "Video":
        from g4f.Provider.needs_auth.Video import Video; return Video
    elif name == "WeWordle":
        from g4f.Provider.WeWordle import WeWordle; return WeWordle
    elif name == "WhiteRabbitNeo":
        from g4f.Provider.needs_auth.WhiteRabbitNeo import WhiteRabbitNeo; return WhiteRabbitNeo
    elif name == "You":
        from g4f.Provider.needs_auth.You import You; return You
    elif name == "YouTube":
        from g4f.Provider.search.YouTube import YouTube; return YouTube
    elif name == "Yqcloud":
        from g4f.Provider.Yqcloud import Yqcloud; return Yqcloud
    elif name == "gTTS":
        from g4f.Provider.audio.gTTS import gTTS; return gTTS
    elif name == "xAI":
        from g4f.Provider.needs_auth.xAI import xAI; return xAI
    else:
        raise ImportError(f"Provider '{name}' not found")

_provider_names = [
    "AnyProvider",
    "AIBadgr",
    "Anthropic",
    "Antigravity",
    "ApiAirforce",
    "BingCreateImages",
    "BlackForestLabs_Flux1Dev",
    "BlackForestLabs_Flux1KontextDev",
    "BlackboxPro",
    "CablyAI",
    "CachedSearch",
    "Cerebras",
    "Claude",
    "Cloudflare",
    "Cohere",
    "CohereForAI_C4AI_Command",
    "Copilot",
    "CopilotAccount",
    "CopilotApp",
    "CopilotSession",
    "Custom",
    "DeepInfra",
    "DeepSeek",
    "EasyChat",
    "EdgeTTS",
    "Felo",
    "FenayAI",
    "GLM",
    "Gemini",
    "GeminiCLI",
    "GeminiPro",
    "GigaChat",
    "GithubCopilot",
    "GithubCopilotAPI",
    "GlhfChat",
    "GoogleSearch",

    "GradientNetwork",
    "Grok",
    "Groq",
    "HailuoAI",
    "HuggingChat",
    "HuggingFace",
    "HuggingFaceAPI",
    "HuggingFaceInference",
    "HuggingFaceMedia",
    "HuggingSpace",
    "LMArena",
    "Local",
    "MarkItDown",
    "MetaAI",
    "MetaAIAccount",
    "MicrosoftDesigner",
    "Miklium",
    "MiniMax",
    "Nvidia",
    "Ollama",
    "OllamaSwarm",
    "OpenAIFM",
    "OpenRouter",
    "OpenRouterFree",
    "OpenaiAPI",
    "OpenaiAccount",
    "OpenaiChat",
    "OpenaiTemplate",
    "OperaAria",
    "Perchance",
    "Perplexity",
    "PerplexityApi",
    "PhindAi",
    "Pi",
    "PollinationsAI",
    "PollinationsAudio",
    "PollinationsImage",
    "PuterJS",
    "Qwen",
    "QwenCode",
    "Reka",
    "Replicate",
    "SearXNG",
    "StabilityAI_SD35Large",
    "Surfsense",
    "TeachAnything",
    "ThebApi",
    "Together",
    "Video",
    "WeWordle",
    "WhiteRabbitNeo",
    "You",
    "YouTube",
    "Yqcloud",
    "gTTS",
    "xAI",
]

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
] + _provider_names

_loaded_providers = {}

def __getattr__(name: str):
    if name in _loaded_providers:
        return _loaded_providers[name]
    if name == "__providers__":
        # Load all providers if specifically requested
        providers_list = []
        for provider_name in _provider_names:
            try:
                providers_list.append(__getattr__(provider_name))
            except AttributeError:
                pass
        return providers_list
    try:
        return _resolve_provider(name)
    except ImportError as e:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from e

def __dir__():
    return __all__

class _ConvertDict(dict):
    def __contains__(self, item):
        return item in _provider_names
    def __getitem__(self, item):
        try:
            return __getattr__(item)
        except AttributeError:
            raise KeyError(f"Provider '{item}' not found")
    def keys(self):
        return _provider_names
    def items(self):
        return [(k, self[k]) for k in _provider_names]
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
            return __getattr__(label)
        except AttributeError:
            pass
            
        # Fallback to search
        for provider_name in _provider_names:
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
        
        try:
            return __getattr__(name)
        except AttributeError:
            pass
        
        return super().__getattribute__(name)

sys.modules[__name__].__class__ = LazyProviderModule

