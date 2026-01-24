from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider, RotatedProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider
from .. import debug

from .needs_auth import *
from .needs_auth.hf import HuggingFace, HuggingChat, HuggingFaceAPI, HuggingFaceInference, HuggingFaceMedia
try:
    from .needs_auth.mini_max import HailuoAI, MiniMax
except ImportError as e:
    debug.error("MiniMax providers not loaded:", e)
try:
    from .local import *
except ImportError as e:
    debug.error("Local providers not loaded:", e)
try:
    from .hf_space import *
except ImportError as e:
    debug.error("HuggingFace Space providers not loaded:", e)
try:
    from .audio import *
except ImportError as e:
    debug.error("Audio providers not loaded:", e)
try:
    from .search import *
except ImportError as e:
    debug.error("Search providers not loaded:", e)

from .template import OpenaiTemplate, BackendApi
from .qwen.QwenCode import QwenCode

from .ApiAirforce          import ApiAirforce
from .Chatai               import Chatai
from .Cloudflare           import Cloudflare
from .Copilot              import Copilot
from .CopilotSession       import CopilotSession
from .DeepInfra            import DeepInfra
from .EasyChat             import EasyChat
from .GLM                  import GLM
from .GradientNetwork      import GradientNetwork
from .ItalyGPT             import ItalyGPT
from .LambdaChat           import LambdaChat
from .Mintlify             import Mintlify
from .OIVSCodeSer          import OIVSCodeSer2, OIVSCodeSer0501
from .OperaAria            import OperaAria
from .Perplexity           import Perplexity
from .PollinationsAI       import PollinationsAI
from .PollinationsImage    import PollinationsImage
from .Qwen                 import Qwen
from .TeachAnything        import TeachAnything
from .WeWordle             import WeWordle
from .Yqcloud              import Yqcloud
from .Yupp                 import Yupp

import sys

__modules__: list = [
    getattr(sys.modules[__name__], provider) for provider in dir()
    if not provider.startswith("__")
]
__providers__: list[ProviderType] = [
    provider for provider in __modules__
    if isinstance(provider, type)
    and issubclass(provider, BaseProvider)
]
__all__: list[str] = [
    provider.__name__ for provider in __providers__
]
__map__: dict[str, ProviderType] = {
    provider.__name__: provider for provider in __providers__
}

class ProviderUtils:
    convert: dict[str, ProviderType] = __map__
