from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

from .deprecated       import *
from .needs_auth       import *
from .not_working      import *
from .local            import *
from .hf               import HuggingFace, HuggingChat, HuggingFaceAPI, HuggingFaceInference
from .hf_space         import HuggingSpace
from .mini_max         import HailuoAI, MiniMax
from .template         import OpenaiTemplate, BackendApi

from .Blackbox             import Blackbox
from .CablyAI              import CablyAI
from .ChatGLM              import ChatGLM
from .ChatGpt              import ChatGpt
from .ChatGptEs            import ChatGptEs
from .ChatGptt             import ChatGptt
from .Cloudflare           import Cloudflare
from .Copilot              import Copilot
from .DarkAI               import DarkAI
from .DDG                  import DDG
from .DeepInfraChat        import DeepInfraChat
from .Free2GPT             import Free2GPT
from .FreeGpt              import FreeGpt
from .GizAI                import GizAI
from .Glider               import Glider
from .GPROChat             import GPROChat
from .ImageLabs            import ImageLabs
from .Jmuz                 import Jmuz
from .Liaobots             import Liaobots
from .Mhystical            import Mhystical
from .OIVSCode             import OIVSCode
from .PerplexityLabs       import PerplexityLabs
from .Pi                   import Pi
from .Pizzagpt             import Pizzagpt
from .PollinationsAI       import PollinationsAI
from .Prodia               import Prodia
from .TeachAnything        import TeachAnything
from .You                  import You
from .Yqcloud              import Yqcloud

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
__providers__ = __providers__ + HuggingSpace.providers
__all__: list[str] = [
    provider.__name__ for provider in __providers__
]
__map__: dict[str, ProviderType] = dict([
    (provider.__name__, provider) for provider in __providers__
])

class ProviderUtils:
    convert: dict[str, ProviderType] = __map__
