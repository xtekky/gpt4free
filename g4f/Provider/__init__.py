from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider
from .. import debug
try:
    from .deprecated import *
except ImportError as e:
    debug.error("Deprecated providers not loaded:", e)
from .needs_auth       import *
from .template         import OpenaiTemplate, BackendApi
from .hf               import HuggingFace, HuggingChat, HuggingFaceAPI, HuggingFaceInference, HuggingFaceMedia
try:
    from .not_working import *
except ImportError as e:
    debug.error("Not working providers not loaded:", e)
try:
    from .local import *
except ImportError as e:
    debug.error("Local providers not loaded:", e)
try:
    from .hf_space import *
except ImportError as e:
    debug.error("HuggingFace Space providers not loaded:", e)
try:
    from .mini_max import HailuoAI, MiniMax
except ImportError as e:
    debug.error("MiniMax providers not loaded:", e)

try:
    from .AllenAI              import AllenAI
    from .ARTA                 import ARTA
    from .Blackbox             import Blackbox
    from .Chatai               import Chatai
    from .ChatGLM              import ChatGLM
    from .ChatGpt              import ChatGpt
    from .ChatGptEs            import ChatGptEs
    from .Cloudflare           import Cloudflare
    from .Copilot              import Copilot
    from .DDG                  import DDG
    from .DeepInfraChat        import DeepInfraChat
    from .DuckDuckGo           import DuckDuckGo
    from .Dynaspark            import Dynaspark
except ImportError as e:
    debug.error("Providers not loaded (A-D):", e)
try:
    from .Free2GPT             import Free2GPT
    from .FreeGpt              import FreeGpt
    from .FreeRouter           import FreeRouter
    from .GizAI                import GizAI
    from .Glider               import Glider
    from .Goabror              import Goabror
    from .ImageLabs            import ImageLabs
    from .Jmuz                 import Jmuz
    from .LambdaChat           import LambdaChat
    from .Liaobots             import Liaobots
    from .LMArenaProvider      import LMArenaProvider
    from .OIVSCode             import OIVSCode
except ImportError as e:
    debug.error("Providers not loaded (F-L):", e)
try:
    from .PerplexityLabs       import PerplexityLabs
    from .Pi                   import Pi
    from .Pizzagpt             import Pizzagpt
    from .PollinationsAI       import PollinationsAI
    from .PollinationsImage    import PollinationsImage
    from .TeachAnything        import TeachAnything
    from .TypeGPT              import TypeGPT
    from .You                  import You
    from .Websim               import Websim
    from .Yqcloud              import Yqcloud
except ImportError as e:
    debug.error("Providers not loaded (M-Z):", e)

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
