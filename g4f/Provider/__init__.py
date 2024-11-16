from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

from .deprecated       import *
from .selenium         import *
from .needs_auth       import *
from .not_working      import *
from .local            import *

from .AIUncensored     import AIUncensored
from .Airforce         import Airforce
from .AmigoChat        import AmigoChat
from .Bing             import Bing
from .Blackbox         import Blackbox
from .ChatGpt          import ChatGpt
from .ChatGptEs        import ChatGptEs
from .Cloudflare       import Cloudflare
from .DarkAI           import DarkAI
from .DDG              import DDG
from .DeepInfraChat    import DeepInfraChat
from .Free2GPT         import Free2GPT
from .FreeGpt          import FreeGpt
from .GizAI            import GizAI
from .HuggingChat      import HuggingChat
from .Liaobots         import Liaobots
from .MagickPen        import MagickPen
from .PerplexityLabs   import PerplexityLabs
from .Pi               import Pi
from .Pizzagpt         import Pizzagpt
from .Prodia           import Prodia
from .Reka             import Reka
from .ReplicateHome    import ReplicateHome
from .RubiksAI         import RubiksAI
from .TeachAnything    import TeachAnything
from .Upstage          import Upstage
from .You              import You
from .Mhystical       import Mhystical

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
__map__: dict[str, ProviderType] = dict([
    (provider.__name__, provider) for provider in __providers__
])

class ProviderUtils:
    convert: dict[str, ProviderType] = __map__
