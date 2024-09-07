from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

from .deprecated      import *
from .selenium        import *
from .needs_auth      import *

from .AI365VIP         import AI365VIP
from .Allyfy           import Allyfy
from .AiChatOnline     import AiChatOnline
from .AiChats          import AiChats
from .Aura             import Aura
from .Bing             import Bing
from .BingCreateImages import BingCreateImages
from .Binjie           import Binjie
from .Bixin123         import Bixin123
from .Blackbox         import Blackbox
from .ChatGot          import ChatGot
from .Chatgpt4Online   import Chatgpt4Online
from .Chatgpt4o        import Chatgpt4o
from .ChatgptFree      import ChatgptFree
from .CodeNews         import CodeNews
from .DDG              import DDG
from .DeepInfra        import DeepInfra
from .DeepInfraImage   import DeepInfraImage
from .FlowGpt          import FlowGpt
from .FluxAirforce     import FluxAirforce
from .Free2GPT         import Free2GPT
from .FreeChatgpt      import FreeChatgpt
from .FreeGpt          import FreeGpt
from .FreeNetfly       import FreeNetfly
from .GeminiPro        import GeminiPro
from .GigaChat         import GigaChat
from .GptTalkRu        import GptTalkRu
from .HuggingChat      import HuggingChat
from .HuggingFace      import HuggingFace
from .Koala            import Koala
from .Liaobots         import Liaobots
from .LiteIcoding      import LiteIcoding
from .Llama            import Llama
from .Local            import Local
from .MagickPen        import MagickPen
from .MetaAI           import MetaAI
from .MetaAIAccount    import MetaAIAccount
from .Nexra            import Nexra
from .Ollama           import Ollama
from .PerplexityLabs   import PerplexityLabs
from .Pi               import Pi
from .Pizzagpt         import Pizzagpt
from .Reka             import Reka
from .Snova            import Snova
from .Replicate        import Replicate
from .ReplicateHome    import ReplicateHome
from .Rocks            import Rocks
from .TeachAnything    import TeachAnything
from .TwitterBio       import TwitterBio
from .Upstage          import Upstage
from .Vercel           import Vercel
from .WhiteRabbitNeo   import WhiteRabbitNeo
from .You              import You

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
