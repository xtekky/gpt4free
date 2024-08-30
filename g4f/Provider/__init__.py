from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

from .deprecated      import *
from .not_working     import *
from .selenium        import *
from .needs_auth      import *

from .AI365VIP         import AI365VIP
from .Allyfy           import Allyfy
from .AiChatOnline     import AiChatOnline
from .Aura             import Aura
from .Bing             import Bing
from .BingCreateImages import BingCreateImages
from .Blackbox         import Blackbox
from .ChatGot          import ChatGot
from .Chatgpt4o        import Chatgpt4o
from .Chatgpt4Online   import Chatgpt4Online
from .ChatgptFree      import ChatgptFree
from .Cohere           import Cohere
from .DDG              import DDG
from .DeepInfra        import DeepInfra
from .DeepInfraImage   import DeepInfraImage
from .FlowGpt          import FlowGpt
from .FreeChatgpt      import FreeChatgpt
from .FreeGpt          import FreeGpt
from .FreeNetfly       import FreeNetfly
from .GeminiPro        import GeminiPro
from .GeminiProChat    import GeminiProChat
from .GigaChat         import GigaChat
from .GptTalkRu        import GptTalkRu
from .HuggingChat      import HuggingChat
from .HuggingFace      import HuggingFace
from .HuggingFace      import HuggingFace
from .Koala            import Koala
from .Liaobots         import Liaobots
from .LiteIcoding      import LiteIcoding
from .Llama            import Llama
from .Local            import Local
from .MagickPenAsk     import MagickPenAsk
from .MagickPenChat    import MagickPenChat
from .Marsyoo          import Marsyoo
from .MetaAI           import MetaAI
from .MetaAIAccount    import MetaAIAccount
from .Ollama           import Ollama
from .PerplexityLabs   import PerplexityLabs
from .Pi               import Pi
from .Pizzagpt         import Pizzagpt
from .Reka             import Reka
from .Replicate        import Replicate
from .ReplicateHome    import ReplicateHome
from .Rocks            import Rocks
from .TeachAnything    import TeachAnything
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
