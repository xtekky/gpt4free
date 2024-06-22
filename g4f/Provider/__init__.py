from __future__ import annotations

from ..providers.types          import BaseProvider, ProviderType
from ..providers.retry_provider import RetryProvider, IterListProvider
from ..providers.base_provider  import AsyncProvider, AsyncGeneratorProvider
from ..providers.create_images  import CreateImagesProvider

from .deprecated      import *
from .not_working     import *
from .selenium        import *
from .needs_auth      import *

from .Aichatos         import Aichatos
from .Aura             import Aura
from .Bing             import Bing
from .BingCreateImages import BingCreateImages
from .Blackbox         import Blackbox
from .ChatForAi        import ChatForAi
from .Chatgpt4Online   import Chatgpt4Online
from .ChatgptAi        import ChatgptAi
from .ChatgptFree      import ChatgptFree
from .ChatgptNext      import ChatgptNext
from .ChatgptX         import ChatgptX
from .Cnote            import Cnote
from .Cohere           import Cohere
from .DeepInfra        import DeepInfra
from .DeepInfraImage   import DeepInfraImage
from .Feedough         import Feedough
from .FlowGpt          import FlowGpt
from .FreeChatgpt      import FreeChatgpt
from .FreeGpt          import FreeGpt
from .GigaChat         import GigaChat
from .GeminiPro        import GeminiPro
from .GeminiProChat    import GeminiProChat
from .GptTalkRu        import GptTalkRu
from .HuggingChat      import HuggingChat
from .HuggingFace      import HuggingFace
from .Koala            import Koala
from .Liaobots         import Liaobots
from .Llama            import Llama
from .Local            import Local
from .MetaAI           import MetaAI
from .MetaAIAccount    import MetaAIAccount
from .Ollama           import Ollama
from .PerplexityLabs   import PerplexityLabs
from .Pi               import Pi
from .Pizzagpt         import Pizzagpt
from .Replicate        import Replicate
from .ReplicateImage   import ReplicateImage
from .Vercel           import Vercel
from .WhiteRabbitNeo   import WhiteRabbitNeo
from .You              import You
from .Reka             import Reka

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
