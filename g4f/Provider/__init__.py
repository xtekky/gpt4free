from __future__ import annotations

from ..base_provider  import BaseProvider, ProviderType
from .retry_provider  import RetryProvider
from .base_provider   import AsyncProvider, AsyncGeneratorProvider
from .create_images   import CreateImagesProvider
from .deprecated      import *
from .selenium        import *
from .needs_auth      import *
from .unfinished      import *

from .AiAsk           import AiAsk
from .AiChatOnline    import AiChatOnline
from .AItianhu        import AItianhu
from .Aura            import Aura
from .Bestim          import Bestim
from .Bing            import Bing
from .ChatAnywhere    import ChatAnywhere
from .ChatBase        import ChatBase
from .ChatForAi       import ChatForAi
from .Chatgpt4Online  import Chatgpt4Online
from .ChatgptAi       import ChatgptAi
from .ChatgptDemo     import ChatgptDemo
from .ChatgptDemoAi   import ChatgptDemoAi
from .ChatgptFree     import ChatgptFree
from .ChatgptLogin    import ChatgptLogin
from .ChatgptNext     import ChatgptNext
from .ChatgptX        import ChatgptX
from .Chatxyz         import Chatxyz
from .DeepInfra       import DeepInfra
from .FakeGpt         import FakeGpt
from .FreeChatgpt     import FreeChatgpt
from .FreeGpt         import FreeGpt
from .GeekGpt         import GeekGpt
from .GeminiProChat   import GeminiProChat
from .Gpt6            import Gpt6
from .GPTalk          import GPTalk
from .GptChatly       import GptChatly
from .GptForLove      import GptForLove
from .GptGo           import GptGo
from .GptGod          import GptGod
from .GptTalkRu       import GptTalkRu
from .Hashnode        import Hashnode
from .HuggingChat     import HuggingChat
from .Koala           import Koala
from .Liaobots        import Liaobots
from .Llama2          import Llama2
from .OnlineGpt       import OnlineGpt
from .PerplexityLabs  import PerplexityLabs
from .Phind           import Phind
from .Pi              import Pi
from .Vercel          import Vercel
from .Ylokh           import Ylokh
from .You             import You

from .BingCreateImages import BingCreateImages

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