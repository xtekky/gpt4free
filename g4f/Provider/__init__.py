from __future__ import annotations

from ..base_provider  import BaseProvider, ProviderType
from .retry_provider  import RetryProvider
from .base_provider   import AsyncProvider, AsyncGeneratorProvider
from .deprecated      import *
from .needs_auth      import *
from .unfinished      import *
from .selenium        import *

from .Aura            import Aura
from .AiAsk           import AiAsk
from .Aichat          import Aichat
from .AiChatOnline    import AiChatOnline
from .AItianhu        import AItianhu
from .AItianhuSpace   import AItianhuSpace
from .Berlin          import Berlin
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
from .Koala           import Koala
from .Liaobots        import Liaobots
from .Llama2          import Llama2
from .MyShell         import MyShell
from .OnlineGpt       import OnlineGpt
from .Opchatgpts      import Opchatgpts
from .PerplexityAi    import PerplexityAi
from .Phind           import Phind
from .Pi              import Pi
from .TalkAi          import TalkAi
from .Vercel          import Vercel
from .Ylokh           import Ylokh
from .You             import You
from .Yqcloud         import Yqcloud
from .Bestim          import Bestim

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