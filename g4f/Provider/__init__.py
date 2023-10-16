from __future__       import annotations
from .Acytoo          import Acytoo
from .AiAsk           import AiAsk
from .Aibn            import Aibn
from .Aichat          import Aichat
from .Ails            import Ails
from .AItianhu        import AItianhu
from .AItianhuSpace   import AItianhuSpace
from .Bing            import Bing
from .ChatBase        import ChatBase
from .ChatForAi       import ChatForAi
from .Chatgpt4Online  import Chatgpt4Online
from .ChatgptAi       import ChatgptAi
from .ChatgptDemo     import ChatgptDemo
from .ChatgptDuo      import ChatgptDuo
from .ChatgptFree     import ChatgptFree
from .ChatgptLogin    import ChatgptLogin
from .ChatgptX        import ChatgptX
from .Cromicle        import Cromicle
from .FreeGpt         import FreeGpt
from .GPTalk          import GPTalk
from .GptChatly       import GptChatly
from .GptForLove      import GptForLove
from .GptGo           import GptGo
from .GptGod          import GptGod
from .Liaobots        import Liaobots
from .Llama2          import Llama2
from .Myshell         import Myshell
from .NoowAi          import NoowAi
from .Opchatgpts      import Opchatgpts
from .Phind           import Phind
from .Vercel          import Vercel
from .Ylokh           import Ylokh
from .You             import You
from .Yqcloud         import Yqcloud
from .Geekgpt         import GeekGpt

from .base_provider  import BaseProvider, AsyncProvider, AsyncGeneratorProvider
from .retry_provider import RetryProvider
from .deprecated     import *
from .needs_auth     import *
from .unfinished     import *

class ProviderUtils:
    convert: dict[str, BaseProvider] = {
        'AItianhu': AItianhu,
        'AItianhuSpace': AItianhuSpace,
        'Acytoo': Acytoo,
        'AiAsk': AiAsk,
        'AiService': AiService,
        'Aibn': Aibn,
        'Aichat': Aichat,
        'Ails': Ails,
        'Aivvm': Aivvm,
        'AsyncGeneratorProvider': AsyncGeneratorProvider,
        'AsyncProvider': AsyncProvider,
        'Bard': Bard,
        'BaseProvider': BaseProvider,
        'Bing': Bing,
        'ChatBase': ChatBase,
        'ChatForAi': ChatForAi,
        'Chatgpt4Online': Chatgpt4Online,
        'ChatgptAi': ChatgptAi,
        'ChatgptDemo': ChatgptDemo,
        'ChatgptDuo': ChatgptDuo,
        'ChatgptFree': ChatgptFree,
        'ChatgptLogin': ChatgptLogin,
        'ChatgptX': ChatgptX,
        'CodeLinkAva': CodeLinkAva,
        'Cromicle': Cromicle,
        'DfeHub': DfeHub,
        'EasyChat': EasyChat,
        'Equing': Equing,
        'FastGpt': FastGpt,
        'Forefront': Forefront,
        'FreeGpt': FreeGpt,
        'GPTalk': GPTalk,
        'GptChatly': GptChatly,
        'GetGpt': GetGpt,
        'GptForLove': GptForLove,
        'GptGo': GptGo,
        'GptGod': GptGod,
        'H2o': H2o,
        'HuggingChat': HuggingChat,
        'Komo': Komo,
        'Liaobots': Liaobots,
        'Llama2': Llama2,
        'Lockchat': Lockchat,
        'MikuChat': MikuChat,
        'Myshell': Myshell,
        'NoowAi': NoowAi,
        'Opchatgpts': Opchatgpts,
        'OpenAssistant': OpenAssistant,
        'OpenaiChat': OpenaiChat,
        'PerplexityAi': PerplexityAi,
        'Phind': Phind,
        'Raycast': Raycast,
        'Theb': Theb,
        'V50': V50,
        'Vercel': Vercel,
        'Vitalentum': Vitalentum,
        'Wewordle': Wewordle,
        'Wuguokai': Wuguokai,
        'Ylokh': Ylokh,
        'You': You,
        'Yqcloud': Yqcloud,
        'Geekgpt': GeekGpt,
        
        'BaseProvider': BaseProvider,
        'AsyncProvider': AsyncProvider,
        'AsyncGeneratorProvider': AsyncGeneratorProvider,
        'RetryProvider': RetryProvider,
    }

__all__ = [
    'BaseProvider',
    'AsyncProvider',
    'AsyncGeneratorProvider',
    'RetryProvider',
    'Acytoo',
    'AiAsk',
    'Aibn',
    'Aichat',
    'Ails',
    'Aivvm',
    'AiService',
    'AItianhu',
    'AItianhuSpace',
    'Aivvm',
    'Bard',
    'Bing',
    'ChatBase',
    'ChatForAi',
    'Chatgpt4Online',
    'ChatgptAi',
    'ChatgptDemo',
    'ChatgptDuo',
    'ChatgptFree',
    'ChatgptLogin',
    'ChatgptX',
    'Cromicle',
    'CodeLinkAva',
    'DfeHub',
    'EasyChat',
    'Forefront',
    'FreeGpt',
    'GPTalk',
    'GptChatly',
    'GptForLove',
    'GetGpt',
    'GptGo',
    'GptGod',
    'H2o',
    'HuggingChat',
    'Liaobots',
    'Llama2',
    'Lockchat',
    'Myshell',
    'NoowAi',
    'Opchatgpts',
    'Raycast',
    'OpenaiChat',
    'OpenAssistant',
    'PerplexityAi',
    'Phind',
    'Theb',
    'Vercel',
    'Vitalentum',
    'Wewordle',
    'Ylokh',
    'You',
    'Yqcloud',
    'Equing',
    'FastGpt',
    'Wuguokai',
    'V50',
    'GeekGpt'
]
