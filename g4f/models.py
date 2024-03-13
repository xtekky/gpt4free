from __future__  import annotations

from dataclasses import dataclass

from .Provider import RetryProvider, ProviderType
from .Provider   import (
    Chatgpt4Online,
    PerplexityLabs,
    GeminiProChat,
    ChatgptNext,
    HuggingChat,
    HuggingFace,
    ChatgptDemo,
    GptForLove,
    ChatgptAi,
    DeepInfra,
    ChatBase,
    GigaChat,
    Liaobots,
    FreeGpt,
    Llama2,
    Vercel,
    Gemini,
    GptGo,
    Gpt6,
    Bing,
    You,
    Pi,
)

@dataclass(unsafe_hash=True)
class Model:
    """
    Represents a machine learning model configuration.

    Attributes:
        name (str): Name of the model.
        base_provider (str): Default provider for the model.
        best_provider (ProviderType): The preferred provider for the model, typically with retry logic.
    """
    name: str
    base_provider: str
    best_provider: ProviderType = None
    
    @staticmethod
    def __all__() -> list[str]:
        """Returns a list of all model names."""
        return _all_models

default = Model(
    name          = "",
    base_provider = "",
    best_provider = RetryProvider([
        Bing,
        ChatgptAi, GptGo,
        You,
        Chatgpt4Online
    ])
)

# GPT-3.5 too, but all providers supports long requests and responses
gpt_35_long = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = RetryProvider([
        FreeGpt, You,
        Chatgpt4Online,
        ChatgptNext,
        ChatgptDemo,
        Gpt6,
    ])
)

# GPT-3.5 / GPT-4
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = RetryProvider([ 
        GptGo, You,
        GptForLove, ChatBase,
        Chatgpt4Online,
    ])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'openai',
    best_provider = RetryProvider([
        Bing, Liaobots, 
    ])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'openai',
    best_provider = Bing
)

gigachat = Model(
    name          = 'GigaChat:latest',
    base_provider = 'gigachat',
    best_provider = GigaChat
)

gigachat_plus = Model(
    name          = 'GigaChat-Plus',
    base_provider = 'gigachat',
    best_provider = GigaChat
)

gigachat_pro = Model(
    name          = 'GigaChat-Pro',
    base_provider = 'gigachat',
    best_provider = GigaChat
)

llama2_7b = Model(
    name          = "meta-llama/Llama-2-7b-chat-hf",
    base_provider = 'meta',
    best_provider = RetryProvider([Llama2, DeepInfra])
)

llama2_13b = Model(
    name          = "meta-llama/Llama-2-13b-chat-hf",
    base_provider = 'meta',
    best_provider = RetryProvider([Llama2, DeepInfra])
)

llama2_70b = Model(
    name          = "meta-llama/Llama-2-70b-chat-hf",
    base_provider = "meta",
    best_provider = RetryProvider([Llama2, DeepInfra, HuggingChat])
)

codellama_34b_instruct = Model(
    name          = "codellama/CodeLlama-34b-Instruct-hf",
    base_provider = "meta",
    best_provider = RetryProvider([HuggingChat, DeepInfra])
)

codellama_70b_instruct = Model(
    name          = "codellama/CodeLlama-70b-Instruct-hf",
    base_provider = "meta",
    best_provider = RetryProvider([DeepInfra, PerplexityLabs])
)

# Mistral
mixtral_8x7b = Model(
    name          = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat, HuggingFace, PerplexityLabs])
)

mistral_7b = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat, HuggingFace, PerplexityLabs])
)

# Misc models
dolphin_mixtral_8x7b = Model(
    name          = "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    base_provider = "huggingface",
    best_provider = DeepInfra
)

lzlv_70b = Model(
    name          = "lizpreciatior/lzlv_70b_fp16_hf",
    base_provider = "huggingface",
    best_provider = DeepInfra
)

airoboros_70b = Model(
    name          = "deepinfra/airoboros-70b",
    base_provider = "huggingface",
    best_provider = DeepInfra
)

airoboros_l2_70b = Model(
    name          = "jondurbin/airoboros-l2-70b-gpt4-1.4.1",
    base_provider = "huggingface",
    best_provider = DeepInfra
)

openchat_35 = Model(
    name          = "openchat/openchat_3.5",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat])
)

# Bard
gemini = bard = palm = Model(
    name          = 'gemini',
    base_provider = 'google',
    best_provider = Gemini
)

claude_v2 = Model(
    name          = 'claude-v2',
    base_provider = 'anthropic',
    best_provider = RetryProvider([Vercel])
)

claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'anthropic',
    best_provider = You
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'anthropic',
    best_provider = You
)

gpt_35_turbo_16k = Model(
    name          = 'gpt-3.5-turbo-16k',
    base_provider = 'openai',
    best_provider = gpt_35_long.best_provider
)

gpt_35_turbo_16k_0613 = Model(
    name          = 'gpt-3.5-turbo-16k-0613',
    base_provider = 'openai',
    best_provider = gpt_35_long.best_provider
)

gpt_35_turbo_0613 = Model(
    name          = 'gpt-3.5-turbo-0613',
    base_provider = 'openai',
    best_provider = gpt_35_turbo.best_provider
)

gpt_4_0613 = Model(
    name          = 'gpt-4-0613',
    base_provider = 'openai',
    best_provider = gpt_4.best_provider
)

gpt_4_32k = Model(
    name          = 'gpt-4-32k',
    base_provider = 'openai',
    best_provider = gpt_4.best_provider
)

gpt_4_32k_0613 = Model(
    name          = 'gpt-4-32k-0613',
    base_provider = 'openai',
    best_provider = gpt_4.best_provider
)

gemini_pro = Model(
    name          = 'gemini-pro',
    base_provider = 'google',
    best_provider = RetryProvider([GeminiProChat, You])
)

pi = Model(
    name = 'pi',
    base_provider = 'inflection',
    best_provider = Pi
)

class ModelUtils:
    """
    Utility class for mapping string identifiers to Model instances.

    Attributes:
        convert (dict[str, Model]): Dictionary mapping model string identifiers to Model instances.
    """
    convert: dict[str, Model] = {
        # gpt-3.5
        'gpt-3.5-turbo'          : gpt_35_turbo,
        'gpt-3.5-turbo-0613'     : gpt_35_turbo_0613,
        'gpt-3.5-turbo-16k'      : gpt_35_turbo_16k,
        'gpt-3.5-turbo-16k-0613' : gpt_35_turbo_16k_0613,
        
        'gpt-3.5-long': gpt_35_long,
        
        # gpt-4
        'gpt-4'          : gpt_4,
        'gpt-4-0613'     : gpt_4_0613,
        'gpt-4-32k'      : gpt_4_32k,
        'gpt-4-32k-0613' : gpt_4_32k_0613,
        'gpt-4-turbo'    : gpt_4_turbo,

        # Llama 2
        'llama2-7b' : llama2_7b,
        'llama2-13b': llama2_13b,
        'llama2-70b': llama2_70b,
        'codellama-34b-instruct': codellama_34b_instruct,
        'codellama-70b-instruct': codellama_70b_instruct,

        # GigaChat
        'gigachat'     : gigachat,
        'gigachat_plus': gigachat_plus,
        'gigachat_pro' : gigachat_pro,

        'mixtral-8x7b': mixtral_8x7b,
        'mistral-7b': mistral_7b,
        'dolphin-mixtral-8x7b': dolphin_mixtral_8x7b,
        'lzlv-70b': lzlv_70b,
        'airoboros-70b': airoboros_70b,
        'airoboros-l2-70b': airoboros_l2_70b,
        'openchat_3.5': openchat_35,
        'gemini': gemini,
        'gemini-pro': gemini_pro,
        'claude-v2': claude_v2,
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'pi': pi
    }

_all_models = list(ModelUtils.convert.keys())
