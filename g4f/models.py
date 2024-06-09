from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    Aichatos,
    Bing,
    Blackbox,
    ChatgptAi,
    ChatgptNext,
    Cnote,
    DeepInfra,
    Feedough,
    FreeGpt,
    Gemini,
    GeminiPro,
    GigaChat,
    HuggingChat,
    HuggingFace,
    Koala,
    Liaobots,
    MetaAI,
    OpenaiChat,
    PerplexityLabs,
    Replicate,
    Pi,
    Vercel,
    You,
    Reka
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
    best_provider = IterListProvider([
        Bing,
        ChatgptAi,
        You,
        OpenaiChat,
    ])
)

# GPT-3.5 too, but all providers supports long requests and responses
gpt_35_long = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = IterListProvider([
        FreeGpt,
        You,
        ChatgptNext,
        OpenaiChat,
        Koala,
    ])
)

# GPT-3.5 / GPT-4
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = IterListProvider([
        FreeGpt,
        You,
        ChatgptNext,
        Koala,
        OpenaiChat,
        Aichatos,
        Cnote,
        Feedough,
    ])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'openai',
    best_provider = IterListProvider([
        Bing, Liaobots, 
    ])
)

gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'openai',
    best_provider = IterListProvider([
        You, Liaobots
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

meta = Model(
    name          = "meta",
    base_provider = "meta",
    best_provider = MetaAI
)

llama3_8b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-8B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate])
)

llama3_70b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-70B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate])
)

codellama_34b_instruct = Model(
    name          = "codellama/CodeLlama-34b-Instruct-hf",
    base_provider = "meta",
    best_provider = HuggingChat
)

codellama_70b_instruct = Model(
    name          = "codellama/CodeLlama-70b-Instruct-hf",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs])
)

# Mistral
mixtral_8x7b = Model(
    name          = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, PerplexityLabs])
)

mistral_7b = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = IterListProvider([HuggingChat, HuggingFace, PerplexityLabs])
)

mistral_7b_v02 = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.2",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, PerplexityLabs])
)

# Bard
gemini = Model(
    name          = 'gemini',
    base_provider = 'google',
    best_provider = Gemini
)

claude_v2 = Model(
    name          = 'claude-v2',
    base_provider = 'anthropic',
    best_provider = IterListProvider([Vercel])
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

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'anthropic',
    best_provider = None
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
    best_provider = IterListProvider([GeminiPro, You])
)

pi = Model(
    name = 'pi',
    base_provider = 'inflection',
    best_provider = Pi
)

dbrx_instruct = Model(
    name = 'databricks/dbrx-instruct',
    base_provider = 'mistral',
    best_provider = IterListProvider([DeepInfra, PerplexityLabs])
)

command_r_plus = Model(
    name = 'CohereForAI/c4ai-command-r-plus',
    base_provider = 'mistral',
    best_provider = IterListProvider([HuggingChat])
)

blackbox = Model(
    name = 'blackbox',
    base_provider = 'blackbox',
    best_provider = Blackbox
)

reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
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
        'gpt-4o'         : gpt_4o,
        'gpt-4'          : gpt_4,
        'gpt-4-0613'     : gpt_4_0613,
        'gpt-4-32k'      : gpt_4_32k,
        'gpt-4-32k-0613' : gpt_4_32k_0613,
        'gpt-4-turbo'    : gpt_4_turbo,

        "meta-ai": meta,
        'llama3-8b': llama3_8b_instruct, # alias
        'llama3-70b': llama3_70b_instruct, # alias
        'llama3-8b-instruct' : llama3_8b_instruct,
        'llama3-70b-instruct': llama3_70b_instruct,

        'codellama-34b-instruct': codellama_34b_instruct,
        'codellama-70b-instruct': codellama_70b_instruct,

        # Mistral Opensource
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-7b': mistral_7b,
        'mistral-7b-v02': mistral_7b_v02,

        # google gemini
        'gemini': gemini,
        'gemini-pro': gemini_pro,

        # anthropic
        'claude-v2': claude_v2,
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'claude-3-haiku': claude_3_haiku,

        # reka core
        'reka': reka_core,

        # other
        'blackbox': blackbox,
        'command-r+': command_r_plus,
        'dbrx-instruct': dbrx_instruct,
        'gigachat': gigachat,
        'pi': pi
    }

_all_models = list(ModelUtils.convert.keys())
