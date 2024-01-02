from __future__  import annotations
from dataclasses import dataclass
from .Provider   import RetryProvider, ProviderType
from .Provider   import (
    Chatgpt4Online,
    ChatgptDemoAi,
    ChatgptNext,
    HuggingChat,
    ChatgptDemo,
    GptForLove,
    ChatgptAi,
    DeepInfra,
    OnlineGpt,
    ChatBase,
    Liaobots,
    GeekGpt,
    FakeGpt,
    FreeGpt,
    Berlin,
    Llama2,
    Vercel, 
    Phind,
    Koala,
    GptGo,
    Gpt6,
    Bard, 
    Bing,
    You,
    H2o,
    Pi,
)

@dataclass(unsafe_hash=True)
class Model:
    name: str
    base_provider: str
    best_provider: ProviderType = None
    
    @staticmethod
    def __all__() -> list[str]:
        return _all_models

default = Model(
    name          = "",
    base_provider = "",
    best_provider = RetryProvider([
        Bing,
        ChatgptAi, GptGo, GeekGpt,
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
        GeekGpt, FakeGpt,
        Berlin, Koala,
        Chatgpt4Online,
        ChatgptDemoAi,
        OnlineGpt,
        ChatgptNext,
        ChatgptDemo,
        Gpt6,
    ])
)

# GPT-3.5 / GPT-4
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider=RetryProvider([
        GptGo, You, 
        GptForLove, ChatBase,
        Chatgpt4Online,
    ])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'openai',
    best_provider = RetryProvider([
        Bing, Phind, Liaobots
    ])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'openai',
    best_provider = Bing
)

llama2_7b = Model(
    name          = "meta-llama/Llama-2-7b-chat-hf",
    base_provider = 'huggingface',
    best_provider = RetryProvider([Llama2, DeepInfra])
)

llama2_13b = Model(
    name          = "meta-llama/Llama-2-13b-chat-hf",
    base_provider = 'huggingface',
    best_provider = RetryProvider([Llama2, DeepInfra])
)

llama2_70b = Model(
    name          = "meta-llama/Llama-2-70b-chat-hf",
    base_provider = "huggingface",
    best_provider = RetryProvider([Llama2, DeepInfra, HuggingChat])
)

# Mistal
mixtral_8x7b = Model(
    name          = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat])
)

mistral_7b = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat])
)

openchat_35 = Model(
    name          = "openchat/openchat_3.5",
    base_provider = "huggingface",
    best_provider = RetryProvider([DeepInfra, HuggingChat])
)

# Bard
palm = Model(
    name          = 'palm',
    base_provider = 'google',
    best_provider = Bard)

# H2o
falcon_7b = Model(
    name          = 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3',
    base_provider = 'huggingface',
    best_provider = H2o)

falcon_40b = Model(
    name          = 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1',
    base_provider = 'huggingface',
    best_provider = H2o)

llama_13b = Model(
    name          = 'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b',
    base_provider = 'huggingface',
    best_provider = H2o)

# Vercel
claude_instant_v1 = Model(
    name          = 'claude-instant-v1',
    base_provider = 'anthropic',
    best_provider = Vercel)

claude_v1 = Model(
    name          = 'claude-v1',
    base_provider = 'anthropic',
    best_provider = Vercel)

claude_v2 = Model(
    name          = 'claude-v2',
    base_provider = 'anthropic',
    best_provider = Vercel)

command_light_nightly = Model(
    name          = 'command-light-nightly',
    base_provider = 'cohere',
    best_provider = Vercel)

command_nightly = Model(
    name          = 'command-nightly',
    base_provider = 'cohere',
    best_provider = Vercel)

gpt_neox_20b = Model(
    name          = 'EleutherAI/gpt-neox-20b',
    base_provider = 'huggingface',
    best_provider = Vercel)

oasst_sft_1_pythia_12b = Model(
    name          = 'OpenAssistant/oasst-sft-1-pythia-12b',
    base_provider = 'huggingface',
    best_provider = Vercel)

oasst_sft_4_pythia_12b_epoch_35 = Model(
    name          = 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
    base_provider = 'huggingface',
    best_provider = Vercel)

santacoder = Model(
    name          = 'bigcode/santacoder',
    base_provider = 'huggingface',
    best_provider = Vercel)

bloom = Model(
    name          = 'bigscience/bloom',
    base_provider = 'huggingface',
    best_provider = Vercel)

flan_t5_xxl = Model(
    name          = 'google/flan-t5-xxl',
    base_provider = 'huggingface',
    best_provider = Vercel)

code_davinci_002 = Model(
    name          = 'code-davinci-002',
    base_provider = 'openai',
    best_provider = Vercel)

gpt_35_turbo_16k = Model(
    name          = 'gpt-3.5-turbo-16k',
    base_provider = 'openai',
    best_provider = gpt_35_long.best_provider)

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

text_ada_001 = Model(
    name          = 'text-ada-001',
    base_provider = 'openai',
    best_provider = Vercel)

text_babbage_001 = Model(
    name          = 'text-babbage-001',
    base_provider = 'openai',
    best_provider = Vercel)

text_curie_001 = Model(
    name          = 'text-curie-001',
    base_provider = 'openai',
    best_provider = Vercel)

text_davinci_002 = Model(
    name          = 'text-davinci-002',
    base_provider = 'openai',
    best_provider = Vercel)

text_davinci_003 = Model(
    name          = 'text-davinci-003',
    base_provider = 'openai',
    best_provider = Vercel)

llama13b_v2_chat = Model(
    name          = 'replicate:a16z-infra/llama13b-v2-chat',
    base_provider = 'replicate',
    best_provider = Vercel)

llama7b_v2_chat = Model(
    name          = 'replicate:a16z-infra/llama7b-v2-chat',
    base_provider = 'replicate',
    best_provider = Vercel)

llama70b_v2_chat = Model(
    name          = 'replicate/llama70b-v2-chat',
    base_provider = 'replicate',
    best_provider = Vercel)

pi = Model(
    name = 'pi',
    base_provider = 'inflection',
    best_provider=Pi
)

class ModelUtils:
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
        
        # Mistral
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-7b': mistral_7b,
        'openchat_3.5': openchat_35,
        
        # Bard
        'palm2'       : palm,
        'palm'        : palm,
        'google'      : palm,
        'google-bard' : palm,
        'google-palm' : palm,
        'bard'        : palm,
        
        # H2o
        'falcon-40b' : falcon_40b,
        'falcon-7b'  : falcon_7b,
        'llama-13b'  : llama_13b,
        
        # Vercel
        #'claude-instant-v1' : claude_instant_v1,
        #'claude-v1'         : claude_v1,
        #'claude-v2'         : claude_v2,
        'command-nightly'   : command_nightly,
        'gpt-neox-20b'      : gpt_neox_20b,
        'santacoder'        : santacoder,
        'bloom'             : bloom,
        'flan-t5-xxl'       : flan_t5_xxl,
        'code-davinci-002'  : code_davinci_002,
        'text-ada-001'      : text_ada_001,
        'text-babbage-001'  : text_babbage_001,
        'text-curie-001'    : text_curie_001,
        'text-davinci-002'  : text_davinci_002,
        'text-davinci-003'  : text_davinci_003,
        'llama70b-v2-chat'  : llama70b_v2_chat,
        'llama13b-v2-chat'  : llama13b_v2_chat,
        'llama7b-v2-chat'   : llama7b_v2_chat,
        
        'oasst-sft-1-pythia-12b'           : oasst_sft_1_pythia_12b,
        'oasst-sft-4-pythia-12b-epoch-3.5' : oasst_sft_4_pythia_12b_epoch_35,
        'command-light-nightly'            : command_light_nightly,

        'pi': pi
    }

_all_models = list(ModelUtils.convert.keys())
