from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AI365VIP,
    Allyfy,
    Bing,
    Blackbox,
    ChatGot,
    Chatgpt4o,
    Chatgpt4Online,
    ChatgptFree,
    DDG,
    DeepInfra,
    DeepInfraImage,
    FreeChatgpt,
    FreeGpt,
    FreeNetfly,
    Gemini,
    GeminiPro,
    GeminiProChat,
    GigaChat,
    HuggingChat,
    HuggingFace,
    Koala,
    Liaobots,
    LiteIcoding,
    MagickPenAsk,
    MagickPenChat,
    Marsyoo,
    MetaAI,
    OpenaiChat,
    PerplexityLabs,
    Pi,
    Pizzagpt,
    Reka,
    Replicate,
    ReplicateHome,
    TeachAnything,
    You,
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
        You,
        OpenaiChat,
        FreeChatgpt,
        AI365VIP,
        Chatgpt4o,
        DDG,
        ChatgptFree,
        Koala,
        Pizzagpt,
    ])
)

# GPT-3.5 too, but all providers supports long requests and responses
gpt_35_long = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = IterListProvider([
        FreeGpt,
        You,
        Koala,
        ChatgptFree,
        FreeChatgpt,
        DDG,
        AI365VIP,
        Pizzagpt,
        Allyfy,
    ])
)

############
### Text ###
############

### OpenAI ###
### GPT-3.5 / GPT-4 ###
# gpt-3.5
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'openai',
    best_provider = IterListProvider([
        FreeGpt,
        You,
        Koala,
        ChatgptFree,
        FreeChatgpt,
        AI365VIP,
        Pizzagpt,
        Allyfy,
    ])
)

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'openai',
    best_provider = IterListProvider([
        Bing, Chatgpt4Online
    ])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'openai',
    best_provider = IterListProvider([
        Bing, Liaobots, LiteIcoding
    ])
)
gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'openai',
    best_provider = IterListProvider([
        You, Liaobots, Chatgpt4o, AI365VIP, OpenaiChat, Marsyoo, LiteIcoding, MagickPenAsk,
    ])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'openai',
    best_provider = IterListProvider([
        DDG, Liaobots, OpenaiChat, You, FreeNetfly, MagickPenChat,
    ])
)


### GigaChat ###
gigachat = Model(
    name          = 'GigaChat:latest',
    base_provider = 'gigachat',
    best_provider = GigaChat
)


### Meta ###
meta = Model(
    name          = "meta",
    base_provider = "meta",
    best_provider = MetaAI
)

llama_3_8b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-8B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate])
)

llama_3_70b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-70B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate])
)

llama_3_70b_instruct = Model(
    name          = "meta/meta-llama-3-70b-instruct",
    base_provider = "meta",
    best_provider = IterListProvider([ReplicateHome, TeachAnything])
)

llama_3_70b_chat_hf = Model(
    name          = "meta-llama/Llama-3-70b-chat-hf",
    base_provider = "meta",
    best_provider = IterListProvider([DDG])
)

llama_3_1_70b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

llama_3_1_405b_instruct_FP8 = Model(
    name          = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    base_provider = "meta",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)


### Mistral ###
mixtral_8x7b = Model(
    name          = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, PerplexityLabs, HuggingChat, DDG, ReplicateHome])
)

mistral_7b_v02 = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.2",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, HuggingChat])
)


### NousResearch ###
Nous_Hermes_2_Mixtral_8x7B_DPO = Model(
    name          = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    base_provider = "NousResearch",
    best_provider = IterListProvider([HuggingFace, HuggingChat])
)


### 01-ai ###
Yi_1_5_34B_chat = Model(
    name          = "01-ai/Yi-1.5-34B-Chat",
    base_provider = "01-ai",
    best_provider = IterListProvider([HuggingFace, HuggingChat])
)


### Microsoft ###
Phi_3_mini_4k_instruct = Model(
    name          = "microsoft/Phi-3-mini-4k-instruct",
    base_provider = "Microsoft",
    best_provider = IterListProvider([HuggingFace, HuggingChat])
)


### Google ###
# gemini
gemini = Model(
    name          = 'gemini',
    base_provider = 'Google',
    best_provider = Gemini
)

gemini_pro = Model(
    name          = 'gemini-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([GeminiPro, You, ChatGot, GeminiProChat, Liaobots, LiteIcoding])
)

gemini_flash = Model(
    name          = 'gemini-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Liaobots])
)

gemini_1_5 = Model(
    name          = 'gemini-1.5',
    base_provider = 'Google',
    best_provider = IterListProvider([LiteIcoding])
)

# gemma
gemma_2b_it = Model(
    name          = 'gemma-2b-it',
    base_provider = 'Google',
    best_provider = IterListProvider([ReplicateHome])
)

gemma_2_9b_it = Model(
    name          = 'gemma-2-9b-it',
    base_provider = 'Google',
    best_provider = IterListProvider([PerplexityLabs])
)

gemma_2_27b_it = Model(
    name          = 'gemma-2-27b-it',
    base_provider = 'Google',
    best_provider = IterListProvider([PerplexityLabs])
)


### Anthropic ###
claude_2 = Model(
    name          = 'claude-2',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([You])
)

claude_2_0 = Model(
    name          = 'claude-2.0',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Liaobots])
)

claude_2_1 = Model(
    name          = 'claude-2.1',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Liaobots])
)

claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([You, Liaobots])
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([You, Liaobots])
)

claude_3_5_sonnet = Model(
    name          = 'claude-3-5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Liaobots])
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, AI365VIP, Liaobots])
)

claude_3 = Model(
    name          = 'claude-3',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([LiteIcoding])
)

claude_3_5 = Model(
    name          = 'claude-3.5',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([LiteIcoding])
)


### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
)


### NVIDIA ###
nemotron_4_340b_instruct = Model(
    name = 'nemotron-4-340b-instruct',
    base_provider = 'NVIDIA',
    best_provider = IterListProvider([PerplexityLabs])
)


### Blackbox ###
blackbox = Model(
    name = 'blackbox',
    base_provider = 'Blackbox',
    best_provider = Blackbox
)


### Databricks ###
dbrx_instruct = Model(
    name = 'databricks/dbrx-instruct',
    base_provider = 'Databricks',
    best_provider = IterListProvider([DeepInfra])
)


### CohereForAI ###
command_r_plus = Model(
    name = 'CohereForAI/c4ai-command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingChat])
)


### iFlytek ###
SparkDesk_v1_1 = Model(
    name = 'SparkDesk-v1.1',
    base_provider = 'iFlytek',
    best_provider = IterListProvider([FreeChatgpt])
)


### DeepSeek ###
deepseek_coder = Model(
    name = 'deepseek-coder',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([FreeChatgpt])
)

deepseek_chat = Model(
    name = 'deepseek-chat',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([FreeChatgpt])
)


### Qwen ###
Qwen2_7B_instruct = Model(
    name = 'Qwen2-7B-Instruct',
    base_provider = 'Qwen',
    best_provider = IterListProvider([FreeChatgpt])
)


### Zhipu AI ###
glm4_9B_chat = Model(
    name = 'glm4-9B-chat',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([FreeChatgpt])
)

chatglm3_6B = Model(
    name = 'chatglm3-6B',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([FreeChatgpt])
)


### 01-ai ###
Yi_1_5_9B_chat = Model(
    name = 'Yi-1.5-9B-Chat',
    base_provider = '01-ai',
    best_provider = IterListProvider([FreeChatgpt])
)


### Other ###
pi = Model(
    name = 'pi',
    base_provider = 'inflection',
    best_provider = Pi
)


#############
### Image ###
#############

### Stability AI ###
sdxl = Model(
    name = 'stability-ai/sdxl',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([DeepInfraImage])
    
)

stable_diffusion_3 = Model(
    name = 'stability-ai/stable-diffusion-3',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

sdxl_lightning_4step = Model(
    name = 'bytedance/sdxl-lightning-4step',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

playground_v2_5_1024px_aesthetic = Model(
    name = 'playgroundai/playground-v2.5-1024px-aesthetic',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

class ModelUtils:
    """
    Utility class for mapping string identifiers to Model instances.

    Attributes:
        convert (dict[str, Model]): Dictionary mapping model string identifiers to Model instances.
    """
    convert: dict[str, Model] = {
    
        ############
        ### Text ###
        ############
        
        ### OpenAI ###
        ### GPT-3.5 / GPT-4 ###
        # gpt-3.5
        'gpt-3.5-turbo': gpt_35_turbo,
        'gpt-3.5-long': gpt_35_long,

        # gpt-4
        'gpt-4o'         : gpt_4o,
        'gpt-4o-mini'    : gpt_4o_mini,
        'gpt-4'          : gpt_4,
        'gpt-4-turbo'    : gpt_4_turbo,
        
        ### Meta ###
        "meta-ai": meta,
        
        'llama-3-8b-instruct': llama_3_8b_instruct,
        'llama-3-70b-instruct': llama_3_70b_instruct,
        'llama-3-70b-chat': llama_3_70b_chat_hf, 
        'llama-3-70b-instruct': llama_3_70b_instruct, 
        
        'llama-3.1-70b': llama_3_1_70b_instruct,
        'llama-3.1-405b': llama_3_1_405b_instruct_FP8,
        'llama-3.1-70b-instruct': llama_3_1_70b_instruct,
        'llama-3.1-405b-instruct': llama_3_1_405b_instruct_FP8,
        
        ### Mistral (Opensource) ###
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-7b-v02': mistral_7b_v02,
        
        ### NousResearch ###
        'Nous-Hermes-2-Mixtral-8x7B-DPO': Nous_Hermes_2_Mixtral_8x7B_DPO,

        ### 01-ai ###
        'Yi-1.5-34b-chat': Yi_1_5_34B_chat,
        
        ### Microsoft ###
        'Phi-3-mini-4k-instruct': Phi_3_mini_4k_instruct,

        ### Google ###
        # gemini
        'gemini': gemini,
        'gemini-pro': gemini_pro,
        'gemini-pro': gemini_1_5,
        'gemini-flash': gemini_flash,
        
        # gemma
        'gemma-2b': gemma_2b_it,
        'gemma-2-9b': gemma_2_9b_it,
        'gemma-2-27b': gemma_2_27b_it,

        ### Anthropic ###
        'claude-2': claude_2,
        'claude-2.0': claude_2_0,
        'claude-2.1': claude_2_1,
        
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'claude-3-5-sonnet': claude_3_5_sonnet,
        'claude-3-haiku': claude_3_haiku,
        
        'claude-3-opus': claude_3,
        'claude-3-5-sonnet': claude_3_5,
        
        

        ### Reka AI ###
        'reka': reka_core,

        ### NVIDIA ###
        'nemotron-4-340b-instruct': nemotron_4_340b_instruct,
        
        ### Blackbox ###
        'blackbox': blackbox,
        
        ### CohereForAI ###
        'command-r+': command_r_plus,
        
        ### Databricks ###
        'dbrx-instruct': dbrx_instruct,

        ### GigaChat ###
        'gigachat': gigachat,
        
        ### iFlytek ###
        'SparkDesk-v1.1': SparkDesk_v1_1,
        
        ### DeepSeek ###
        'deepseek-coder': deepseek_coder,
        'deepseek-chat': deepseek_chat,
        
        ### Qwen ###
        'Qwen2-7b-instruct': Qwen2_7B_instruct,
        
        ### Zhipu AI ###
        'glm4-9b-chat': glm4_9B_chat,
        'chatglm3-6b': chatglm3_6B,
        
        ### 01-ai ###
        'Yi-1.5-9b-chat': Yi_1_5_9B_chat,
        
        # Other
        'pi': pi,
        
        #############
        ### Image ###
        #############
        
        ### Stability AI ###
        'sdxl': sdxl,
        'stable-diffusion-3': stable_diffusion_3,
        
        ### ByteDance ###
        'sdxl-lightning': sdxl_lightning_4step,
        
        ### Playground ###
        'playground-v2.5': playground_v2_5_1024px_aesthetic,

    }

_all_models = list(ModelUtils.convert.keys())
