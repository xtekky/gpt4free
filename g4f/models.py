from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AiChatOnline,
    Allyfy,
    Bing,
    Binjie,
    Bixin123,
    Blackbox,
    ChatGot,
    Chatgpt4Online,
    Chatgpt4o,
    ChatgptFree,
    CodeNews,
    DDG,
    DeepInfra,
    DeepInfraImage,
    FluxAirforce,
    Free2GPT,
    FreeChatgpt,
    FreeGpt,
    FreeNetfly,
    Gemini,
    GeminiPro,
    GigaChat,
    HuggingChat,
    HuggingFace,
    Koala,
    Liaobots,
    MagickPen,
    MetaAI,
    Nexra,
    OpenaiChat,
    PerplexityLabs,
    Pi,
    Pizzagpt,
    Reka,
    Replicate,
    ReplicateHome,
    Snova,
    TeachAnything,
    TwitterBio,
    Upstage,
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
        DDG,
        FreeChatgpt,
        HuggingChat,
        Pizzagpt,
        ChatgptFree,
        ReplicateHome,
        Upstage,
        Blackbox,
        Bixin123,
        Binjie,
        Free2GPT,
        MagickPen,
    ])
)

############
### Text ###
############

### OpenAI ###
# gpt-3
gpt_3 = Model(
    name          = 'gpt-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Nexra,
    ])
)

# gpt-3.5
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Allyfy, TwitterBio, Nexra, Bixin123, CodeNews,
    ])
)

# gpt-4
gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Liaobots, Chatgpt4o, OpenaiChat,
    ])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        DDG, Liaobots, You, FreeNetfly, Pizzagpt, ChatgptFree, AiChatOnline, CodeNews, 
        MagickPen, OpenaiChat, Koala,       
    ])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Nexra, Bixin123, Liaobots, Bing
    ])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Chatgpt4Online, Nexra, Binjie, Bing,
        gpt_4_turbo.best_provider, gpt_4o.best_provider, gpt_4o_mini.best_provider
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
    name          = "meta-ai",
    base_provider = "Meta",
    best_provider = MetaAI
)

llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta",
    best_provider = IterListProvider([DeepInfra, Replicate])
)

llama_3_70b = Model(
    name          = "llama-3-70b",
    base_provider = "Meta",
    best_provider = IterListProvider([ReplicateHome, DeepInfra, PerplexityLabs, Replicate])
)

llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta",
    best_provider = IterListProvider([Blackbox])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta",
    best_provider = IterListProvider([DDG, HuggingChat, FreeGpt, Blackbox, TeachAnything, Free2GPT, HuggingFace])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta",
    best_provider = IterListProvider([HuggingChat, Blackbox, HuggingFace])
)

### Mistral ###
mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral",
    best_provider = IterListProvider([HuggingChat, DDG, ReplicateHome, TwitterBio, DeepInfra, HuggingFace,])
)

mistral_7b = Model(
    name          = "mistral-7b",
    base_provider = "Mistral",
    best_provider = IterListProvider([HuggingChat, HuggingFace, DeepInfra])
)

### 01-ai ###
yi_1_5_34b = Model(
    name          = "yi-1.5-34b",
    base_provider = "01-ai",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)


### Microsoft ###
phi_3_mini_4k = Model(
    name          = "phi-3-mini-4k",
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
    best_provider = IterListProvider([GeminiPro, ChatGot, Liaobots])
)

gemini_flash = Model(
    name          = 'gemini-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Liaobots, Blackbox])
)

# gemma
gemma_2b = Model(
    name          = 'gemma-2b',
    base_provider = 'Google',
    best_provider = IterListProvider([ReplicateHome])
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
    best_provider = IterListProvider([Liaobots])
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Liaobots])
)

claude_3_5_sonnet = Model(
    name          = 'claude-3-5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Liaobots])
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, Liaobots])
)


### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
)


### Blackbox ###
blackbox = Model(
    name = 'blackbox',
    base_provider = 'Blackbox',
    best_provider = Blackbox
)


### Databricks ###
dbrx_instruct = Model(
    name = 'dbrx-instruct',
    base_provider = 'Databricks',
    best_provider = IterListProvider([DeepInfra])
)


### CohereForAI ###
command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingChat])
)


### iFlytek ###
sparkdesk_v1_1 = Model(
    name = 'sparkdesk-v1.1',
    base_provider = 'iFlytek',
    best_provider = IterListProvider([FreeChatgpt])
)

### Qwen ###
qwen_1_5_14b = Model(
    name = 'qwen-1.5-14b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([FreeChatgpt])
)

qwen_turbo = Model(
    name = 'qwen-turbo',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Bixin123])
)


### Zhipu AI ###
glm_3_6b = Model(
    name = 'glm-3-6b',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([FreeChatgpt])
)

glm_4_9b = Model(
    name = 'glm-4-9B',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([FreeChatgpt])
)

glm_4 = Model(
    name = 'glm-4',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([CodeNews, glm_4_9b.best_provider,])
)

### 01-ai ###
yi_1_5_9b = Model(
    name = 'yi-1.5-9b',
    base_provider = '01-ai',
    best_provider = IterListProvider([FreeChatgpt])
)


### Pi ###
solar_1_mini = Model(
    name = 'solar-1-mini',
    base_provider = 'Upstage',
    best_provider = IterListProvider([Upstage])
)

### Pi ###
pi = Model(
    name = 'pi',
    base_provider = 'inflection',
    best_provider = Pi
)

### SambaNova ###
samba_coe_v0_1 = Model(
    name = 'samba-coe-v0.1',
    base_provider = 'SambaNova',
    best_provider = Snova
)

### Trong-Hieu Nguyen-Mau ###
v1olet_merged_7b = Model(
    name = 'v1olet-merged-7b',
    base_provider = 'Trong-Hieu Nguyen-Mau',
    best_provider = Snova
)

### Macadeliccc ###
westlake_7b_v2 = Model(
    name = 'westlake-7b-v2',
    base_provider = 'Macadeliccc',
    best_provider = Snova
)

### CookinAI ###
donutlm_v1 = Model(
    name = 'donutlm-v1',
    base_provider = 'CookinAI',
    best_provider = Snova
)

### DeepSeek ###
deepseek = Model(
    name = 'deepseek',
    base_provider = 'DeepSeek',
    best_provider = CodeNews
)



#############
### Image ###
#############

### Stability AI ###
sdxl = Model(
    name = 'sdxl',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome, DeepInfraImage])
    
)

sd_3 = Model(
    name = 'sd-3',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

### Playground ###
playground_v2_5 = Model(
    name = 'playground-v2.5',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

### Flux AI ###
flux = Model(
    name = 'flux',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([FluxAirforce])
    
)

flux_realism = Model(
    name = 'flux-realism',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([FluxAirforce])
    
)

flux_anime = Model(
    name = 'flux-anime',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([FluxAirforce])
    
)

flux_3d = Model(
    name = 'flux-3d',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([FluxAirforce])
    
)

flux_disney = Model(
    name = 'flux-disney',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([FluxAirforce])
    
)

### ###
dalle = Model(
    name = 'dalle',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
)

dalle_mini = Model(
    name = 'dalle-mini',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
)

emi = Model(
    name = 'emi',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
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
# gpt-3
'gpt-3': gpt_3,

# gpt-3.5
'gpt-3.5-turbo': gpt_35_turbo,

# gpt-4
'gpt-4o'         : gpt_4o,
'gpt-4o-mini'    : gpt_4o_mini,
'gpt-4'          : gpt_4,
'gpt-4-turbo'    : gpt_4_turbo,
        
        
### Meta ###
"meta-ai": meta,

# llama-3
'llama-3-8b': llama_3_8b,
'llama-3-70b': llama_3_70b,
        
# llama-3.1
'llama-3.1-8b': llama_3_1_8b,
'llama-3.1-70b': llama_3_1_70b,
'llama-3.1-405b': llama_3_1_405b,
        
        
### Mistral ###
'mixtral-8x7b': mixtral_8x7b,
'mistral-7b': mistral_7b,


### 01-ai ###
'yi-1.5-34b': yi_1_5_34b,
        
        
### Microsoft ###
'phi-3-mini-4k': phi_3_mini_4k,


### Google ###
# gemini
'gemini': gemini,
'gemini-pro': gemini_pro,
'gemini-flash': gemini_flash,
        
# gemma
'gemma-2b': gemma_2b,


### Anthropic ###
'claude-2': claude_2,
'claude-2.0': claude_2_0,
'claude-2.1': claude_2_1,
        
'claude-3-opus': claude_3_opus,
'claude-3-sonnet': claude_3_sonnet,
'claude-3-5-sonnet': claude_3_5_sonnet,
'claude-3-haiku': claude_3_haiku,
        
        
### Reka AI ###
'reka-core': reka_core,
      
        
### Blackbox ###
'blackbox': blackbox,
        
        
### CohereForAI ###
'command-r+': command_r_plus,
        
        
### Databricks ###
'dbrx-instruct': dbrx_instruct,


### GigaChat ###
'gigachat': gigachat,
        
        
### iFlytek ###
'sparkdesk-v1.1': sparkdesk_v1_1,
        
        
### Qwen ###
'qwen-1.5-14b': qwen_1_5_14b,
'qwen-turbo': qwen_turbo,
        
        
### Zhipu AI ###
'glm-3-6b': glm_3_6b,
'glm-4-9b': glm_4_9b,
'glm-4': glm_4,
        
        
### 01-ai ###
'yi-1.5-9b': yi_1_5_9b,
        
        
### Upstage ###
'solar-1-mini': solar_1_mini,


### Pi ###
'pi': pi,


### SambaNova ###
'samba-coe-v0.1': samba_coe_v0_1,


### Trong-Hieu Nguyen-Mau ###
'v1olet-merged-7b': v1olet_merged_7b,


### Macadeliccc ###
'westlake-7b-v2': westlake_7b_v2,


### CookinAI ###
'donutlm-v1': donutlm_v1,

### DeepSeek ###
'deepseek': deepseek,
        
        
        
#############
### Image ###
#############
        
### Stability AI ###
'sdxl': sdxl,
'sd-3': sd_3,
        
        
### Playground ###
'playground-v2.5': playground_v2_5,


### Flux AI ###
'flux': flux,
'flux-realism': flux_realism,
'flux-anime': flux_anime,
'flux-3d': flux_3d,
'flux-disney': flux_disney,


###  ###
'dalle': dalle,
'dalle-mini': dalle_mini,
'emi': emi,
    }

_all_models = list(ModelUtils.convert.keys())
