from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AIChatFree,
    Airforce,
    Allyfy,
    Bing,
    Binjie,
    Blackbox,
    ChatGpt,
    Chatgpt4Online,
    ChatGptEs,
    ChatgptFree,
    ChatHub,
    DDG,
    DeepInfra,
    DeepInfraChat,
    DeepInfraImage,
    Free2GPT,
    FreeChatgpt,
    FreeGpt,
    FreeNetfly,
    Gemini,
    GeminiPro,
    GigaChat,
    GPROChat,
    HuggingChat,
    HuggingFace,
    Koala,
    Liaobots,
    LiteIcoding,
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
    TeachAnything,
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
        ReplicateHome,
        Upstage,
        Blackbox,
        Binjie,
        Free2GPT,
        MagickPen,
        DeepInfraChat,
        LiteIcoding,
        Airforce, 
        ChatHub,
        Nexra,
        ChatGptEs,
        ChatHub,
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
    best_provider = Nexra
)

# gpt-3.5
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Allyfy, Nexra, Airforce, Liaobots,
    ])
)

# gpt-4
gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Liaobots, Nexra, ChatGptEs, Airforce, 
        OpenaiChat
    ])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        DDG, ChatGptEs, You, FreeNetfly, Pizzagpt, LiteIcoding, MagickPen, Liaobots, Airforce, ChatgptFree, Koala,
        OpenaiChat, ChatGpt
    ])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Nexra, Liaobots, Airforce, Bing
    ])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([
        Nexra, Binjie, Airforce, Chatgpt4Online, Bing, OpenaiChat,
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

# llama 2
llama_2_13b = Model(
    name          = "llama-2-13b",
    base_provider = "Meta Llama",
    best_provider = Airforce
)

# llama 3
llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Airforce, DeepInfra, Replicate])
)

llama_3_70b = Model(
    name          = "llama-3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([ReplicateHome, Airforce, DeepInfra, Replicate])
)

llama_3 = Model(
    name          = "llama-3",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([llama_3_8b.best_provider, llama_3_70b.best_provider])
)

# llama 3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, ChatHub, Airforce, PerplexityLabs])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DDG, HuggingChat, Blackbox, FreeGpt, TeachAnything, Free2GPT, DeepInfraChat, Airforce, HuggingFace, PerplexityLabs])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, Blackbox, Airforce])
)

llama_3_1 = Model(
    name          = "llama-3.1",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Nexra, llama_3_1_8b.best_provider, llama_3_1_70b.best_provider, llama_3_1_405b.best_provider,])
)

# llama 3.2
llama_3_2_11b = Model(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Airforce])
)

# llamaguard
llamaguard_7b = Model(
    name          = "llamaguard-7b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Airforce])
)

llamaguard_2_8b = Model(
    name          = "llamaguard-2-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Airforce])
)


### Mistral ###
mistral_7b = Model(
    name          = "mistral-7b",
    base_provider = "Mistral",
    best_provider = IterListProvider([DeepInfraChat, Airforce, HuggingFace, DeepInfra])
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral",
    best_provider = IterListProvider([DDG, ReplicateHome, DeepInfraChat, ChatHub, Airforce, DeepInfra])
)

mixtral_8x22b = Model(
    name          = "mixtral-8x22b",
    base_provider = "Mistral",
    best_provider = IterListProvider([DeepInfraChat, Airforce])
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)


### NousResearch ###
mixtral_8x7b_dpo = Model(
    name          = "mixtral-8x7b-dpo",
    base_provider = "NousResearch",
    best_provider = IterListProvider([Airforce])
)

yi_34b = Model(
    name          = "yi-34b",
    base_provider = "NousResearch",
    best_provider = IterListProvider([Airforce])
)

hermes_3 = Model(
    name          = "hermes-3",
    base_provider = "NousResearch",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)


### Microsoft ###
phi_3_medium_4k = Model(
    name          = "phi-3-medium-4k",
    base_provider = "Microsoft",
    best_provider = DeepInfraChat
)

phi_3_5_mini = Model(
    name          = "phi-3.5-mini",
    base_provider = "Microsoft",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

### Google DeepMind ###
# gemini
gemini_pro = Model(
    name          = 'gemini-pro',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([GeminiPro, LiteIcoding, Blackbox, AIChatFree, GPROChat, Nexra, Liaobots, Airforce])
)

gemini_flash = Model(
    name          = 'gemini-flash',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Blackbox, Liaobots, Airforce])
)

gemini = Model(
    name          = 'gemini',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Gemini, gemini_flash.best_provider, gemini_pro.best_provider])
)

# gemma
gemma_2b_9b = Model(
    name          = 'gemma-2b-9b',
    base_provider = 'Google',
    best_provider = Airforce
)

gemma_2b_27b = Model(
    name          = 'gemma-2b-27b',
    base_provider = 'Google',
    best_provider = IterListProvider([DeepInfraChat, Airforce])
)

gemma_2b = Model(
    name          = 'gemma-2b',
    base_provider = 'Google',
    best_provider = IterListProvider([
        ReplicateHome, Airforce,
        gemma_2b_9b.best_provider, gemma_2b_27b.best_provider,
   ])
)

# gemma 2
gemma_2_27b = Model(
    name          = 'gemma-2-27b',
    base_provider = 'Google',
    best_provider = Airforce
)

gemma_2 = Model(
    name          = 'gemma-2',
    base_provider = 'Google',
    best_provider = IterListProvider([
        ChatHub,
        gemma_2_27b.best_provider,
   ])
)


### Anthropic ###
claude_2_1 = Model(
    name          = 'claude-2.1',
    base_provider = 'Anthropic',
    best_provider = Liaobots
)

claude_2 = Model(
    name          = 'claude-2',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([
        You,
        claude_2_1.best_provider,
   ])
)

# claude 3
claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Airforce, Liaobots])
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Airforce, Liaobots])
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, Airforce, Liaobots])
)

claude_3 = Model(
    name          = 'claude-3',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([
        claude_3_opus.best_provider, claude_3_sonnet.best_provider, claude_3_haiku.best_provider
   ])
)

# claude 3.5
claude_3_5_sonnet = Model(
    name          = 'claude-3.5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, Airforce, Liaobots])
)

claude_3_5 = Model(
    name          = 'claude-3.5',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([
        LiteIcoding,
        claude_3_5_sonnet.best_provider
   ])
)



### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
)


### Blackbox AI ###
blackbox = Model(
    name = 'blackbox',
    base_provider = 'Blackbox AI',
    best_provider = Blackbox
)


### Databricks ###
dbrx_instruct = Model(
    name = 'dbrx-instruct',
    base_provider = 'Databricks',
    best_provider = IterListProvider([Airforce, DeepInfra])
)


### CohereForAI ###
command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = HuggingChat
)


### iFlytek ###
sparkdesk_v1_1 = Model(
    name = 'sparkdesk-v1.1',
    base_provider = 'iFlytek',
    best_provider = IterListProvider([FreeChatgpt])
)


### Qwen ###
# qwen 1
qwen_1_5_7b = Model(
    name = 'qwen-1.5-7b',
    base_provider = 'Qwen',
    best_provider = Airforce
)

qwen_1_5_14b = Model(
    name = 'qwen-1.5-14b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([FreeChatgpt, Airforce])
)

qwen_1_5_72b = Model(
    name = 'qwen-1.5-72b',
    base_provider = 'Qwen',
    best_provider = Airforce
)

qwen_1_5_110b = Model(
    name = 'qwen-1.5-110b',
    base_provider = 'Qwen',
    best_provider = Airforce
)

# qwen 2
qwen_2_72b = Model(
    name = 'qwen-2-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, Airforce, HuggingFace])
)

qwen = Model(
    name = 'qwen',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Nexra, qwen_1_5_14b.best_provider, qwen_1_5_72b.best_provider, qwen_1_5_110b.best_provider, qwen_2_72b.best_provider])
)


### Zhipu AI ###
glm_3_6b = Model(
    name = 'glm-3-6b',
    base_provider = 'Zhipu AI',
    best_provider = FreeChatgpt
)

glm_4_9b = Model(
    name = 'glm-4-9B',
    base_provider = 'Zhipu AI',
    best_provider = FreeChatgpt
)

glm_4 = Model(
    name = 'glm-4',
    base_provider = 'Zhipu AI',
    best_provider = IterListProvider([
        glm_3_6b.best_provider, glm_4_9b.best_provider
    ])
)


### 01-ai ###
yi_1_5_9b = Model(
    name = 'yi-1.5-9b',
    base_provider = '01-ai',
    best_provider = FreeChatgpt
)

### Upstage ###
solar_1_mini = Model(
    name = 'solar-1-mini',
    base_provider = 'Upstage',
    best_provider = Upstage
)

solar_10_7b = Model(
    name = 'solar-10-7b',
    base_provider = 'Upstage',
    best_provider = Airforce
)

solar_pro = Model(
    name = 'solar-pro',
    base_provider = 'Upstage',
    best_provider = Upstage
)


### Inflection ###
pi = Model(
    name = 'pi',
    base_provider = 'Inflection',
    best_provider = Pi
)

### DeepSeek ###
deepseek = Model(
    name = 'deepseek',
    base_provider = 'DeepSeek',
    best_provider = Airforce
)

### WizardLM ###
wizardlm_2_7b = Model(
    name = 'wizardlm-2-7b',
    base_provider = 'WizardLM',
    best_provider = DeepInfraChat
)

wizardlm_2_8x22b = Model(
    name = 'wizardlm-2-8x22b',
    base_provider = 'WizardLM',
    best_provider = IterListProvider([DeepInfraChat, Airforce])
)

### Yorickvp ###
llava_13b = Model(
    name = 'llava-13b',
    base_provider = 'Yorickvp',
    best_provider = ReplicateHome
)


### OpenBMB ###
minicpm_llama_3_v2_5 = Model(
    name = 'minicpm-llama-3-v2.5',
    base_provider = 'OpenBMB',
    best_provider = DeepInfraChat
)


### Lzlv ###
lzlv_70b = Model(
    name = 'lzlv-70b',
    base_provider = 'Lzlv',
    best_provider = DeepInfraChat
)


### OpenChat ###
openchat_3_6_8b = Model(
    name = 'openchat-3.6-8b',
    base_provider = 'OpenChat',
    best_provider = DeepInfraChat
)


### Phind ###
phind_codellama_34b_v2 = Model(
    name = 'phind-codellama-34b-v2',
    base_provider = 'Phind',
    best_provider = DeepInfraChat
)


### Cognitive Computations ###
dolphin_2_9_1_llama_3_70b = Model(
    name = 'dolphin-2.9.1-llama-3-70b',
    base_provider = 'Cognitive Computations',
    best_provider = DeepInfraChat
)


### x.ai ###
grok_2 = Model(
    name = 'grok-2',
    base_provider = 'x.ai',
    best_provider = Liaobots
)

grok_2_mini = Model(
    name = 'grok-2-mini',
    base_provider = 'x.ai',
    best_provider = Liaobots
)


### Perplexity AI ### 
sonar_online = Model(
    name = 'sonar-online',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([ChatHub, PerplexityLabs])
)

sonar_chat = Model(
    name = 'sonar-chat',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)


### Gryphe ### 
mythomax_l2_13b = Model(
    name = 'mythomax-l2-13b',
    base_provider = 'Gryphe',
    best_provider = IterListProvider([Airforce])
)


### Pawan ### 
cosmosrp = Model(
    name = 'cosmosrp',
    base_provider = 'Pawan',
    best_provider = IterListProvider([Airforce])
)



#############
### Image ###
#############

### Stability AI ###
sdxl = Model(
    name = 'sdxl',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome, Nexra, DeepInfraImage])
    
)

sd_3 = Model(
    name = 'sd-3',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)


### Playground ###
playground_v2_5 = Model(
    name = 'playground-v2.5',
    base_provider = 'Playground AI',
    best_provider = IterListProvider([ReplicateHome])
    
)


### Flux AI ###
flux = Model(
    name = 'flux',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce, Blackbox])
    
)

flux_realism = Model(
    name = 'flux-realism',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_anime = Model(
    name = 'flux-anime',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_3d = Model(
    name = 'flux-3d',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_disney = Model(
    name = 'flux-disney',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_pixel = Model(
    name = 'flux-pixel',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_4o = Model(
    name = 'flux-4o',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce])
    
)

flux_schnell = Model(
    name = 'flux-schnell',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([ReplicateHome])
    
)


### ###
dalle_2 = Model(
    name = 'dalle-2',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
)
dalle_3 = Model(
    name = 'dalle-3',
    base_provider = '',
    best_provider = IterListProvider([Airforce])
    
)

dalle = Model(
    name = 'dalle',
    base_provider = '',
    best_provider = IterListProvider([Nexra, dalle_2.best_provider, dalle_3.best_provider])
    
)

dalle_mini = Model(
    name = 'dalle-mini',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
)

### Other ###
emi = Model(
    name = 'emi',
    base_provider = '',
    best_provider = IterListProvider([Nexra])
    
)

any_dark = Model(
    name = 'any-dark',
    base_provider = '',
    best_provider = IterListProvider([Airforce])
    
)

prodia = Model(
    name = 'prodia',
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
'gpt-4o': gpt_4o,
'gpt-4o-mini': gpt_4o_mini,
'gpt-4': gpt_4,
'gpt-4-turbo': gpt_4_turbo,
       
        
### Meta ###
"meta-ai": meta,

# llama-2
'llama-2-13b': llama_2_13b,

# llama-3
'llama-3': llama_3,
'llama-3-8b': llama_3_8b,
'llama-3-70b': llama_3_70b,
        
# llama-3.1
'llama-3.1': llama_3_1,
'llama-3.1-8b': llama_3_1_8b,
'llama-3.1-70b': llama_3_1_70b,
'llama-3.1-405b': llama_3_1_405b,

# llama-3.2
'llama-3.2-11b': llama_3_2_11b,
'llama-3.2-90b': llama_3_2_90b,

# llamaguard
'llamaguard-7b': llamaguard_7b,
'llamaguard-2-8b': llamaguard_2_8b,
      
        
### Mistral ###
'mistral-7b': mistral_7b,
'mixtral-8x7b': mixtral_8x7b,
'mixtral-8x22b': mixtral_8x22b,
'mistral-nemo': mistral_nemo,
     
     
### NousResearch ###
'mixtral-8x7b-dpo': mixtral_8x7b_dpo,
'hermes-3': hermes_3,
 
'yi-34b': yi_34b,   
        
        
### Microsoft ###
'phi_3_medium-4k': phi_3_medium_4k,
'phi-3.5-mini': phi_3_5_mini,

### Google ###
# gemini
'gemini': gemini,
'gemini-pro': gemini_pro,
'gemini-flash': gemini_flash,
        
# gemma
'gemma-2b': gemma_2b,
'gemma-2b-9b': gemma_2b_9b,
'gemma-2b-27b': gemma_2b_27b,

# gemma-2
'gemma-2': gemma_2,
'gemma-2-27b': gemma_2_27b,


### Anthropic ###
'claude-2': claude_2,
'claude-2.1': claude_2_1,

# claude 3
'claude-3': claude_3,
'claude-3-opus': claude_3_opus,
'claude-3-sonnet': claude_3_sonnet,
'claude-3-haiku': claude_3_haiku,

# claude 3.5
'claude-3.5': claude_3_5,
'claude-3.5-sonnet': claude_3_5_sonnet,
        
        
### Reka AI ###
'reka-core': reka_core,
      
        
### Blackbox AI ###
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
'qwen': qwen,
'qwen-1.5-7b': qwen_1_5_7b,
'qwen-1.5-14b': qwen_1_5_14b,
'qwen-1.5-72b': qwen_1_5_72b,
'qwen-1.5-110b': qwen_1_5_110b,
'qwen-2-72b': qwen_2_72b,
        
        
### Zhipu AI ###
'glm-3-6b': glm_3_6b,
'glm-4-9b': glm_4_9b,
'glm-4': glm_4,
        
        
### 01-ai ###
'yi-1.5-9b': yi_1_5_9b,
        
        
### Upstage ###
'solar-1-mini': solar_1_mini,
'solar-10-7b': solar_10_7b,
'solar-pro': solar_pro,


### Inflection ###
'pi': pi,

### DeepSeek ###
'deepseek': deepseek,
     
        
### Yorickvp ###
'llava-13b': llava_13b,


### WizardLM ###
'wizardlm-2-7b': wizardlm_2_7b,
'wizardlm-2-8x22b': wizardlm_2_8x22b,
      
        
### OpenBMB ###
'minicpm-llama-3-v2.5': minicpm_llama_3_v2_5,
        
        
### Lzlv ###
'lzlv-70b': lzlv_70b,
     
        
### OpenChat ###
'openchat-3.6-8b': openchat_3_6_8b,


### Phind ###
'phind-codellama-34b-v2': phind_codellama_34b_v2,
        
        
### Cognitive Computations ###
'dolphin-2.9.1-llama-3-70b': dolphin_2_9_1_llama_3_70b,
    
        
### x.ai ###
'grok-2': grok_2,
'grok-2-mini': grok_2_mini,
        
        
### Perplexity AI ###
'sonar-online': sonar_online,
'sonar-chat': sonar_chat,


### Gryphe ###   
'mythomax-l2-13b': sonar_chat,

   
### Pawan ###   
'cosmosrp': cosmosrp,
        
        
        
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
'flux-pixel': flux_pixel,
'flux-4o': flux_4o,
'flux-schnell': flux_schnell,


###  ###
'dalle': dalle,
'dalle-2': dalle_2,
'dalle-3': dalle_3,
'dalle-mini': dalle_mini,
'emi': emi,
'any-dark': any_dark,
'prodia': prodia,
    }

_all_models = list(ModelUtils.convert.keys())
