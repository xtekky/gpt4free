from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AIChatFree,
    Airforce,
    AIUncensored,
    Bing,
    Blackbox,
    ChatGpt,
    Chatgpt4Online,
    ChatGptEs,
    Cloudflare,
    DarkAI,
    DDG,
    DeepInfraChat,
    Free2GPT,
    FreeNetfly,
    GigaChat,
    Gemini,
    GeminiPro,
    HuggingChat,
    HuggingFace,
    Liaobots,
    MagickPen,
    Mhystical,
    MetaAI,
    OpenaiChat,
    PerplexityLabs,
    Pi,
    Pizzagpt,
    Reka,
    ReplicateHome,
    RubiksAI,
    TeachAnything,
    Upstage,
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

### Default ###
default = Model(
    name          = "",
    base_provider = "",
    best_provider = IterListProvider([
        DDG,
        Pizzagpt,
        ReplicateHome,
        Upstage,
        Blackbox,
        Free2GPT,
        MagickPen,
        DeepInfraChat,
        Airforce, 
        ChatGptEs,
        Cloudflare,
        AIUncensored,
        DarkAI,
        Mhystical,
    ])
)

############
### Text ###
############

### OpenAI ###
# gpt-3.5
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Airforce])
)

# gpt-4
gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, ChatGptEs, DarkAI, ChatGpt, Airforce, Liaobots, OpenaiChat])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, ChatGptEs, FreeNetfly, Pizzagpt, ChatGpt, Airforce, RubiksAI, MagickPen, Liaobots, OpenaiChat])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Liaobots, Bing])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Chatgpt4Online, Bing, OpenaiChat, DDG, Liaobots, Airforce])
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
llama_2_7b = Model(
    name          = "llama-2-7b",
    base_provider = "Meta Llama",
    best_provider = Cloudflare
)
# llama 3
llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta Llama",
    best_provider = Cloudflare
)

# llama 3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, Cloudflare, Airforce, PerplexityLabs])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DDG, DeepInfraChat, Blackbox, TeachAnything, DarkAI, Airforce, RubiksAI, HuggingChat, HuggingFace, PerplexityLabs])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DarkAI])
)

# llama 3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Cloudflare])
)

llama_3_2_11b = Model(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral",
    best_provider = DDG
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

hermes_3 = Model(
    name          = "hermes-3",
    base_provider = "NousResearch",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

### Microsoft ###
phi_2 = Model(
    name          = "phi-2",
    base_provider = "Microsoft",
    best_provider = IterListProvider([Airforce])
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
    best_provider = IterListProvider([GeminiPro, Blackbox, AIChatFree, Liaobots])
)

gemini_flash = Model(
    name          = 'gemini-flash',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Blackbox, Liaobots])
)

gemini = Model(
    name          = 'gemini',
    base_provider = 'Google DeepMind',
    best_provider = Gemini
)

# gemma
gemma_2b = Model(
    name          = 'gemma-2b',
    base_provider = 'Google',
    best_provider = ReplicateHome
)

### Anthropic ###
claude_2_1 = Model(
    name          = 'claude-2.1',
    base_provider = 'Anthropic',
    best_provider = Liaobots
)

# claude 3
claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = Liaobots
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = Liaobots
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, Liaobots])
)

# claude 3.5
claude_3_5_sonnet = Model(
    name          = 'claude-3.5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, Liaobots])
)

### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
)

### Blackbox AI ###
blackboxai = Model(
    name = 'blackboxai',
    base_provider = 'Blackbox AI',
    best_provider = Blackbox
)

blackboxai_pro = Model(
    name = 'blackboxai-pro',
    base_provider = 'Blackbox AI',
    best_provider = Blackbox
)

### CohereForAI ###
command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = HuggingChat
)

### Qwen ###
# qwen 1_5
qwen_1_5_7b = Model(
    name = 'qwen-1.5-7b',
    base_provider = 'Qwen',
    best_provider = Cloudflare
)

# qwen 2
qwen_2_72b = Model(
    name = 'qwen-2-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, HuggingFace])
)

# qwen 2.5
qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

### Upstage ###
solar_mini = Model(
    name = 'solar-mini',
    base_provider = 'Upstage',
    best_provider = Upstage
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
deepseek_coder = Model(
    name = 'deepseek-coder',
    base_provider = 'DeepSeek',
    best_provider = Airforce
)

### WizardLM ###
wizardlm_2_8x22b = Model(
    name = 'wizardlm-2-8x22b',
    base_provider = 'WizardLM',
    best_provider = IterListProvider([DeepInfraChat])
)

### Yorickvp ###
llava_13b = Model(
    name = 'llava-13b',
    base_provider = 'Yorickvp',
    best_provider = ReplicateHome
)

### OpenChat ###
openchat_3_5 = Model(
    name = 'openchat-3.5',
    base_provider = 'OpenChat',
    best_provider = Airforce
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

grok_beta = Model(
    name = 'grok-beta',
    base_provider = 'x.ai',
    best_provider = Liaobots
)


### Perplexity AI ### 
sonar_online = Model(
    name = 'sonar-online',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PerplexityLabs])
)

sonar_chat = Model(
    name = 'sonar-chat',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

### Nvidia ### 
nemotron_70b = Model(
    name = 'nemotron-70b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)


### Teknium ### 
openhermes_2_5 = Model(
    name = 'openhermes-2.5',
    base_provider = 'Teknium',
    best_provider = Airforce
)

### Liquid ### 
lfm_40b = Model(
    name = 'lfm-40b',
    base_provider = 'Liquid',
    best_provider = IterListProvider([Airforce, PerplexityLabs])
)


### DiscoResearch ### 
german_7b = Model(
    name = 'german-7b',
    base_provider = 'DiscoResearch',
    best_provider = Airforce
)

### HuggingFaceH4 ### 
zephyr_7b = Model(
    name = 'zephyr-7b',
    base_provider = 'HuggingFaceH4',
    best_provider = Airforce
)

### Inferless ### 
neural_7b = Model(
    name = 'neural-7b',
    base_provider = 'inferless',
    best_provider = Airforce
)

#############
### Image ###
#############

### Stability AI ###
sdxl = Model(
    name = 'sdxl',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome])
    
)

sd_3 = Model(
    name = 'sd-3',
    base_provider = 'Stability AI',
    best_provider = ReplicateHome
    
)

### Playground ###
playground_v2_5 = Model(
    name = 'playground-v2.5',
    base_provider = 'Playground AI',
    best_provider = ReplicateHome
    
)


### Flux AI ###
flux = Model(
    name = 'flux',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Blackbox, AIUncensored, Airforce])
)

flux_pro = Model(
    name = 'flux-pro',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_realism = Model(
    name = 'flux-realism',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_anime = Model(
    name = 'flux-anime',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_3d = Model(
    name = 'flux-3d',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_disney = Model(
    name = 'flux-disney',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_pixel = Model(
    name = 'flux-pixel',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_4o = Model(
    name = 'flux-4o',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

### Other ###
any_dark = Model(
    name = 'any-dark',
    base_provider = '',
    best_provider = Airforce
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
        'gpt-3': gpt_35_turbo,

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
        'llama-2-7b': llama_2_7b,

        # llama-3
        'llama-3-8b': llama_3_8b,
                
        # llama-3.1
        'llama-3.1-8b': llama_3_1_8b,
        'llama-3.1-70b': llama_3_1_70b,
        'llama-3.1-405b': llama_3_1_405b,

        # llama-3.2
        'llama-3.2-1b': llama_3_2_1b,
        'llama-3.2-11b': llama_3_2_11b,
                
        ### Mistral ###
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-nemo': mistral_nemo,

        ### NousResearch ###
        'hermes-3': hermes_3,
                
        ### Microsoft ###
        'phi-2': phi_2,
        'phi-3.5-mini': phi_3_5_mini,

        ### Google ###
        # gemini
        'gemini': gemini,
        'gemini-pro': gemini_pro,
        'gemini-flash': gemini_flash,

        # gemma
        'gemma-2b': gemma_2b,

        ### Anthropic ###
        'claude-2.1': claude_2_1,

        # claude 3
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'claude-3-haiku': claude_3_haiku,

        # claude 3.5
        'claude-3.5-sonnet': claude_3_5_sonnet,

        ### Reka AI ###
        'reka-core': reka_core,

        ### Blackbox AI ###
        'blackboxai': blackboxai,
        'blackboxai-pro': blackboxai_pro,

        ### CohereForAI ###
        'command-r+': command_r_plus,

        ### GigaChat ###
        'gigachat': gigachat,

        'qwen-1.5-7b': qwen_1_5_7b,
        'qwen-2-72b': qwen_2_72b,
                
        ### Upstage ###
        'solar-pro': solar_pro,

        ### Inflection ###
        'pi': pi,

        ### Yorickvp ###
        'llava-13b': llava_13b,

        ### WizardLM ###
        'wizardlm-2-8x22b': wizardlm_2_8x22b,

        ### OpenChat ###
        'openchat-3.5': openchat_3_5,

        ### x.ai ###
        'grok-2': grok_2,
        'grok-2-mini': grok_2_mini,
        'grok-beta': grok_beta,

        ### Perplexity AI ###
        'sonar-online': sonar_online,
        'sonar-chat': sonar_chat,

        ### TheBloke ###   
        'german-7b': german_7b,

        ### Nvidia ###   
        'nemotron-70b': nemotron_70b,

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
        'flux-pro': flux_pro,
        'flux-realism': flux_realism,
        'flux-anime': flux_anime,
        'flux-3d': flux_3d,
        'flux-disney': flux_disney,
        'flux-pixel': flux_pixel,
        'flux-4o': flux_4o,

        ### Other ###
        'any-dark': any_dark,
    }

_all_models = list(ModelUtils.convert.keys())