from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AIChatFree,
    AmigoChat,
    Blackbox,
    Blackbox2,
    BingCreateImages,
    ChatGpt,
    ChatGptEs,
    Cloudflare,
    Copilot,
    CopilotAccount,
    DDG,
    DeepInfraChat,
    Free2GPT,
    GigaChat,
    Gemini,
    GeminiPro,
    HuggingChat,
    HuggingFace,
    Liaobots,
    Airforce,
    MagickPen,
    Mhystical,
    MetaAI,
    MicrosoftDesigner,
    OpenaiChat,
    OpenaiAccount,
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
        Blackbox2,
        Upstage,
        Blackbox,
        Free2GPT,
        DeepInfraChat,
        Airforce, 
        ChatGptEs,
        Cloudflare,
        Mhystical,
        AmigoChat,
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
    best_provider = Blackbox
)

# gpt-4
gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, ChatGptEs, ChatGpt, AmigoChat, Airforce, Liaobots, OpenaiChat])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, ChatGptEs, Pizzagpt, ChatGpt, AmigoChat, Airforce, RubiksAI, MagickPen, Liaobots, OpenaiChat])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Liaobots, Airforce])
)

gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, Copilot, OpenaiChat, Liaobots, Airforce])
)

# o1
o1_preview = Model(
    name          = 'o1-preview',
    base_provider = 'OpenAI',
    best_provider = Liaobots
)

o1_mini = Model(
    name          = 'o1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Liaobots, Airforce])
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
    best_provider = IterListProvider([Cloudflare, Airforce])
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
    best_provider = IterListProvider([DDG, DeepInfraChat, Blackbox, Blackbox2, TeachAnything, Airforce, RubiksAI, HuggingChat, HuggingFace, PerplexityLabs])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, AmigoChat])
)

# llama 3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = Cloudflare
)

llama_3_2_11b = Model(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = AmigoChat
)

# CodeLlama
codellama_34b = Model(
    name          = "codellama-34b",
    base_provider = "Meta Llama",
    best_provider = AmigoChat
)

### Mistral ###
mixtral_7b = Model(
    name          = "mixtral-7b",
    base_provider = "Mistral",
    best_provider = AmigoChat
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral",
    best_provider = DDG
)

mistral_tiny = Model(
    name          = "mistral-tiny",
    base_provider = "Mistral",
    best_provider = AmigoChat
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral",
    best_provider = IterListProvider([HuggingChat, AmigoChat, HuggingFace])
)

### NousResearch ###
hermes_2_dpo = Model(
    name          = "hermes-2-dpo",
    base_provider = "NousResearch",
    best_provider = Airforce
)

hermes_2_pro = Model(
    name          = "hermes-2-pro",
    base_provider = "NousResearch",
    best_provider = Airforce
)

hermes_3 = Model(
    name          = "hermes-3",
    base_provider = "NousResearch",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

mixtral_8x7b_dpo = Model(
    name          = "mixtral-8x7b-dpo",
    base_provider = "NousResearch",
    best_provider = IterListProvider([AmigoChat, Airforce])
)

### Microsoft ###
phi_2 = Model(
    name          = "phi-2",
    base_provider = "Microsoft",
    best_provider = Airforce
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
    best_provider = IterListProvider([Blackbox, AIChatFree, GeminiPro, Liaobots])
)

gemini_flash = Model(
    name          = 'gemini-flash',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Blackbox, AmigoChat, Liaobots])
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
    best_provider = IterListProvider([ReplicateHome, AmigoChat])
)

### Anthropic ###
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
    best_provider = IterListProvider([Blackbox, AmigoChat, Liaobots])
)

claude_3_5_haiku = Model(
    name          = 'claude-3.5-haiku',
    base_provider = 'Anthropic',
    best_provider = AmigoChat
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
    best_provider = IterListProvider([HuggingChat, AmigoChat])
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
qwen_2_5_72b = Model(
    name = 'qwen-2.5-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([AmigoChat, HuggingChat, HuggingFace])
)

qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, HuggingFace])
)

qwq_32b = Model(
    name = 'qwq-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, HuggingFace])
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
deepseek_chat = Model(
    name = 'deepseek-chat',
    base_provider = 'DeepSeek',
    best_provider = AmigoChat
)

deepseek_coder = Model(
    name = 'deepseek-coder',
    base_provider = 'DeepSeek',
    best_provider = Airforce
)

### WizardLM ###
wizardlm_2_8x22b = Model(
    name = 'wizardlm-2-8x22b',
    base_provider = 'WizardLM',
    best_provider = DeepInfraChat
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
    best_provider = IterListProvider([AmigoChat, Liaobots])
)


### Perplexity AI ### 
sonar_online = Model(
    name = 'sonar-online',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
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
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, HuggingFace])
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

### Gryphe ### 
mythomax_13b = Model(
    name = 'mythomax-13b',
    base_provider = 'Gryphe',
    best_provider = AmigoChat
)

### databricks ### 
dbrx_instruct = Model(
    name = 'dbrx-instruct',
    base_provider = 'databricks',
    best_provider = AmigoChat
)

### anthracite-org ### 
magnum_72b = Model(
    name = 'magnum-72b',
    base_provider = 'anthracite-org',
    best_provider = AmigoChat
)

### ai21 ### 
jamba_mini = Model(
    name = 'jamba-mini',
    base_provider = 'ai21',
    best_provider = AmigoChat
)

### llmplayground.net ### 
any_uncensored = Model(
    name = 'any-uncensored',
    base_provider = 'llmplayground.net',
    best_provider = Airforce
)

#############
### Image ###
#############

### Stability AI ###
sdxl = Model(
    name = 'sdxl',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([ReplicateHome, Airforce])
    
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
    best_provider = IterListProvider([Blackbox, Airforce])
)

flux_pro = Model(
    name = 'flux-pro',
    base_provider = 'Flux AI',
    best_provider = Airforce
)

flux_dev = Model(
    name = 'flux-dev',
    base_provider = 'Flux AI',
    best_provider = AmigoChat
)

flux_realism = Model(
    name = 'flux-realism',
    base_provider = 'Flux AI',
    best_provider = IterListProvider([Airforce, AmigoChat])
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

### OpenAI ###
dall_e_3 = Model(
    name = 'dall-e-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Airforce, CopilotAccount, OpenaiAccount, MicrosoftDesigner, BingCreateImages])
)

### Recraft ###
recraft_v3 = Model(
    name = 'recraft-v3',
    base_provider = 'Recraft',
    best_provider = AmigoChat
)

### Other ###
any_dark = Model(
    name = 'any-dark',
    base_provider = 'Other',
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
        
        # o1
        'o1-preview': o1_preview,
        'o1-mini': o1_mini,

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
        'llama-3.2-90b': llama_3_2_90b,
        
        # CodeLlama
        'codellama-34b': codellama_34b,
                
        ### Mistral ###
        'mixtral-7b': mixtral_7b,
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-tiny': mistral_tiny,
        'mistral-nemo': mistral_nemo,

        ### NousResearch ###
        'mixtral-8x7b-dpo': mixtral_8x7b_dpo,
        'hermes-2-dpo': hermes_2_dpo,
        'hermes-2-pro': hermes_2_pro,
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
        # claude 3
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'claude-3-haiku': claude_3_haiku,

        # claude 3.5
        'claude-3.5-sonnet': claude_3_5_sonnet,
        'claude-3.5-haiku': claude_3_5_haiku,

        ### Reka AI ###
        'reka-core': reka_core,

        ### Blackbox AI ###
        'blackboxai': blackboxai,
        'blackboxai-pro': blackboxai_pro,

        ### CohereForAI ###
        'command-r+': command_r_plus,

        ### GigaChat ###
        'gigachat': gigachat,

        ### Qwen ###
        # qwen 1_5
        'qwen-1.5-7b': qwen_1_5_7b,
        
        # qwen 2
        'qwen-2-72b': qwen_2_72b,
        
        # qwen 2.5
        'qwen-2.5-72b': qwen_2_5_72b,
        'qwen-2.5-coder-32b': qwen_2_5_coder_32b,
        'qwq-32b': qwq_32b,
                
        ### Upstage ###
        'solar-mini': solar_mini,
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
        
        ### DeepSeek ###
        'deepseek-chat': deepseek_chat,
        'deepseek-coder': deepseek_coder,

        ### TheBloke ###   
        'german-7b': german_7b,

        ### Nvidia ###   
        'nemotron-70b': nemotron_70b,
        
        ### Teknium ###   
        'openhermes-2.5': openhermes_2_5,
        
        ### Liquid ### 
        'lfm-40b': lfm_40b,
        
        ### databricks ###   
        'dbrx-instruct': dbrx_instruct,
        
        ### anthracite-org ###   
        'magnum-72b': magnum_72b,
        
        ### anthracite-org ###   
        'jamba-mini': jamba_mini,
        
        ### HuggingFaceH4 ###   
        'zephyr-7b': zephyr_7b,
        
        ### Inferless ###   
        'neural-7b': neural_7b,
        
        ### Gryphe ###   
        'mythomax-13b': mythomax_13b,
        
        ### llmplayground.net ###   
        'any-uncensored': any_uncensored,

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
        'flux-dev': flux_dev,
        'flux-realism': flux_realism,
        'flux-anime': flux_anime,
        'flux-3d': flux_3d,
        'flux-disney': flux_disney,
        'flux-pixel': flux_pixel,
        'flux-4o': flux_4o,

        ### OpenAI ###
        'dall-e-3': dall_e_3,

        ### Recraft ###
        'recraft-v3': recraft_v3,
        
        ### Other ###
        'any-dark': any_dark,
    }

_all_models = list(ModelUtils.convert.keys())
