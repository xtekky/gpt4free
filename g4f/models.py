from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    ### No Auth Required ###
    ARTA,
    Blackbox,
    Chatai,
    ChatGLM,
    Cloudflare,
    Copilot,
    DDG,
    DeepInfraChat,
    DocsBot,
    Dynaspark,
    Free2GPT,
    FreeGpt,
    HuggingSpace,
    Grok,
    DeepseekAI_JanusPro7b,
    ImageLabs,
    LambdaChat,
    Liaobots,
    OIVSCodeSer2,
    OIVSCodeSer5,
    OIVSCodeSer0501,
    OpenAIFM,
    PerplexityLabs,
    Pi,
    PollinationsAI,
    PollinationsImage,
    PuterJS,
    TeachAnything,
    Websim,
    WeWordle,
    Yqcloud,
    
    ### Needs Auth ###
    BingCreateImages,
    CopilotAccount,
    Gemini,
    GeminiPro,
    HailuoAI,
    HuggingChat,
    HuggingFace,
    HuggingFaceAPI,
    MetaAI,
    MicrosoftDesigner,
    OpenaiAccount,
    OpenaiChat,
    Reka,
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

class ImageModel(Model):
    pass

class AudioModel(Model):
    pass
    
class VideoModel(Model):
    pass
    
class VisionModel(Model):
    pass

### Default ###
default = Model(
    name = "",
    base_provider = "",
    best_provider = IterListProvider([
        Blackbox,
        DDG,
        Copilot,
        OIVSCodeSer2,
        OIVSCodeSer5,
        OIVSCodeSer0501,
        DeepInfraChat,
        LambdaChat,
        PollinationsAI,
        PuterJS,
        Free2GPT,
        FreeGpt,
        Dynaspark,
        Chatai,
        WeWordle,
        DocsBot,
        OpenaiChat,
        Cloudflare,
    ])
)

default_vision = VisionModel(
    name = "",
    base_provider = "",
    best_provider = IterListProvider([
        Blackbox,
        DeepInfraChat,
        OIVSCodeSer2,
        OIVSCodeSer5,
        OIVSCodeSer0501,
        PollinationsAI,
        Dynaspark,
        PuterJS,
        DocsBot,
        HuggingSpace,
        GeminiPro,
        HuggingFaceAPI,
        CopilotAccount,
        OpenaiAccount,
        Gemini,
    ], shuffle=False)
)

##########################
### Text//Audio/Vision ###
##########################

### OpenAI ###
# gpt-3.5
gpt_3_5_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, DDG, PollinationsAI, Copilot, Yqcloud, PuterJS, WeWordle, Liaobots, OpenaiChat])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PuterJS, Liaobots])
)

# gpt-4o
gpt_4o = VisionModel(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, PollinationsAI, PuterJS, DocsBot, Liaobots, OpenaiChat])
)

gpt_4o_search = Model(
    name          = 'gpt-4o-search',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, DDG, OIVSCodeSer2, PollinationsAI, PuterJS, Chatai, Liaobots, OpenaiChat])
)

gpt_4o_mini_search = Model(
    name          = 'gpt-4o-mini-search',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

gpt_4o_mini_audio = AudioModel(
    name          = 'gpt-4o-mini-audio',
    base_provider = 'OpenAI',
    best_provider = PollinationsAI
)

gpt_4o_mini_tts = AudioModel(
    name          = 'gpt-4o-mini-tts',
    base_provider = 'OpenAI',
    best_provider = OpenAIFM
)

# o1
o1 = Model(
    name          = 'o1',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Copilot, PuterJS, OpenaiAccount])
)

o1_mini = Model(
    name          = 'o1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PuterJS, OpenaiAccount])
)

o1_pro = Model(
    name          = 'o1-pro',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

# o3
o3 = Model(
    name          = 'o3',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

o3_mini = Model(
    name          = 'o3-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, Blackbox, PuterJS, Liaobots])
)

o3_mini_high = Model(
    name          = 'o3-mini-high',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

# o4
o4_mini = Model(
    name          = 'o4-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsAI, PuterJS])
)

o4_mini_high = Model(
    name          = 'o4-mini-high',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

# gpt-4.1
gpt_4_1 = Model(
    name          = 'gpt-4.1',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsAI, PuterJS, Liaobots])
)

gpt_4_1_mini = Model(
    name          = 'gpt-4.1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([OIVSCodeSer5, OIVSCodeSer0501, PollinationsAI, PuterJS, Liaobots])
)

gpt_4_1_nano = Model(
    name          = 'gpt-4.1-nano',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, PollinationsAI, PuterJS])
)

gpt_4_5 = Model(
    name          = 'gpt-4.5',
    base_provider = 'OpenAI',
    best_provider = PuterJS
)

# dall-e
dall_e_3 = ImageModel(
    name = 'dall-e-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsImage, CopilotAccount, OpenaiAccount, MicrosoftDesigner, BingCreateImages])
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

llama_2_70b = Model(
    name          = "llama-2-70b",
    base_provider = "Meta Llama",
    best_provider = PuterJS
)

# llama-3
llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([PuterJS, Cloudflare])
)

llama_3_70b = Model(
    name          = "",
    base_provider = "Meta Llama",
    best_provider = PuterJS
)


# llama-3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PuterJS, Cloudflare])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = PuterJS
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = PuterJS
)

# llama-3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, PuterJS, Cloudflare])
)

llama_3_2_3b = Model(
    name          = "llama-3.2-3b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

llama_3_2_11b = VisionModel(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, PuterJS, HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)

# llama-3.3
llama_3_3_8b = Model(
    name          = "llama-3.3-8b",
    base_provider = "Meta Llama",
    best_provider = PuterJS
)

llama_3_3_70b = Model(
    name          = "llama-3.3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat, LambdaChat, PollinationsAI, PuterJS, HuggingChat, HuggingFace])
)

# llama-4
llama_4_scout = Model(
    name          = "llama-4-scout",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, PollinationsAI, PuterJS, Cloudflare])
)

llama_4_scout_17b = Model(
    name          = "llama-4-scout-17b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI])
)

llama_4_maverick = Model(
    name          = "llama-4-maverick",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PuterJS])
)

llama_4_maverick_17b = Model(
    name          = "llama-4-maverick-17b",
    base_provider = "Meta Llama",
    best_provider = DeepInfraChat
)

### MistralAI ###
ministral_3b = Model(
    name          = "ministral-3b",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

ministral_8b = Model(
    name          = "ministral-8b",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mistral_7b = Model(
    name          = "mistral-7b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mixtral_8x22b = Model(
    name          = "mixtral-8x22b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)

pixtral_12b = Model(
    name          = "pixtral-12b",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mistral_tiny = Model(
    name          = "mistral-tiny",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mistral_saba = Model(
    name          = "mistral-saba",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

pixtral_large = Model(
    name          = "pixtral-large",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

codestral = Model(
    name          = "codestral",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mistral_large = Model(
    name          = "mistral-large",
    base_provider = "Mistral AI",
    best_provider = PuterJS
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, PuterJS, HuggingChat, HuggingFace])
)

mistral_small = Model(
    name          = "mistral-small",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat, PuterJS])
)

mistral_small_24b = Model(
    name          = "mistral-small-24b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat])
)

mistral_small_3_1_24b = Model(
    name          = "mistral-small-3.1-24b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, PollinationsAI])
)

### NousResearch ###
# hermes-2
hermes_2_dpo = Model(
    name          = "hermes-2-dpo",
    base_provider = "NousResearch",
    best_provider = PuterJS
)

hermes_2_pro = Model(
    name          = "hermes-2-pro",
    base_provider = "NousResearch",
    best_provider = PuterJS
)

# hermes-3
hermes_3_70b = Model(
    name          = "hermes-3-70b",
    base_provider = "NousResearch",
    best_provider = PuterJS
)

hermes_3_405b = Model(
    name          = "hermes-3-405b",
    base_provider = "NousResearch",
    best_provider = IterListProvider([LambdaChat, PuterJS])
)

# deephermes-3
deephermes_3_8b = Model(
    name          = "deephermes-3-8b",
    base_provider = "NousResearch",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

deephermes_3_24b = Model(
    name          = "deephermes-3-24b",
    base_provider = "NousResearch",
    best_provider = PuterJS
)

### Microsoft ###
# phi-3
phi_3_mini = Model(
    name          = "phi-3-mini",
    base_provider = "Microsoft",
    best_provider = PuterJS
)

phi_3_5_mini = Model(
    name          = "phi-3.5-mini",
    base_provider = "Microsoft",
    best_provider = IterListProvider([PuterJS, HuggingChat])
)

# phi-4
phi_4 = Model(
    name          = "phi-4",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, PuterJS, HuggingSpace])
)

phi_4_multimodal = VisionModel(
    name          = "phi-4-multimodal",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, PuterJS, HuggingSpace])
)

phi_4_reasoning = Model(
    name          = "phi-4-reasoning",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)

phi_4_reasoning_plus = Model(
    name          = "phi-4-reasoning-plus",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)


# wizardlm
wizardlm_2_7b = Model(
    name = 'wizardlm-2-7b',
    base_provider = 'Microsoft',
    best_provider = DeepInfraChat
)

wizardlm_2_8x22b = Model(
    name = 'wizardlm-2-8x22b',
    base_provider = 'Microsoft',
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)

# mai-ds
mai_ds_r1 = Model(
    name = 'mai-ds-r1',
    base_provider = 'Microsoft',
    best_provider = PuterJS
)

### Google DeepMind ###
# gemini
gemini = Model(
    name          = 'gemini-2.0',
    base_provider = 'Google',
    best_provider = Gemini
)

# gemini-1.0
gemini_1_0_pro = Model(
    name          = 'gemini-1.0-pro',
    base_provider = 'Google',
    best_provider = Liaobots
)

# gemini-1.5
gemini_1_5_flash = Model(
    name          = 'gemini-1.5-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Free2GPT, FreeGpt, TeachAnything, Websim, Dynaspark, PuterJS, GeminiPro])
)

gemini_1_5_8b_flash = Model(
    name          = 'gemini-1.5-8b-flash',
    base_provider = 'Google',
    best_provider = PuterJS
)

gemini_1_5_pro = Model(
    name          = 'gemini-1.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([Free2GPT, FreeGpt, TeachAnything, Websim, PuterJS, GeminiPro])
)

# gemini-2.0
gemini_2_0_flash = Model(
    name          = 'gemini-2.0-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, Dynaspark, PuterJS, Liaobots, GeminiPro, Gemini])
)

gemini_2_0_flash_thinking = Model(
    name          = 'gemini-2.0-flash-thinking',
    base_provider = 'Google',
    best_provider = IterListProvider([PollinationsAI, Liaobots, Gemini])
)

gemini_2_0_flash_thinking_with_apps = Model(
    name          = 'gemini-2.0-flash-thinking-with-apps',
    base_provider = 'Google',
    best_provider = Gemini
)

# gemini-2.5
gemini_2_5_flash = Model(
    name          = 'gemini-2.5-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([PollinationsAI, PuterJS, Gemini])
)

gemini_2_5_flash_thinking = Model(
    name          = 'gemini-2.5-flash-thinking',
    base_provider = 'Google',
    best_provider = PuterJS
)

gemini_2_5_pro = Model(
    name          = 'gemini-2.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([PuterJS, Liaobots, Gemini])
)

# gemma-2
gemma_2_9b = Model(
    name          = 'gemma-2-9b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

gemma_2_27b = Model(
    name          = 'gemma-2-27b',
    base_provider = 'Google',
    best_provider = PuterJS
)

# gemma-3
gemma_3_1b = Model(
    name          = 'gemma-3-1b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

gemma_3_4b = Model(
    name          = 'gemma-3-4b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

gemma_3_12b = Model(
    name          = 'gemma-3-12b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PuterJS])
)

gemma_3_27b = Model(
    name          = 'gemma-3-27b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PuterJS])
)

### Anthropic ###
# claude 2
claude_2 = Model(
    name          = 'claude-2',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

# claude 2.0
claude_2_0 = Model(
    name          = 'claude-2.0',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

# claude 2.1
claude_2_1 = Model(
    name          = 'claude-2.1',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

# claude 3
claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, PuterJS])
)

# claude 3.5
claude_3_5_sonnet = Model(
    name          = 'claude-3.5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, PuterJS, Liaobots])
)

claude_3_5_haiku = Model(
    name          = 'claude-3.5-haiku',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

# claude 3.7
claude_3_7_sonnet = Model(
    name          = 'claude-3.7-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, PuterJS, Liaobots])
)

claude_3_7_sonnet_thinking = Model(
    name          = 'claude-3.7-sonnet-thinking',
    base_provider = 'Anthropic',
    best_provider = PuterJS
)

### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = Reka
)

reka_flash = Model(
    name = 'reka-flash',
    base_provider = 'Reka AI',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

### Blackbox AI ###
blackboxai = Model(
    name = 'blackboxai',
    base_provider = 'Blackbox AI',
    best_provider = Blackbox
)

### CohereForAI ###
command = Model(
    name = 'command',
    base_provider = 'CohereForAI',
    best_provider = PuterJS
)

command_r = Model(
    name = 'command-r',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([PollinationsAI, HuggingSpace, PuterJS])
)

command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, PuterJS, HuggingChat])
)

command_r7b = Model(
    name = 'command-r7b',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

command_a = Model(
    name = 'command-a',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

### Qwen ###
# qwen-1.5
qwen_1_5_7b = Model(
    name = 'qwen-1.5-7b',
    base_provider = 'Qwen',
    best_provider = Cloudflare
)

# qwen-2
qwen_2_72b = Model(
    name = 'qwen-2-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, PuterJS])
)
qwen_2_vl_7b = VisionModel(
    name = "qwen-2-vl-7b",
    base_provider = 'Qwen',
    best_provider = HuggingFaceAPI
)

# qwen-2.5
qwen_2_5 = Model(
    name = 'qwen-2.5',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_7b = Model(
    name = 'qwen-2.5-7b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

qwen_2_5_72b = Model(
    name = 'qwen-2.5-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PollinationsAI, LambdaChat, PuterJS, HuggingChat])
)

qwen_2_5_coder_7b = Model(
    name = 'qwen-2.5-coder-7b',
    base_provider = 'Qwen',
    best_provider = PuterJS
)
qwen_2_5_1m = Model(
    name = 'qwen-2.5-1m',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_max = Model(
    name = 'qwen-2-5-max',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_vl_3b = Model(
    name = 'qwen-2.5-vl-3b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

qwen_2_5_vl_7b = Model(
    name = 'qwen-2.5-vl-7b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

qwen_2_5_vl_32b = Model(
    name = 'qwen-2.5-vl-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

qwen_2_5_vl_72b = Model(
    name = 'qwen-2.5-vl-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

# qwen3
qwen_3_235b = Model(
    name = 'qwen-3-235b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, PuterJS, Liaobots])
)

qwen_3_32b = Model(
    name = 'qwen-3-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, PuterJS])
)

qwen_3_30b = Model(
    name = 'qwen-3-30b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, PuterJS])
)

qwen_3_14b = Model(
    name = 'qwen-3-14b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, PuterJS])
)

qwen_3_8b = Model(
    name = 'qwen-3-8b',
    base_provider = 'Qwen',
    best_provider = PuterJS
)

qwen_3_4b = Model(
    name = 'qwen-3-4b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

qwen_3_1_7b = Model(
    name = 'qwen-3-1.7b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

qwen_3_0_6b = Model(
    name = 'qwen-3-0.6b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([HuggingSpace, PuterJS])
)

### qwq/qvq ###
qwq_32b = Model(
    name = 'qwq-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI, PuterJS, HuggingChat])
)

qwq_32b_preview = Model(
    name = 'qwq-32b-preview',
    base_provider = 'Qwen',
    best_provider = Blackbox
)

qwq_32b_arliai = Model(
    name = 'qwq-32b-arliai',
    base_provider = 'Qwen',
    best_provider = Blackbox
)

qvq_72b = VisionModel(
    name = 'qvq-72b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

# qwen-vl
qwen_vl_plus = Model(
    name = 'qwen-vl-plus',
    base_provider = 'Qwen',
    best_provider = PuterJS
)

qwen_vl_max = Model(
    name = 'qwen-vl-max',
    base_provider = 'Qwen',
    best_provider = PuterJS
)

# qwen
qwen_turbo = Model(
    name = 'qwen-turbo',
    base_provider = 'Qwen',
    best_provider = PuterJS
)

qwen_plus = Model(
    name = 'qwen-plus',
    base_provider = 'Qwen',
    best_provider = PuterJS
)

qwen_max = Model(
    name = 'qwen-max',
    base_provider = 'Qwen',
    best_provider = PuterJS
)


### Inflection ###
pi = Model(
    name = 'pi',
    base_provider = 'Inflection',
    best_provider = Pi
)

inflection_3_productivity = Model(
    name = 'inflection-3-productivity',
    base_provider = 'Inflection',
    best_provider = PuterJS
)

inflection_3_pi = Model(
    name = 'inflection-3-pi',
    base_provider = 'Inflection',
    best_provider = PuterJS
)

### DeepSeek ###
# deepseek
deepseek_chat = Model(
    name = 'deepseek-chat',
    base_provider = 'DeepSeek',
    best_provider = PuterJS
)

deepseek_coder = Model(
    name = 'deepseek-coder',
    base_provider = 'DeepSeek',
    best_provider = PuterJS
)

# deepseek-v3
deepseek_v3 = Model(
    name = 'deepseek-v3',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, PuterJS, Liaobots])
)

# deepseek-r1
deepseek_r1 = Model(
    name = 'deepseek-r1',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LambdaChat, PollinationsAI, PuterJS, HuggingChat, HuggingFace])
)

deepseek_r1_zero = Model(
    name = 'deepseek-r1-zero',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

deepseek_r1_turbo = Model(
    name = 'deepseek-r1-turbo',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

deepseek_r1_distill_llama_8b = Model(
    name = 'deepseek-r1-distill-llama-70b',
    base_provider = 'DeepSeek',
    best_provider = PuterJS
)

deepseek_r1_distill_llama_70b = Model(
    name = 'deepseek-r1-distill-llama-70b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI])
)

deepseek_r1_distill_qwen_1_5b = Model(
    name = 'deepseek-r1-distill-qwen-1.5b',
    base_provider = 'DeepSeek',
    best_provider = PuterJS
)

deepseek_r1_distill_qwen_14b = Model(
    name = 'deepseek-r1-distill-qwen-14b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

deepseek_r1_distill_qwen_32b = Model(
    name = 'deepseek-r1-distill-qwen-32b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI, PuterJS])
)

# deepseek-v2
deepseek_prover_v2 = Model(
    name = 'deepseek-prover-v2',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, PuterJS])
)

deepseek_prover_v2_671b = Model(
    name = 'deepseek-prover-v2-671b',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

# deepseek-v3-0324
deepseek_v3_0324 = Model(
    name = 'deepseek-v3-0324',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, PuterJS])
)

# janus
janus_pro_7b = VisionModel(
    name = DeepseekAI_JanusPro7b.default_model,
    base_provider = 'DeepSeek',
    best_provider = DeepseekAI_JanusPro7b
)

### x.ai ###
grok = Model(
    name = 'grok',
    base_provider = 'x.ai',
    best_provider = PuterJS
)

grok_2 = Model(
    name = 'grok-2',
    base_provider = 'x.ai',
    best_provider = IterListProvider([PuterJS, Grok, Liaobots])
)

grok_3 = Model(
    name = 'grok-3',
    base_provider = 'x.ai',
    best_provider = IterListProvider([Grok, Liaobots])
)

grok_3_mini = Model(
    name = 'grok-3-mini',
    base_provider = 'x.ai',
    best_provider = PuterJS
)

grok_3_r1 = Model(
    name = 'grok-3-r1',
    base_provider = 'x.ai',
    best_provider = Grok
)

grok_3_reason = Model(
    name = 'grok-3-reason',
    base_provider = 'x.ai',
    best_provider = Liaobots
)

grok_3_beta = Model(
    name = 'grok-3-beta',
    base_provider = 'x.ai',
    best_provider = PuterJS
)

grok_beta = Model(
    name = 'grok-beta',
    base_provider = 'x.ai',
    best_provider = PuterJS
)

### Perplexity AI ### 
sonar = Model(
    name = 'sonar',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PuterJS, PerplexityLabs])
)

sonar_pro = Model(
    name = 'sonar-pro',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PuterJS, PerplexityLabs])
)

sonar_reasoning = Model(
    name = 'sonar-reasoning',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PuterJS, PerplexityLabs])
)

sonar_reasoning_pro = Model(
    name = 'sonar-reasoning-pro',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PuterJS, PerplexityLabs])
)

sonar_deep_research = Model(
    name = 'sonar-deep-research',
    base_provider = 'Perplexity AI',
    best_provider = PuterJS
)

r1_1776 = Model(
    name = 'r1-1776',
    base_provider = 'Perplexity AI',
    best_provider = IterListProvider([PuterJS, PerplexityLabs])
)

llama_3_1_sonar_small_online = Model(
    name = 'llama-3.1-sonar-small-online',
    base_provider = 'Perplexity AI',
    best_provider = PuterJS
)

llama_3_1_sonar_large_online = Model(
    name = 'llama-3.1-sonar-small-online',
    base_provider = 'Perplexity AI',
    best_provider = PuterJS
)

### Nvidia ### 
nemotron_49b = Model(
    name = 'nemotron-49b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

nemotron_70b = Model(
    name = 'nemotron-70b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([LambdaChat, PuterJS, HuggingChat, HuggingFace])
)

nemotron_253b = Model(
    name = 'nemotron-253b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

### THUDM ### 
glm_4 = Model(
    name = 'glm-4',
    base_provider = 'THUDM',
    best_provider = IterListProvider([ChatGLM, PuterJS])
)

glm_4_32b = Model(
    name = 'glm-4-32b',
    base_provider = 'THUDM',
    best_provider = PuterJS
)

glm_z1_32b = Model(
    name = 'glm-z1-32b',
    base_provider = 'THUDM',
    best_provider = PuterJS
)

glm_4_9b = Model(
    name = 'glm-4-9b',
    base_provider = 'THUDM',
    best_provider = PuterJS
)

glm_z1_9b = Model(
    name = 'glm-z1-9b',
    base_provider = 'THUDM',
    best_provider = PuterJS
)

glm_z1_rumination_32b = Model(
    name = 'glm-z1-rumination-32b',
    base_provider = 'THUDM',
    best_provider = PuterJS
)

### MiniMax ###
mini_max = Model(
    name = "minimax",
    base_provider = "MiniMax",
    best_provider = IterListProvider([PuterJS, HailuoAI])
)

### Cognitive Computations ###
# dolphin-2
dolphin_2_6 = Model(
    name = "dolphin-2.6",
    base_provider = "Cognitive Computations",
    best_provider = DeepInfraChat
)

dolphin_2_9 = Model(
    name = "dolphin-2.9",
    base_provider = "Cognitive Computations",
    best_provider = DeepInfraChat
)

# dolphin-3
dolphin_3_0_24b = Model(
    name = "dolphin-3.0-24b",
    base_provider = "Cognitive Computations",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

dolphin_3_0_r1_24b = Model(
    name = "dolphin-3.0-r1-24b",
    base_provider = "Cognitive Computations",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

dolphin_8x22b = Model(
    name = "dolphin-8x22b",
    base_provider = "Cognitive Computations",
    best_provider = PuterJS
)

### DeepInfra ###
airoboros_70b = Model(
    name = "airoboros-70b",
    base_provider = "DeepInfra",
    best_provider = DeepInfraChat
)

### Lizpreciatior ###
lzlv_70b = Model(
    name = "lzlv-70b",
    base_provider = "Lizpreciatior",
    best_provider = DeepInfraChat
)

### Ai2 ###
molmo_7b = Model(
    name = "molmo-7b",
    base_provider = "Ai2",
    best_provider = Blackbox
)

### Liquid AI ###
lfm_3b = Model(
    name = "lfm-3b",
    base_provider = "Liquid AI",
    best_provider = PuterJS
)

lfm_7b = Model(
    name = "lfm-7b",
    base_provider = "Liquid AI",
    best_provider = PuterJS
)

lfm_40b = Model(
    name = "lfm-40b",
    base_provider = "Liquid AI",
    best_provider = IterListProvider([LambdaChat, PuterJS])
)

### Agentica ###
deepcode_14b = Model(
    name = "deepcoder-14b",
    base_provider = "Agentica",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

### Moonshot AI ###
kimi_vl_thinking = Model(
    name = "kimi-vl-thinking",
    base_provider = "Moonshot AI",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

moonlight_16b = Model(
    name = "moonlight-16b",
    base_provider = "Moonshot AI",
    best_provider = IterListProvider([Blackbox, PuterJS])
)

### Featherless Serverless LLM ### 
qwerky_72b = Model(
    name = 'qwerky-72b',
    base_provider = 'Featherless Serverless LLM',
    best_provider = IterListProvider([Blackbox, PuterJS])
)

### Uncensored AI ### 
evil = Model(
    name = 'evil',
    base_provider = 'Evil Mode - Experimental',
    best_provider = PollinationsAI
)

### Stability AI ###
sdxl_turbo = ImageModel(
    name = 'sdxl-turbo',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([PollinationsImage, ImageLabs])
)

sd_3_5 = ImageModel(
    name = 'stable-diffusion-3.5-large',
    base_provider = 'Stability AI',
    best_provider = HuggingSpace
)

### Black Forest Labs ###
flux = ImageModel(
    name = 'flux',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, Websim, HuggingSpace, ARTA])
)

flux_pro = ImageModel(
    name = 'flux-pro',
    base_provider = 'Black Forest Labs',
    best_provider = PollinationsImage
)

flux_dev = ImageModel(
    name = 'flux-dev',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, HuggingSpace, HuggingChat, HuggingFace])
)

flux_schnell = ImageModel(
    name = 'flux-schnell',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, HuggingSpace, HuggingChat, HuggingFace])
)

### Midjourney ###
midjourney = ImageModel(
    name = 'midjourney',
    base_provider = 'Midjourney',
    best_provider = PollinationsImage
)

class ModelUtils:
    """
    Utility class for mapping string identifiers to Model instances.

    Attributes:
        convert (dict[str, Model]): Dictionary mapping model string identifiers to Model instances.
    """
    convert: dict[str, Model] = { 
        ### OpenAI ###       
        # gpt-3.5
        gpt_3_5_turbo.name: gpt_3_5_turbo,
        
        # gpt-4
        gpt_4.name: gpt_4,
        gpt_4_turbo.name: gpt_4_turbo,
        
        # gpt-4o
        gpt_4o.name: gpt_4o,
        gpt_4o_search.name: gpt_4o_search,
        gpt_4o_mini.name: gpt_4o_mini,
        gpt_4o_mini_search.name: gpt_4o_mini_search,
        gpt_4o_mini_audio.name: gpt_4o_mini_audio,
        gpt_4o_mini_tts.name: gpt_4o_mini_tts,
        
        # o1
        o1.name: o1,
        o1_mini.name: o1_mini,
        o1_pro.name: o1_pro,
        
        # o3
        o3.name: o3,
        o3_mini.name: o3_mini,
        o3_mini_high.name: o3_mini_high,
        
        # o4
        o4_mini.name: o4_mini,
        o4_mini_high.name: o4_mini_high,

        # gpt-4.1
        gpt_4_1.name: gpt_4_1,
        gpt_4_1_nano.name: gpt_4_1_nano,
        gpt_4_1_mini.name: gpt_4_1_mini,
        
        # gpt-4.5
        gpt_4_5.name: gpt_4_5,
        
        # dall-e
        dall_e_3.name: dall_e_3,

        ### Meta ###
        meta.name: meta,

        # llama-2
        llama_2_7b.name: llama_2_7b,
        llama_2_70b.name: llama_2_70b,

        # llama-3
        llama_3_8b.name: llama_3_8b,
        llama_3_70b.name: llama_3_70b,
                
        # llama-3.1
        llama_3_1_8b.name: llama_3_1_8b,
        llama_3_1_70b.name: llama_3_1_70b,
        llama_3_1_405b.name: llama_3_1_405b,

        # llama-3.2
        llama_3_2_1b.name: llama_3_2_1b,
        llama_3_2_3b.name: llama_3_2_3b,
        llama_3_2_11b.name: llama_3_2_11b,
        llama_3_2_90b.name: llama_3_2_90b,
        
        # llama-3.3
        llama_3_3_8b.name: llama_3_3_8b,
        llama_3_3_70b.name: llama_3_3_70b,

        # llama-4
        llama_4_scout.name: llama_4_scout,
        llama_4_scout_17b.name: llama_4_scout_17b,
        llama_4_maverick.name: llama_4_maverick,
        llama_4_maverick_17b.name: llama_4_maverick_17b,
                
        ### Mistral ###
        ministral_3b.name: ministral_3b,
        ministral_8b.name: ministral_8b,
        mistral_7b.name: mistral_7b,
        mixtral_8x7b.name: mixtral_8x7b,
        mixtral_8x22b.name: mixtral_8x22b,
        pixtral_12b.name: pixtral_12b,
        pixtral_large.name: pixtral_large,
        mistral_tiny.name: mistral_tiny,
        mistral_saba.name: mistral_saba,
        mistral_large.name: mistral_large,
        codestral.name: codestral,
        mistral_nemo.name: mistral_nemo,
        mistral_small.name: mistral_small,
        mistral_small_24b.name: mistral_small_24b,
        mistral_small_3_1_24b.name: mistral_small_3_1_24b,

        ### NousResearch ###
        # hermes-2
        hermes_2_dpo.name: hermes_2_dpo,
        hermes_2_pro.name: hermes_2_pro,
        
        # hermes-3
        hermes_3_70b.name: hermes_3_70b,
        hermes_3_405b.name: hermes_3_405b,
        
         # deephermes-3
        deephermes_3_8b.name: deephermes_3_8b,
        deephermes_3_24b.name: deephermes_3_24b,
                
        ### Microsoft ###
        # phi-3
        phi_3_mini.name: phi_3_mini,
        phi_3_5_mini.name: phi_3_5_mini,
        
        # phi-4
        phi_4.name: phi_4,
        phi_4_multimodal.name: phi_4_multimodal,
        phi_4_reasoning_plus.name: phi_4_reasoning_plus,
        
        # wizardlm
        wizardlm_2_7b.name: wizardlm_2_7b,
        wizardlm_2_8x22b.name: wizardlm_2_8x22b,
        
        # mai-ds
        mai_ds_r1.name: mai_ds_r1,

        ### Google ###
        ### gemini
        "gemini": gemini,
        gemini.name: gemini,
        
        # gemini-1.0
        gemini_1_0_pro.name: gemini_1_0_pro,
        
        # gemini-1.5
        gemini_1_5_pro.name: gemini_1_5_pro,
        gemini_1_5_8b_flash.name: gemini_1_5_8b_flash,
        gemini_1_5_flash.name: gemini_1_5_flash,
        
        # gemini-2.0
        gemini_2_0_flash.name: gemini_2_0_flash,
        gemini_2_0_flash_thinking.name: gemini_2_0_flash_thinking,
        gemini_2_0_flash_thinking_with_apps.name: gemini_2_0_flash_thinking_with_apps,
        
        # gemini-2.5
        gemini_2_5_flash.name: gemini_2_5_flash,
        gemini_2_5_flash_thinking.name: gemini_2_5_flash_thinking,
        gemini_2_5_pro.name: gemini_2_5_pro,
        
        # gemma-2
        gemma_2_9b.name: gemma_2_9b,
        gemma_2_27b.name: gemma_2_27b,
        # gemma-3
        gemma_3_12b.name: gemma_3_12b,
        gemma_3_1b.name: gemma_3_1b,
        gemma_3_27b.name: gemma_3_27b,
        gemma_3_4b.name: gemma_3_4b,

        ### Anthropic ###
        # claude 2
        claude_2.name: claude_2,
        
        # claude-2.0
        claude_2_0.name: claude_2_0,
        
        # claude-2.1
        claude_2_1.name: claude_2_1,
        
        # claude 3
        claude_3_opus.name: claude_3_opus,
        claude_3_sonnet.name: claude_3_sonnet,
        claude_3_haiku.name: claude_3_haiku,
        
        # claude 3.5
        claude_3_5_sonnet.name: claude_3_5_sonnet,
        claude_3_5_haiku.name: claude_3_5_haiku,
        
        # claude 3.7
        claude_3_7_sonnet.name: claude_3_7_sonnet,
        claude_3_7_sonnet_thinking.name: claude_3_7_sonnet_thinking,

        ### Reka AI ###
        reka_core.name: reka_core,
        reka_flash.name: reka_flash,

        ### Blackbox AI ###
        blackboxai.name: blackboxai,

        ### CohereForAI ###
        command.name: command,
        command_r.name: command_r,
        command_r_plus.name: command_r_plus,
        command_r7b.name: command_r7b,
        command_a.name: command_a,

        ### Qwen ###
        # qwen-1.5
        qwen_1_5_7b.name: qwen_1_5_7b,
        
        # qwen-2
        qwen_2_72b.name: qwen_2_72b,
        qwen_2_vl_7b.name: qwen_2_vl_7b,
        
        # qwen-2.5
        qwen_2_5.name: qwen_2_5,
        qwen_2_5_7b.name: qwen_2_5_7b,
        qwen_2_5_72b.name: qwen_2_5_72b,
        qwen_2_5_coder_32b.name: qwen_2_5_coder_32b,
        qwen_2_5_coder_7b.name: qwen_2_5_coder_7b,
        qwen_2_5_1m.name: qwen_2_5_1m,
        qwen_2_5_max.name: qwen_2_5_max,
        qwen_2_5_vl_3b.name: qwen_2_5_vl_3b,
        qwen_2_5_vl_7b.name: qwen_2_5_vl_7b,
        qwen_2_5_vl_32b.name: qwen_2_5_vl_32b,
        qwen_2_5_vl_72b.name: qwen_2_5_vl_72b,
        
        # qwen-3
        qwen_3_235b.name: qwen_3_235b,
        qwen_3_32b.name: qwen_3_32b,
        qwen_3_30b.name: qwen_3_30b,
        qwen_3_14b.name: qwen_3_14b,
        qwen_3_8b.name: qwen_3_8b,
        qwen_3_4b.name: qwen_3_4b,
        qwen_3_1_7b.name: qwen_3_1_7b,
        qwen_3_0_6b.name: qwen_3_0_6b,

        # qwq/qvq
        qwq_32b.name: qwq_32b,
        qwq_32b_preview.name: qwq_32b_preview,
        qwq_32b_arliai.name: qwq_32b_arliai,
        qvq_72b.name: qvq_72b,
        
        # qwen-vl
        qwen_vl_plus.name: qwen_vl_plus,
        qwen_vl_max.name: qwen_vl_max,
        
        # qwen
        qwen_turbo.name: qwen_turbo,
        qwen_plus.name: qwen_plus,
        qwen_max.name: qwen_max,

        ### Inflection ###
        pi.name: pi,
        inflection_3_productivity.name: inflection_3_productivity,
        inflection_3_pi.name: inflection_3_pi,

        ### x.ai ###
        grok.name: grok,
        grok_3.name: grok_3,
        grok_3_mini.name: grok_3_mini,
        grok_3_r1.name: grok_3_r1,
        grok_3_reason.name: grok_3_reason,
        grok_3_beta.name: grok_3_beta,
        grok_beta.name: grok_beta,

        ### Perplexity AI ###
        sonar.name: sonar,
        sonar_pro.name: sonar_pro,
        sonar_reasoning.name: sonar_reasoning,
        sonar_reasoning_pro.name: sonar_reasoning_pro,
        sonar_deep_research.name: sonar_deep_research,
        r1_1776.name: r1_1776,
        llama_3_1_sonar_small_online.name: llama_3_1_sonar_small_online,
        llama_3_1_sonar_large_online.name: llama_3_1_sonar_large_online,
        
        ### DeepSeek ###       
        # deepseek
        deepseek_chat.name: deepseek_chat,
        deepseek_coder.name: deepseek_coder,
        
        # deepseek-v3
        deepseek_v3.name: deepseek_v3,
        
        # deepseek-r1
        deepseek_r1.name: deepseek_r1,
        deepseek_r1_zero.name: deepseek_r1_zero,
        deepseek_r1_turbo.name: deepseek_r1_turbo,
        deepseek_r1_distill_llama_8b.name: deepseek_r1_distill_llama_8b,
        deepseek_r1_distill_qwen_1_5b.name: deepseek_r1_distill_qwen_1_5b,
        deepseek_r1_distill_qwen_14b.name: deepseek_r1_distill_qwen_14b,
        deepseek_r1_distill_qwen_32b.name: deepseek_r1_distill_qwen_32b,
        
        # deepseek-v2
        deepseek_prover_v2_671b.name: deepseek_prover_v2_671b,
        
        # deepseek-v3-0324
        deepseek_v3_0324.name: deepseek_v3_0324,

        ### Nvidia ###
        nemotron_49b.name: nemotron_49b,
        nemotron_70b.name: nemotron_70b,
        nemotron_253b.name: nemotron_253b,
        
        ### THUDM ###
        glm_4.name: glm_4,
        glm_4_32b.name: glm_4_32b,
        glm_z1_32b.name: glm_z1_32b,
        glm_4_9b.name: glm_4_9b,
        glm_z1_9b.name: glm_z1_9b,
        glm_z1_rumination_32b.name: glm_z1_rumination_32b,
        
        ### MiniMax ###
        mini_max.name: mini_max, 
        
        ### Cognitive Computations ###
        # dolphin-2
        dolphin_2_6.name: dolphin_2_6,
        dolphin_2_9.name: dolphin_2_9,
        
        # dolphin-3
        dolphin_3_0_24b.name: dolphin_3_0_24b,
        dolphin_3_0_r1_24b.name: dolphin_3_0_r1_24b,
        
        # dolphin-8x22b
        dolphin_8x22b.name: dolphin_3_0_r1_24b,
        
        ### DeepInfra ###
        airoboros_70b.name: airoboros_70b,
        
        ### Lizpreciatior ###
        lzlv_70b.name: lzlv_70b,

        ### Ai2 ###
        molmo_7b.name: molmo_7b,
        
        ### Liquid AI ###
        lfm_3b.name: lfm_3b,
        lfm_7b.name: lfm_7b,
        lfm_40b.name: lfm_40b,
        
        ### Agentica ###
        deepcode_14b.name: deepcode_14b,
        
        ### Moonshot AI ###
        kimi_vl_thinking.name: kimi_vl_thinking,
        moonlight_16b.name: moonlight_16b,
        
        ### Featherless Serverless LLM ###
        qwerky_72b.name: qwerky_72b,
        
        ### Uncensored AI ###
        evil.name: evil,

        ### Stability AI ###
        sdxl_turbo.name: sdxl_turbo,
        sd_3_5.name: sd_3_5,

        ### Flux AI ###
        flux.name: flux,
        flux_pro.name: flux_pro,
        flux_dev.name: flux_dev,
        flux_schnell.name: flux_schnell,
        
        ### Midjourney ###
        midjourney.name: midjourney,
    }


demo_models = {
    llama_3_2_11b.name: [llama_3_2_11b, [HuggingChat]],
    qwen_2_vl_7b.name: [qwen_2_vl_7b, [HuggingFaceAPI]],
    deepseek_r1.name: [deepseek_r1, [HuggingFace, PollinationsAI]],
    janus_pro_7b.name: [janus_pro_7b, [HuggingSpace]],
    command_r.name: [command_r, [HuggingSpace]],
    command_r_plus.name: [command_r_plus, [HuggingSpace]],
    command_r7b.name: [command_r7b, [HuggingSpace]],
    qwen_2_5_coder_32b.name: [qwen_2_5_coder_32b, [HuggingFace]],
    qwq_32b.name: [qwq_32b, [HuggingFace]],
    llama_3_3_70b.name: [llama_3_3_70b, [HuggingFace]],
    sd_3_5.name: [sd_3_5, [HuggingSpace, HuggingFace]],
    flux_dev.name: [flux_dev, [PollinationsImage, HuggingFace, HuggingSpace]],
    flux_schnell.name: [flux_schnell, [PollinationsImage, HuggingFace, HuggingSpace]],
}

# Create a list of all models and his providers
__models__  = {
    model.name: (model, providers)
        for model, providers in [
            (model, [provider for provider in model.best_provider.providers if provider.working]
                if isinstance(model.best_provider, IterListProvider)
                else [model.best_provider]
                if model.best_provider is not None and model.best_provider.working
                else [])
        for model in ModelUtils.convert.values()]
        if model.name and [True for provider in providers if provider.working]
    }
_all_models = list(__models__.keys())
