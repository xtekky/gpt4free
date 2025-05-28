from __future__ import annotations

import sys
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type

from .Provider import IterListProvider, ProviderType
from .Provider import (
    ### No Auth Required ###
    ARTA,
    Blackbox,
    Blackboxapi,
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
    DeepSeekAPI,
    ImageLabs,
    LambdaChat,
    LegacyLMArena,
    OIVSCodeSer2,
    OIVSCodeSer5,
    OIVSCodeSer0501,
    OpenAIFM,
    PerplexityLabs,
    Pi,
    PollinationsAI,
    PollinationsImage,
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

class ModelRegistry:
    """Central registry for all models with automatic discovery"""
    _models: Dict[str, 'Model'] = {}
    _aliases: Dict[str, str] = {}
    _discovered: bool = False
    
    @classmethod
    def register(cls, model: 'Model', aliases: List[str] = None):
        """Register a model and optional aliases"""
        if model.name:
            cls._models[model.name] = model
            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = model.name
    
    @classmethod
    def get(cls, name: str) -> Optional['Model']:
        """Get model by name or alias"""
        cls._ensure_discovered()
        if name in cls._models:
            return cls._models[name]
        if name in cls._aliases:
            return cls._models[cls._aliases[name]]
        return None
    
    @classmethod
    def all_models(cls) -> Dict[str, 'Model']:
        """Get all registered models"""
        cls._ensure_discovered()
        return cls._models.copy()
    
    @classmethod
    def _ensure_discovered(cls):
        """Ensure models have been discovered"""
        if not cls._discovered:
            cls._discover_models()
    
    @classmethod
    def _discover_models(cls):
        """Automatically discover all Model instances in current module"""
        if cls._discovered:
            return
            
        current_module = sys.modules[__name__]
        
        # Find all Model instances (not classes)
        for name in dir(current_module):
            if name.startswith('_'):
                continue
                
            obj = getattr(current_module, name)
            
            # Check if it's a Model instance (not a class)
            if isinstance(obj, Model) and not inspect.isclass(obj):
                cls.register(obj)
        
        # Register special aliases
        cls._aliases["gemini"] = "gemini-2.0"  # Special case for gemini
        
        cls._discovered = True
    
    @classmethod
    def refresh(cls):
        """Force refresh of model registry"""
        cls._models.clear()
        cls._aliases.clear()
        cls._discovered = False
        cls._discover_models()

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
    _registered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Auto-register model after initialization"""
        if not self._registered and self.name:
            ModelRegistry.register(self)
            self._registered = True

    @staticmethod
    def __all__() -> list[str]:
        """Returns a list of all model names."""
        return list(ModelRegistry.all_models().keys())

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
        OIVSCodeSer5,
        OIVSCodeSer0501,
        OIVSCodeSer2,
        Blackbox,
        Blackboxapi,
        DDG,
        Copilot,
        DeepInfraChat,
        LambdaChat,
        PollinationsAI,
        Free2GPT,
        FreeGpt,
        Dynaspark,
        Chatai,
        WeWordle,
        DocsBot,
        LegacyLMArena,
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
    best_provider = LegacyLMArena
)

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, DDG, PollinationsAI, Copilot, Yqcloud, WeWordle, LegacyLMArena, OpenaiChat])
)

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'OpenAI',
    best_provider = LegacyLMArena
)

# gpt-4o
gpt_4o = VisionModel(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, PollinationsAI, DocsBot, LegacyLMArena, OpenaiChat])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, DDG, OIVSCodeSer2, PollinationsAI, Chatai, LegacyLMArena, OpenaiChat])
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
    best_provider = IterListProvider([Copilot, LegacyLMArena, OpenaiAccount])
)

o1_mini = Model(
    name          = 'o1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([LegacyLMArena, OpenaiAccount])
)

# o3
o3 = Model(
    name          = 'o3',
    base_provider = 'OpenAI',
    best_provider = LegacyLMArena
)

o3_mini = Model(
    name          = 'o3-mini',
    base_provider = 'OpenAI',
    best_provider = LegacyLMArena
)

o3_mini_high = Model(
    name          = 'o3-mini-high',
    base_provider = 'OpenAI',
    best_provider = OpenaiAccount
)

# o4
o4_mini = Model(
    name          = 'o4-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsAI, LegacyLMArena, OpenaiChat])
)

o4_mini_high = Model(
    name          = 'o4-mini-high',
    base_provider = 'OpenAI',
    best_provider = OpenaiChat
)

# gpt-4.1
gpt_4_1 = Model(
    name          = 'gpt-4.1',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsAI, LegacyLMArena, OpenaiChat])
)

gpt_4_1_mini = Model(
    name          = 'gpt-4.1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([OIVSCodeSer5, OIVSCodeSer0501, PollinationsAI, LegacyLMArena])
)

gpt_4_1_nano = Model(
    name          = 'gpt-4.1-nano',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, PollinationsAI, LegacyLMArena])
)

gpt_4_5 = Model(
    name          = 'gpt-4.5',
    base_provider = 'OpenAI',
    best_provider = OpenaiChat
)

# dall-e
dall_e_3 = ImageModel(
    name = 'dall-e-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsImage, CopilotAccount, OpenaiAccount, MicrosoftDesigner, BingCreateImages])
)

gpt_image = ImageModel(
    name = 'gpt-image',
    base_provider = 'OpenAI',
    best_provider = PollinationsImage
)

### Meta ###
meta = Model(
    name          = "meta-ai",
    base_provider = "Meta",
    best_provider = MetaAI
)

# llama
llama_13b = Model(
    name          = "llama-13b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

# codellama
codellama_34b = Model(
    name          = "codellama-34b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

# llama 2
llama_2_7b = Model(
    name          = "llama-2-7b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([LegacyLMArena, Cloudflare])
)

llama_2_13b = Model(
    name          = "llama-2-13b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

llama_2_70b = Model(
    name          = "llama-2-70b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

# llama-3
llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([LegacyLMArena, Cloudflare])
)

llama_3_70b = Model(
    name          = "llama-3-70b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

# llama-3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LegacyLMArena, Cloudflare])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackboxapi, LegacyLMArena])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = LegacyLMArena
)

# llama-3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, LegacyLMArena, Cloudflare])
)

llama_3_2_3b = Model(
    name          = "llama-3.2-3b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

llama_3_2_11b = VisionModel(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = DeepInfraChat
)

# llama-3.3
llama_3_3_70b = Model(
    name          = "llama-3.3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat, LambdaChat, PollinationsAI, LegacyLMArena, HuggingChat, HuggingFace])
)

# llama-4
llama_4_scout = Model(
    name          = "llama-4-scout",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, PollinationsAI, LegacyLMArena, Cloudflare])
)

llama_4_maverick = Model(
    name          = "llama-4-maverick",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LegacyLMArena])
)

### MistralAI ###
mistral_7b = Model(
    name          = "mistral-7b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

mixtral_8x22b = Model(
    name          = "mixtral-8x22b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([DeepInfraChat, LegacyLMArena])
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, HuggingChat, HuggingFace])
)

mistral_small = Model(
    name          = "mistral-small",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat])
)

mistral_small_24b = Model(
    name          = "mistral-small-24b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, DDG, DeepInfraChat, LegacyLMArena])
)

mistral_small_3_1_24b = Model(
    name          = "mistral-small-3.1-24b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([Blackbox, PollinationsAI, LegacyLMArena])
)

mistral_large = Model(
    name          = "mistral-large",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

mistral_medium = Model(
    name          = "mistral-medium",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

mistral_next = Model(
    name          = "mistral-next",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

# pixtral
pixtral_large = Model(
    name          = "pixtral-large",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

# codestral
codestral = Model(
    name          = "codestral",
    base_provider = "Mistral AI",
    best_provider = LegacyLMArena
)

### NousResearch ###
# hermes-2
hermes_2_dpo = Model(
    name          = "hermes-2-dpo",
    base_provider = "NousResearch",
    best_provider = LegacyLMArena
)

# hermes-3
hermes_3_405b = Model(
    name          = "hermes-3-405b",
    base_provider = "NousResearch",
    best_provider = LambdaChat
)

# deephermes-3
deephermes_3_8b = Model(
    name          = "deephermes-3-8b",
    base_provider = "NousResearch",
    best_provider = Blackbox
)

### Microsoft ###
# phi-3
phi_3_small = Model(
    name          = "phi-3-small",
    base_provider = "Microsoft",
    best_provider = LegacyLMArena
)

phi_3_mini = Model(
    name          = "phi-3-mini",
    base_provider = "Microsoft",
    best_provider = LegacyLMArena
)

phi_3_medium = Model(
    name          = "phi-3-medium",
    base_provider = "Microsoft",
    best_provider = LegacyLMArena
)

# phi-3.5
phi_3_5_mini = Model(
    name          = "phi-3.5-mini",
    base_provider = "Microsoft",
    best_provider = HuggingChat
)

# phi-4
phi_4 = Model(
    name          = "phi-4",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, HuggingSpace, LegacyLMArena])
)

phi_4_multimodal = VisionModel(
    name          = "phi-4-multimodal",
    base_provider = "Microsoft",
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace])
)

phi_4_reasoning_plus = Model(
    name          = "phi-4-reasoning-plus",
    base_provider = "Microsoft",
    best_provider = DeepInfraChat
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
    best_provider = DeepInfraChat
)

### Google DeepMind ###
# gemini
gemini = Model(
    name          = 'gemini-2.0',
    base_provider = 'Google',
    best_provider = Gemini
)

# gemini-1.5
gemini_1_5_flash = Model(
    name          = 'gemini-1.5-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Free2GPT, FreeGpt, TeachAnything, Websim, LegacyLMArena, Dynaspark, GeminiPro])
)

gemini_1_5_pro = Model(
    name          = 'gemini-1.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([Free2GPT, FreeGpt, TeachAnything, Websim, LegacyLMArena, GeminiPro])
)

# gemini-2.0
gemini_2_0_pro = Model(
    name          = 'gemini-2.0-pro',
    base_provider = 'Google',
    best_provider = LegacyLMArena
)

gemini_2_0_flash = Model(
    name          = 'gemini-2.0-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, LegacyLMArena, Dynaspark, GeminiPro, Gemini])
)

gemini_2_0_flash_thinking = Model(
    name          = 'gemini-2.0-flash-thinking',
    base_provider = 'Google',
    best_provider = IterListProvider([PollinationsAI, LegacyLMArena, Gemini])
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
    best_provider = IterListProvider([PollinationsAI, LegacyLMArena, Gemini])
)

gemini_2_5_pro = Model(
    name          = 'gemini-2.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([LegacyLMArena, Gemini])
)

# gemma-2
gemma_2_2b = Model(
    name          = 'gemma-2-2b',
    base_provider = 'Google',
    best_provider = LegacyLMArena
)

gemma_2_9b = Model(
    name          = 'gemma-2-9b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

gemma_2_27b = Model(
    name          = 'gemma-2-27b',
    base_provider = 'Google',
    best_provider = LegacyLMArena
)

# gemma-3
gemma_3_1b = Model(
    name          = 'gemma-3-1b',
    base_provider = 'Google',
    best_provider = Blackbox
)

gemma_3_4b = Model(
    name          = 'gemma-3-4b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

gemma_3_12b = Model(
    name          = 'gemma-3-12b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LegacyLMArena])
)

gemma_3_27b = Model(
    name          = 'gemma-3-27b',
    base_provider = 'Google',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LegacyLMArena])
)

### Anthropic ###
# claude 3
claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([LegacyLMArena, DDG])
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = LegacyLMArena
)

claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = LegacyLMArena
)

# claude 3.5
claude_3_5_haiku = Model(
    name          = 'claude-3.5-haiku',
    base_provider = 'Anthropic',
    best_provider = LegacyLMArena
)

claude_3_5_sonnet = Model(
    name          = 'claude-3.5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

# claude 3.7
claude_3_7_sonnet = Model(
    name          = 'claude-3.7-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

claude_3_7_sonnet_thinking = Model(
    name          = 'claude-3.7-sonnet-thinking',
    base_provider = 'Anthropic',
    best_provider = LegacyLMArena
)

### Reka AI ###
reka_core = Model(
    name = 'reka-core',
    base_provider = 'Reka AI',
    best_provider = IterListProvider([LegacyLMArena, Reka])
)

reka_flash = Model(
    name = 'reka-flash',
    base_provider = 'Reka AI',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

### Blackbox AI ###
blackboxai = Model(
    name = 'blackboxai',
    base_provider = 'Blackbox AI',
    best_provider = Blackbox
)

### CohereForAI ###
command_r = Model(
    name = 'command-r',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, LegacyLMArena])
)

command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([PollinationsAI, HuggingSpace, LegacyLMArena, HuggingChat])
)

command_r7b = Model(
    name = 'command-r7b',
    base_provider = 'CohereForAI',
    best_provider = HuggingSpace
)

command_a = Model(
    name = 'command-a',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, LegacyLMArena])
)

### Qwen ###
# qwen
qwen_plus = Model(
    name = 'qwen-plus',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_max = Model(
    name = 'qwen-max',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_vl_max = Model(
    name = 'qwen-vl-max',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_14b = Model(
    name = 'qwen-14b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

# qwen-1.5
qwen_1_5_4b = Model(
    name = 'qwen-1.5-4b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_1_5_7b = Model(
    name = 'qwen-1.5-7b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([LegacyLMArena, Cloudflare])
)

qwen_1_5_14b = Model(
    name = 'qwen-1.5-14b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_1_5_32b = Model(
    name = 'qwen-1.5-32b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_1_5_72b = Model(
    name = 'qwen-1.5-72b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

qwen_1_5_110b = Model(
    name = 'qwen-1.5-110b',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

# qwen-2
qwen_2_72b = Model(
    name = 'qwen-2-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, LegacyLMArena])
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
    best_provider = Blackbox
)

qwen_2_5_72b = Model(
    name = 'qwen-2.5-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, PollinationsAI, LambdaChat, LegacyLMArena, HuggingChat])
)

qwen_2_5_1m = Model(
    name = 'qwen-2.5-1m',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_max = Model(
    name = 'qwen-2.5-max',
    base_provider = 'Qwen',
    best_provider = IterListProvider([HuggingSpace, LegacyLMArena])
)

qwen_2_5_vl_3b = Model(
    name = 'qwen-2.5-vl-3b',
    base_provider = 'Qwen',
    best_provider = Blackbox
)

qwen_2_5_vl_7b = Model(
    name = 'qwen-2.5-vl-7b',
    base_provider = 'Qwen',
    best_provider = Blackbox
)

qwen_2_5_vl_32b = Model(
    name = 'qwen-2.5-vl-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

qwen_2_5_vl_72b = Model(
    name = 'qwen-2.5-vl-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

qwen_2_5_plus = Model(
    name = 'qwen-2.5-plus',
    base_provider = 'Qwen',
    best_provider = LegacyLMArena
)

# qwen3
qwen_3_235b = Model(
    name = 'qwen-3-235b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, LegacyLMArena, HuggingSpace])
)

qwen_3_32b = Model(
    name = 'qwen-3-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace, LegacyLMArena])
)

qwen_3_30b = Model(
    name = 'qwen-3-30b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, LegacyLMArena, HuggingSpace])
)

qwen_3_14b = Model(
    name = 'qwen-3-14b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace])
)

qwen_3_4b = Model(
    name = 'qwen-3-4b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_3_1_7b = Model(
    name = 'qwen-3-1.7b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_3_0_6b = Model(
    name = 'qwen-3-0.6b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

### qwq/qvq ###
qwq_32b = Model(
    name = 'qwq-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI, LegacyLMArena, HuggingChat])
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

### Inflection ###
pi = Model(
    name = 'pi',
    base_provider = 'Inflection',
    best_provider = Pi
)

### DeepSeek ###
# deepseek
deepseek_67b = Model(
    name = 'deepseek-67b',
    base_provider = 'DeepSeek',
    best_provider = LegacyLMArena
)

# deepseek-v3
deepseek_v3 = Model(
    name = 'deepseek-v3',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, LegacyLMArena])
)

# deepseek-r1
deepseek_r1 = Model(
    name = 'deepseek-r1',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, LambdaChat, PollinationsAI, LegacyLMArena, HuggingChat, HuggingFace])
)

deepseek_r1_zero = Model(
    name = 'deepseek-r1-zero',
    base_provider = 'DeepSeek',
    best_provider = Blackbox
)

deepseek_r1_turbo = Model(
    name = 'deepseek-r1-turbo',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

deepseek_r1_distill_llama_70b = Model(
    name = 'deepseek-r1-distill-llama-70b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI])
)

deepseek_r1_distill_qwen_14b = Model(
    name = 'deepseek-r1-distill-qwen-14b',
    base_provider = 'DeepSeek',
    best_provider = Blackbox
)

deepseek_r1_distill_qwen_32b = Model(
    name = 'deepseek-r1-distill-qwen-32b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI])
)

# deepseek-v2
deepseek_v2 = Model(
    name = 'deepseek-v2',
    base_provider = 'DeepSeek',
    best_provider = LegacyLMArena
)

deepseek_coder_v2 = Model(
    name = 'deepseek-coder-v2',
    base_provider = 'DeepSeek',
    best_provider = LegacyLMArena
)

deepseek_prover_v2 = Model(
    name = 'deepseek-prover-v2',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

deepseek_prover_v2_671b = Model(
    name = 'deepseek-prover-v2-671b',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

# deepseek-v2.5
deepseek_v2_5 = Model(
    name = 'deepseek-v2.5',
    base_provider = 'DeepSeek',
    best_provider = LegacyLMArena
)

# deepseek-v3-0324
deepseek_v3_0324 = Model(
    name = 'deepseek-v3-0324',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, LegacyLMArena])
)

# janus
janus_pro_7b = VisionModel(
    name = DeepseekAI_JanusPro7b.default_model,
    base_provider = 'DeepSeek',
    best_provider = DeepseekAI_JanusPro7b
)

### x.ai ###
grok_2 = Model(
    name = 'grok-2',
    base_provider = 'x.ai',
    best_provider = IterListProvider([LegacyLMArena, Grok])
)

grok_2_mini = Model(
    name = 'grok-2-mini',
    base_provider = 'x.ai',
    best_provider = LegacyLMArena
)

grok_3 = Model(
    name = 'grok-3',
    base_provider = 'x.ai',
    best_provider = IterListProvider([LegacyLMArena, Grok])
)

grok_3_mini = Model(
    name = 'grok-3-mini',
    base_provider = 'x.ai',
    best_provider = PollinationsAI
)

grok_3_r1 = Model(
    name = 'grok-3-r1',
    base_provider = 'x.ai',
    best_provider = Grok
)

### Perplexity AI ### 
sonar = Model(
    name = 'sonar',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

sonar_pro = Model(
    name = 'sonar-pro',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

sonar_reasoning = Model(
    name = 'sonar-reasoning',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

sonar_reasoning_pro = Model(
    name = 'sonar-reasoning-pro',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

r1_1776 = Model(
    name = 'r1-1776',
    base_provider = 'Perplexity AI',
    best_provider = PerplexityLabs
)

pplx_7b_online = Model(
    name = 'pplx-7b-online',
    base_provider = 'Perplexity AI',
    best_provider = LegacyLMArena
)

pplx_70b_online = Model(
    name = 'pplx-70b-online',
    base_provider = 'Perplexity AI',
    best_provider = LegacyLMArena
)

### Nvidia ### 
nemotron_49b = Model(
    name = 'nemotron-49b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

nemotron_51b = Model(
    name = 'nemotron-51b',
    base_provider = 'Nvidia',
    best_provider = LegacyLMArena
)

nemotron_70b = Model(
    name = 'nemotron-70b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([LambdaChat, LegacyLMArena, HuggingChat, HuggingFace])
)

nemotron_253b = Model(
    name = 'nemotron-253b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([Blackbox, LegacyLMArena])
)

nemotron_4_340b = Model(
    name = 'nemotron-4-340b',
    base_provider = 'Nvidia',
    best_provider = LegacyLMArena
)

### THUDM ### 
glm_4 = Model(
    name = 'glm-4',
    base_provider = 'THUDM',
    best_provider = IterListProvider([ChatGLM, LegacyLMArena])
)

glm_4_plus = Model(
    name = 'glm-4-plus',
    base_provider = 'THUDM',
    best_provider = LegacyLMArena
)

### MiniMax ###
mini_max = Model(
    name = "minimax",
    base_provider = "MiniMax",
    best_provider = HailuoAI
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
    best_provider = Blackbox
)

dolphin_3_0_r1_24b = Model(
    name = "dolphin-3.0-r1-24b",
    base_provider = "Cognitive Computations",
    best_provider = Blackbox
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
lfm_40b = Model(
    name = "lfm-40b",
    base_provider = "Liquid AI",
    best_provider = LambdaChat
)

### Agentica ###
deepcode_14b = Model(
    name = "deepcoder-14b",
    base_provider = "Agentica",
    best_provider = Blackbox
)

### Moonshot AI ###
kimi_vl_thinking = Model(
    name = "kimi-vl-thinking",
    base_provider = "Moonshot AI",
    best_provider = Blackbox
)

moonlight_16b = Model(
    name = "moonlight-16b",
    base_provider = "Moonshot AI",
    best_provider = Blackbox
)

### Featherless Serverless LLM ### 
qwerky_72b = Model(
    name = 'qwerky-72b',
    base_provider = 'Featherless Serverless LLM',
    best_provider = Blackbox
)

### Allen AI ### 
# tulu-2
tulu_2_70b = Model(
    name = 'tulu-2-70b',
    base_provider = 'Allen AI',
    best_provider = LegacyLMArena
)

# tulu-3
tulu_3_8b = Model(
    name = 'tulu-3-8b',
    base_provider = 'Allen AI',
    best_provider = LegacyLMArena
)

tulu_3_70b = Model(
    name = 'tulu-3-70b',
    base_provider = 'Allen AI',
    best_provider = LegacyLMArena
)

### Teknium ### 
openhermes_2_5_7b = Model(
    name = 'openhermes-2.5-7b',
    base_provider = 'Allen AI',
    best_provider = LegacyLMArena
)

### Databricks ### 
dbrx_instruct = Model(
    name = 'dbrx-instruct',
    base_provider = 'Databricks',
    best_provider = LegacyLMArena
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

sd_3_5_large = ImageModel(
    name = 'sd-3.5-large',
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
    best_provider = IterListProvider([PollinationsImage, HuggingChat, HuggingFace])
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
    Now uses automatic discovery instead of manual mapping.
    """
    convert: Dict[str, Model] = {}  # Will be populated after model discovery
    
    @classmethod
    def refresh(cls):
        """Refresh the model registry and update convert"""
        ModelRegistry.refresh()
        cls.convert = ModelRegistry.all_models()
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Model]:
        """Get model by name or alias"""
        return ModelRegistry.get(name)
    
    @classmethod
    def register_alias(cls, alias: str, model_name: str):
        """Register an alias for a model"""
        ModelRegistry._aliases[alias] = model_name

# Ensure models are discovered when module is imported
ModelRegistry._discover_models()

# Update ModelUtils.convert with discovered models
ModelUtils.convert = ModelRegistry.all_models()

# Demo models configuration
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
    sd_3_5_large.name: [sd_3_5_large, [HuggingSpace, HuggingFace]],
    flux_dev.name: [flux_dev, [PollinationsImage, HuggingFace, HuggingSpace]],
    flux_schnell.name: [flux_schnell, [PollinationsImage, HuggingFace, HuggingSpace]],
}

# Create a list of all models and their providers
def _get_working_providers(model: Model) -> List:
    """Get list of working providers for a model"""
    if model.best_provider is None:
        return []
    
    if isinstance(model.best_provider, IterListProvider):
        return [p for p in model.best_provider.providers if p.working]
    
    return [model.best_provider] if model.best_provider.working else []

# Generate __models__ using the auto-discovered models
__models__ = {
    name: (model, _get_working_providers(model))
    for name, model in ModelRegistry.all_models().items()
    if name and _get_working_providers(model)
}

# Generate _all_models list
_all_models = list(__models__.keys())

# Backward compatibility - ensure Model.__all__() returns the correct list
Model.__all__ = staticmethod(lambda: _all_models)
