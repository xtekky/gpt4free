from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .Provider import IterListProvider, ProviderType
from .Provider import (
    ### No Auth Required ###
    Blackbox,
    Chatai,
    Cloudflare,
    Copilot,
    DeepInfraChat,
    Free2GPT,
    GptOss,
    HuggingSpace,
    Grok,
    DeepseekAI_JanusPro7b,
    ImageLabs,
    Kimi,
    LambdaChat,
    OIVSCodeSer2,
    OIVSCodeSer0501,
    OperaAria,
    Startnest,
    OpenAIFM,
    PerplexityLabs,
    PollinationsAI,
    PollinationsImage,
    TeachAnything,
    Together,
    WeWordle,
    Yqcloud,
    
    ### Needs Auth ###
    Azure,
    BingCreateImages,
    CopilotAccount,
    Gemini,
    GeminiCLI,
    GeminiPro,
    HuggingChat,
    HuggingFace,
    HuggingFaceMedia,
    HuggingFaceAPI,
    LMArenaBeta,
    Groq,
    MetaAI,
    MicrosoftDesigner,
    OpenaiAccount,
    OpenaiChat,
    OpenRouter,
)

class ModelRegistry:
    """Simplified registry for automatic model discovery"""
    _models: Dict[str, 'Model'] = {}
    _aliases: Dict[str, str] = {}
    
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
        if name in cls._models:
            return cls._models[name]
        if name in cls._aliases:
            return cls._models[cls._aliases[name]]
        return None
    
    @classmethod
    def all_models(cls) -> Dict[str, 'Model']:
        """Get all registered models"""
        return cls._models.copy()
    
    @classmethod
    def clear(cls):
        """Clear registry (for testing)"""
        cls._models.clear()
        cls._aliases.clear()
    
    @classmethod
    def list_models_by_provider(cls, provider_name: str) -> List[str]:
        """List all models that use specific provider"""
        return [name for name, model in cls._models.items() 
                if provider_name in str(model.best_provider)]
    
    @classmethod
    def validate_all_models(cls) -> Dict[str, List[str]]:
        """Validate all models and return issues"""
        issues = {}
        for name, model in cls._models.items():
            model_issues = []
            if not model.name:
                model_issues.append("Empty name")
            if not model.base_provider:
                model_issues.append("Empty base_provider")
            if model.best_provider is None:
                model_issues.append("No best_provider")
            if model_issues:
                issues[name] = model_issues
        return issues

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
    long_name: Optional[str] = None

    def get_long_name(self) -> str:
        """Get the long name of the model, if available."""
        return self.long_name if self.long_name else self.name

    def __post_init__(self):
        """Auto-register model after initialization"""
        if self.name:
            ModelRegistry.register(self)

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
        OIVSCodeSer0501,
        OIVSCodeSer2,
        Blackbox,
        Copilot,
        DeepInfraChat,
        OperaAria,
        Startnest,
        LambdaChat,
        PollinationsAI,
        Together,
        Free2GPT,
        Chatai,
        WeWordle,
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
        OIVSCodeSer0501,
        OIVSCodeSer2,
        PollinationsAI,
        OperaAria,
        Startnest,
        Together,
        HuggingSpace,
        GeminiPro,
        HuggingFaceAPI,
        CopilotAccount,
        OpenaiAccount,
        Gemini,
    ], shuffle=False)
)

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, Copilot, Yqcloud, WeWordle, OpenaiChat])
)

# gpt-4o
gpt_4o = VisionModel(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, OpenaiChat])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, Chatai, OIVSCodeSer2, Startnest, OpenaiChat])
)

gpt_4o_mini_audio = AudioModel(
    name          = 'gpt-4o-mini-audio-preview',
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
    best_provider = IterListProvider([Copilot, OpenaiAccount])
)

o1_mini = Model(
    name          = 'o1-mini',
    base_provider = 'OpenAI',
    best_provider = OpenaiAccount
)

# o3
o3_mini = Model(
    name          = 'o3-mini',
    base_provider = 'OpenAI',
    best_provider = OpenaiChat
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
    best_provider = OpenaiChat
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
    best_provider = IterListProvider([PollinationsAI, OpenaiChat])
)

gpt_4_1_mini = Model(
    name          = 'gpt-4.1-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, OIVSCodeSer0501])
)

gpt_4_1_nano = Model(
    name          = 'gpt-4.1-nano',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, PollinationsAI])
)

gpt_4_5 = Model(
    name          = 'gpt-4.5',
    base_provider = 'OpenAI',
    best_provider = OpenaiChat
)

gpt_oss_120b = Model(
    name          = 'gpt-oss-120b',
    long_name     = 'openai/gpt-oss-120b',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([GptOss, Together, DeepInfraChat, HuggingFace, OpenRouter, Groq])
)

# dall-e
dall_e_3 = ImageModel(
    name = 'dall-e-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([CopilotAccount, OpenaiAccount, MicrosoftDesigner, BingCreateImages])
)

gpt_image = ImageModel(
    name = 'gpt-image',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsImage])
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
    best_provider = IterListProvider([Cloudflare])
)

llama_2_70b = Model(
    name          = "llama-2-70b",
    base_provider = "Meta Llama",
    best_provider = Together
)

# llama-3
llama_3_8b = Model(
    name          = "llama-3-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Together, Cloudflare])
)

llama_3_70b = Model(
    name          = "llama-3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Together])
)

# llama-3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, Together, Cloudflare])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = Together
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = Together
)

# llama-3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = Cloudflare
)

llama_3_2_3b = Model(
    name          = "llama-3.2-3b",
    base_provider = "Meta Llama",
    best_provider = Together
)

llama_3_2_11b = VisionModel(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Together, HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, Together])
)

# llama-3.3
llama_3_3_70b = Model(
    name          = "llama-3.3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, Together, HuggingChat, HuggingFace])
)

# llama-4
llama_4_scout = Model(
    name          = "llama-4-scout",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, PollinationsAI, Together, Cloudflare])
)

llama_4_maverick = Model(
    name          = "llama-4-maverick",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, Together])
)

### MistralAI ###
mistral_7b = Model(
    name          = "mistral-7b",
    base_provider = "Mistral AI",
    best_provider = Together
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral AI",
    best_provider = Together
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([HuggingChat, HuggingFace])
)

mistral_small_24b = Model(
    name          = "mistral-small-24b",
    base_provider = "Mistral AI",
    best_provider = Together
)

mistral_small_3_1_24b = Model(
    name          = "mistral-small-3.1-24b",
    base_provider = "Mistral AI",
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI])
)

### NousResearch ###
# hermes-2
hermes_2_dpo = Model(
    name          = "hermes-2-dpo",
    base_provider = "NousResearch",
    best_provider = Together
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
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace])
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
    best_provider = IterListProvider([Free2GPT, TeachAnything])
)

gemini_1_5_pro = Model(
    name          = 'gemini-1.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([Free2GPT, TeachAnything])
)

# gemini-2.0
gemini_2_0_flash = Model(
    name          = 'gemini-2.0-flash',
    base_provider = 'Google',
    best_provider = IterListProvider([Gemini, GeminiPro])
)

gemini_2_0_flash_thinking = Model(
    name          = 'gemini-2.0-flash-thinking',
    base_provider = 'Google',
    best_provider = IterListProvider([Gemini, GeminiPro])
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
    best_provider = IterListProvider([Gemini, GeminiPro, GeminiCLI])
)

gemini_2_5_pro = Model(
    name          = 'gemini-2.5-pro',
    base_provider = 'Google',
    best_provider = IterListProvider([Gemini, GeminiPro, GeminiCLI])
)

# codegemma
codegemma_7b = Model(
    name          = 'codegemma-7b',
    base_provider = 'Google',
    best_provider = DeepInfraChat
)

# gemma
gemma_2b = Model(
    name          = 'gemma-2b',
    base_provider = 'Google',
    best_provider = Together
)

# gemma-1
gemma_1_1_7b = Model(
    name          = 'gemma-1.1-7b',
    base_provider = 'Google',
    best_provider = DeepInfraChat
)

# gemma-2
gemma_2_9b = Model(
    name          = 'gemma-2-9b',
    base_provider = 'Google',
    best_provider = DeepInfraChat
)

gemma_2_27b = Model(
    name          = 'gemma-2-27b',
    base_provider = 'Google',
    best_provider = Together
)

# gemma-3
gemma_3_4b = Model(
    name          = 'gemma-3-4b',
    base_provider = 'Google',
    best_provider = DeepInfraChat
)

gemma_3_12b = Model(
    name          = 'gemma-3-12b',
    base_provider = 'Google',
    best_provider = DeepInfraChat
)

gemma_3_27b = Model(
    name          = 'gemma-3-27b',
    base_provider = 'Google',
    best_provider = IterListProvider([DeepInfraChat, Together])
)

gemma_3n_e4b = Model(
    name          = 'gemma-3n-e4b',
    base_provider = 'Google',
    best_provider = Together
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
    best_provider = HuggingSpace
)

command_r_plus = Model(
    name = 'command-r-plus',
    base_provider = 'CohereForAI',
    best_provider = IterListProvider([HuggingSpace, HuggingChat])
)

command_r7b = Model(
    name = 'command-r7b',
    base_provider = 'CohereForAI',
    best_provider = HuggingSpace
)

command_a = Model(
    name = 'command-a',
    base_provider = 'CohereForAI',
    best_provider = HuggingSpace
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
    best_provider = IterListProvider([HuggingSpace, Together])
)

qwen_2_vl_7b = VisionModel(
    name = "qwen-2-vl-7b",
    base_provider = 'Qwen',
    best_provider = HuggingFaceAPI
)

qwen_2_vl_72b = VisionModel(
    name = "qwen-2-vl-72b",
    base_provider = 'Qwen',
    best_provider = Together
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
    best_provider = Together
)

qwen_2_5_72b = Model(
    name = 'qwen-2.5-72b',
    base_provider = 'Qwen',
    best_provider = Together
)

qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([PollinationsAI, LambdaChat, Together, HuggingChat])
)

qwen_2_5_1m = Model(
    name = 'qwen-2.5-1m',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_max = Model(
    name = 'qwen-2.5-max',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

qwen_2_5_vl_72b = Model(
    name = 'qwen-2.5-vl-72b',
    base_provider = 'Qwen',
    best_provider = Together
)

# qwen3
qwen_3_235b = Model(
    name = 'qwen-3-235b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, Together, HuggingSpace])
)

qwen_3_32b = Model(
    name = 'qwen-3-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, Together, HuggingSpace])
)

qwen_3_30b = Model(
    name = 'qwen-3-30b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, HuggingSpace])
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
    best_provider = IterListProvider([DeepInfraChat, Together, HuggingChat])
)

### DeepSeek ###
# deepseek-v3
deepseek_v3 = Model(
    name = 'deepseek-v3',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, Together])
)

# deepseek-r1
deepseek_r1 = Model(
    name = 'deepseek-r1',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, PollinationsAI, Together, HuggingChat, HuggingFace])
)

deepseek_r1_turbo = Model(
    name = 'deepseek-r1-turbo',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

deepseek_r1_distill_llama_70b = Model(
    name = 'deepseek-r1-distill-llama-70b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, Together])
)

deepseek_r1_distill_qwen_1_5b = Model(
    name = 'deepseek-r1-distill-qwen-1.5b',
    base_provider = 'DeepSeek',
    best_provider = Together
)

deepseek_r1_distill_qwen_14b = Model(
    name = 'deepseek-r1-distill-qwen-14b',
    base_provider = 'DeepSeek',
    best_provider = Together
)

deepseek_r1_distill_qwen_32b = Model(
    name = 'deepseek-r1-distill-qwen-32b',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat])
)

# deepseek-v2
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

# deepseek-v3-0324
deepseek_v3_0324 = Model(
    name = 'deepseek-v3-0324',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, LambdaChat])
)

deepseek_v3_0324_turbo = Model(
    name = 'deepseek-v3-0324-turbo',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
)

# deepseek-r1-0528
deepseek_r1_0528 = Model(
    name = 'deepseek-r1-0528',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([DeepInfraChat, LambdaChat, PollinationsAI])
)

deepseek_r1_0528_turbo = Model(
    name = 'deepseek-r1-0528-turbo',
    base_provider = 'DeepSeek',
    best_provider = DeepInfraChat
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
    best_provider = Grok
)

grok_3 = Model(
    name = 'grok-3',
    base_provider = 'x.ai',
    best_provider = Grok
)

grok_3_r1 = Model(
    name = 'grok-3-r1',
    base_provider = 'x.ai',
    best_provider = Grok
)

kimi = Model(
    name = 'kimi-k2',
    base_provider = 'kimi.com',
    best_provider = IterListProvider([Kimi, HuggingFace, DeepInfraChat, Groq]),
    long_name = "moonshotai/Kimi-K2-Instruct"
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
    best_provider = IterListProvider([Together, PerplexityLabs])
)

### Nvidia ### 
nemotron_70b = Model(
    name = 'nemotron-70b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([LambdaChat, Together, HuggingChat, HuggingFace])
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

### Liquid AI ###
lfm_40b = Model(
    name = "lfm-40b",
    base_provider = "Liquid AI",
    best_provider = LambdaChat
)

### Opera ###
aria = Model(
    name = "aria",
    base_provider = "Opera",
    best_provider = OperaAria
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
    best_provider = IterListProvider([HuggingFaceMedia, PollinationsImage, ImageLabs])
)

sd_3_5_large = ImageModel(
    name = 'sd-3.5-large',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([HuggingFaceMedia, HuggingSpace])
)

### Black Forest Labs ###
flux = ImageModel(
    name = 'flux',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([HuggingFaceMedia, PollinationsImage, Together, HuggingSpace])
)

flux_pro = ImageModel(
    name = 'flux-pro',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, Together])
)

flux_dev = ImageModel(
    name = 'flux-dev',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, HuggingSpace, Together, HuggingChat, HuggingFace])
)

flux_schnell = ImageModel(
    name = 'flux-schnell',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsImage, Together, HuggingChat, HuggingFace])
)

flux_redux = ImageModel(
    name = 'flux-redux',
    base_provider = 'Black Forest Labs',
    best_provider = Together
)

flux_depth = ImageModel(
    name = 'flux-depth',
    base_provider = 'Black Forest Labs',
    best_provider = Together
)

flux_canny = ImageModel(
    name = 'flux-canny',
    base_provider = 'Black Forest Labs',
    best_provider = Together
)

flux_kontext_max = ImageModel(
    name = 'flux-kontext',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([PollinationsAI, Azure, LMArenaBeta, Together])
)

flux_dev_lora = ImageModel(
    name = 'flux-dev-lora',
    base_provider = 'Black Forest Labs',
    best_provider = Together
)

class ModelUtils:
    """
    Utility class for mapping string identifiers to Model instances.
    Now uses automatic discovery instead of manual mapping.
    """
    
    convert: Dict[str, Model] = {}
    
    @classmethod
    def refresh(cls):
        """Refresh the model registry and update convert"""
        cls.convert = ModelRegistry.all_models()
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Model]:
        """Get model by name or alias"""
        return ModelRegistry.get(name)
    
    @classmethod
    def register_alias(cls, alias: str, model_name: str):
        """Register an alias for a model"""
        ModelRegistry._aliases[alias] = model_name

# Register special aliases after all models are created
ModelRegistry._aliases["gemini"] = "gemini-2.0"

# Fill the convert dictionary
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
