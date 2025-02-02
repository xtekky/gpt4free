from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    ### no auth required ###
    Blackbox,
    CablyAI,
    ChatGLM,
    ChatGptEs,
    ChatGptt,
    Cloudflare,
    Copilot,
    DarkAI,
    DDG,
    DeepInfraChat,
    HuggingSpace,
    Glider,
    GPROChat,
    ImageLabs,
    Jmuz,
    Liaobots,
    Mhystical,
    OIVSCode,
    PerplexityLabs,
    Pi,
    PollinationsAI,
    TeachAnything,
    Yqcloud,
    
    ### needs auth ###
    BingCreateImages,
    CopilotAccount,
    Gemini,
    GeminiPro,
    GigaChat,
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

class VisionModel(Model):
    pass

### Default ###
default = Model(
    name = "",
    base_provider = "",
    best_provider = IterListProvider([
        DDG,
        Blackbox,
        Copilot,
        DeepInfraChat,
        ChatGptEs,
        ChatGptt,
        PollinationsAI,
        Jmuz,
        CablyAI,
        OIVSCode,
        DarkAI,
        OpenaiChat,
        Cloudflare,
    ])
)

default_vision = Model(
    name = "",
    base_provider = "",
    best_provider = IterListProvider([
        Blackbox,
        PollinationsAI,
        HuggingSpace,
        GeminiPro,
        HuggingFaceAPI,
        CopilotAccount,
        OpenaiAccount,
        Gemini,
    ], shuffle=False)
)

###################
### Text/Vision ###
###################

### OpenAI ###
# gpt-3.5
gpt_35_turbo = Model(
    name          = 'gpt-3.5-turbo',
    base_provider = 'OpenAI',
    best_provider = DarkAI
)

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, Blackbox, Jmuz, ChatGptEs, ChatGptt, PollinationsAI, Yqcloud, Copilot, OpenaiChat, Liaobots, Mhystical])
)

# gpt-4o
gpt_4o = VisionModel(
    name          = 'gpt-4o',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([Blackbox, ChatGptt, Jmuz, ChatGptEs, PollinationsAI, DarkAI, Copilot, Liaobots, OpenaiChat])
)

gpt_4o_mini = Model(
    name          = 'gpt-4o-mini',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([DDG, ChatGptEs, ChatGptt, Jmuz, PollinationsAI, OIVSCode, Liaobots, OpenaiChat])
)

# o1
o1 = Model(
    name          = 'o1',
    base_provider = 'OpenAI',
    best_provider = OpenaiAccount
)

o1_preview = Model(
    name          = 'o1-preview',
    base_provider = 'OpenAI',
    best_provider = Liaobots
)

o1_mini = Model(
    name          = 'o1-mini',
    base_provider = 'OpenAI',
    best_provider = Liaobots
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
    best_provider = IterListProvider([Jmuz, Cloudflare])
)

llama_3_70b = Model(
    name          = "llama-3-70b",
    base_provider = "Meta Llama",
    best_provider = Jmuz
)

# llama 3.1
llama_3_1_8b = Model(
    name          = "llama-3.1-8b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, Glider, Jmuz, PollinationsAI, Cloudflare])
)

llama_3_1_70b = Model(
    name          = "llama-3.1-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([DDG, Blackbox, Glider, Jmuz, TeachAnything, DarkAI])
)

llama_3_1_405b = Model(
    name          = "llama-3.1-405b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, Jmuz])
)

# llama 3.2
llama_3_2_1b = Model(
    name          = "llama-3.2-1b",
    base_provider = "Meta Llama",
    best_provider = Cloudflare
)

llama_3_2_3b = Model(
    name          = "llama-3.2-3b",
    base_provider = "Meta Llama",
    best_provider = Glider
)

llama_3_2_11b = VisionModel(
    name          = "llama-3.2-11b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Jmuz, HuggingChat, HuggingFace])
)

llama_3_2_90b = Model(
    name          = "llama-3.2-90b",
    base_provider = "Meta Llama",
    best_provider = Jmuz
)

# llama 3.3
llama_3_3_70b = Model(
    name          = "llama-3.3-70b",
    base_provider = "Meta Llama",
    best_provider = IterListProvider([Blackbox, DeepInfraChat, PollinationsAI, Jmuz, HuggingChat, HuggingFace])
)

### Mistral ###
mixtral_7b = Model(
    name          = "mixtral-7b",
    base_provider = "Mistral",
    best_provider = Blackbox
)

mixtral_8x7b = Model(
    name          = "mixtral-8x7b",
    base_provider = "Mistral",
    best_provider = IterListProvider([DDG, Jmuz])
)

mistral_nemo = Model(
    name          = "mistral-nemo",
    base_provider = "Mistral",
    best_provider = IterListProvider([PollinationsAI, HuggingChat, HuggingFace])
)

### NousResearch ###
hermes_2_dpo = Model(
    name          = "hermes-2-dpo",
    base_provider = "NousResearch",
    best_provider = Blackbox
)


### Microsoft ###
# phi
phi_3_5_mini = Model(
    name          = "phi-3.5-mini",
    base_provider = "Microsoft",
    best_provider = HuggingChat
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
    best_provider = IterListProvider([DeepInfraChat, Jmuz])
)

### Google DeepMind ###
# gemini
gemini = Model(
    name          = 'gemini',
    base_provider = 'Google',
    best_provider = Gemini
)

# gemini-exp
gemini_exp = Model(
    name          = 'gemini-exp',
    base_provider = 'Google',
    best_provider = Jmuz
)

# gemini-1.5
gemini_1_5_flash = Model(
    name          = 'gemini-1.5-flash',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Blackbox, Jmuz, Gemini, GeminiPro, Liaobots])
)

gemini_1_5_pro = Model(
    name          = 'gemini-1.5-pro',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([Blackbox, Jmuz, GPROChat, Gemini, GeminiPro, Liaobots])
)

# gemini-2.0
gemini_2_0_flash = Model(
    name          = 'gemini-2.0-flash',
    base_provider = 'Google DeepMind',
    best_provider = IterListProvider([GeminiPro, Liaobots])
)

gemini_2_0_flash_thinking = Model(
    name          = 'gemini-2.0-flash-thinking',
    base_provider = 'Google DeepMind',
    best_provider = Liaobots
)

### Anthropic ###
# claude 3
claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([DDG, Jmuz])
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'Anthropic',
    best_provider = Liaobots
)

claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Jmuz, Liaobots])
)


# claude 3.5
claude_3_5_sonnet = Model(
    name          = 'claude-3.5-sonnet',
    base_provider = 'Anthropic',
    best_provider = IterListProvider([Blackbox, Jmuz, Liaobots])
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

### Qwen ###
qwen_1_5_7b = Model(
    name = 'qwen-1.5-7b',
    base_provider = 'Qwen',
    best_provider = Cloudflare
)
qwen_2_72b = Model(
    name = 'qwen-2-72b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)
qwen_2_vl_7b = VisionModel(
    name = "qwen-2-vl-7b",
    base_provider = 'Qwen',
    best_provider = HuggingFaceAPI
)
qwen_2_5_72b = Model(
    name = 'qwen-2.5-72b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, Jmuz])
)
qwen_2_5_coder_32b = Model(
    name = 'qwen-2.5-coder-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([DeepInfraChat, PollinationsAI, Jmuz, HuggingChat])
)
qwen_2_5_1m = Model(
    name = 'qwen-2.5-1m-demo',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
)

### qwq/qvq ###
qwq_32b = Model(
    name = 'qwq-32b',
    base_provider = 'Qwen',
    best_provider = IterListProvider([Blackbox, DeepInfraChat, Jmuz, HuggingChat])
)
qvq_72b = VisionModel(
    name = 'qvq-72b',
    base_provider = 'Qwen',
    best_provider = HuggingSpace
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
    best_provider = IterListProvider([Blackbox, Jmuz, PollinationsAI])
)

deepseek_v3 = Model(
    name = 'deepseek-v3',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, DeepInfraChat])
)

deepseek_r1 = Model(
    name = 'deepseek-r1',
    base_provider = 'DeepSeek',
    best_provider = IterListProvider([Blackbox, Glider, PollinationsAI, Jmuz, HuggingChat, HuggingFace])
)

### x.ai ###
grok_2 = Model(
    name = 'grok-2',
    base_provider = 'x.ai',
    best_provider = Liaobots
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

### Nvidia ### 
nemotron_70b = Model(
    name = 'nemotron-70b',
    base_provider = 'Nvidia',
    best_provider = IterListProvider([DeepInfraChat, HuggingChat, HuggingFace])
)

### Databricks ### 
dbrx_instruct = Model(
    name = 'dbrx-instruct',
    base_provider = 'Databricks',
    best_provider = Blackbox
)

### PollinationsAI ### 
p1 = Model(
    name = 'p1',
    base_provider = 'PollinationsAI',
    best_provider = PollinationsAI
)

### CablyAI ### 
cably_80b = Model(
    name = 'cably-80b',
    base_provider = 'CablyAI',
    best_provider = CablyAI
)

### THUDM ### 
glm_4 = Model(
    name = 'glm-4',
    base_provider = 'THUDM',
    best_provider = ChatGLM
)

### MiniMax
mini_max = Model(
    name = "MiniMax",
    base_provider = "MiniMax",
    best_provider = HailuoAI
)

### Uncensored AI ### 
evil = Model(
    name = 'evil',
    base_provider = 'Evil Mode - Experimental',
    best_provider = PollinationsAI
)


#############
### Image ###
#############

### Stability AI ###
sdxl_turbo = ImageModel(
    name = 'sdxl-turbo',
    base_provider = 'Stability AI',
    best_provider = IterListProvider([PollinationsAI, ImageLabs])
)

sd_3_5 = ImageModel(
    name = 'sd-3.5',
    base_provider = 'Stability AI',
    best_provider = HuggingSpace
)

### Black Forest Labs ###
flux = ImageModel(
    name = 'flux',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([Blackbox, PollinationsAI, HuggingSpace])
)

flux_pro = ImageModel(
    name = 'flux-pro',
    base_provider = 'Black Forest Labs',
    best_provider = PollinationsAI
)

flux_dev = ImageModel(
    name = 'flux-dev',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([HuggingSpace, HuggingChat, HuggingFace])
)

flux_schnell = ImageModel(
    name = 'flux-schnell',
    base_provider = 'Black Forest Labs',
    best_provider = IterListProvider([HuggingSpace, HuggingChat, HuggingFace])
)


### OpenAI ###
dall_e_3 = ImageModel(
    name = 'dall-e-3',
    base_provider = 'OpenAI',
    best_provider = IterListProvider([PollinationsAI, CopilotAccount, OpenaiAccount, MicrosoftDesigner, BingCreateImages])
)

### Midjourney ###
midjourney = ImageModel(
    name = 'midjourney',
    base_provider = 'Midjourney',
    best_provider = PollinationsAI
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
        gpt_35_turbo.name: gpt_35_turbo,

        # gpt-4
        gpt_4.name: gpt_4,
        
        # gpt-4o
        gpt_4o.name: gpt_4o,
        gpt_4o_mini.name: gpt_4o_mini,
        
        # o1
        o1.name: o1,
        o1_preview.name: o1_preview,
        o1_mini.name: o1_mini,

        ### Meta ###
        meta.name: meta,

        # llama-2
        llama_2_7b.name: llama_2_7b,

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
        llama_3_3_70b.name: llama_3_3_70b,
                
        ### Mistral ###
        mixtral_7b.name: mixtral_7b,
        mixtral_8x7b.name: mixtral_8x7b,
        mistral_nemo.name: mistral_nemo,

        ### NousResearch ###
        hermes_2_dpo.name: hermes_2_dpo,
                
        ### Microsoft ###
        # phi
        phi_3_5_mini.name: phi_3_5_mini,
        
        # wizardlm
        wizardlm_2_7b.name: wizardlm_2_7b,
        wizardlm_2_8x22b.name: wizardlm_2_8x22b,

        ### Google ###
        ### Gemini
        gemini.name: gemini,
        gemini_exp.name: gemini_exp,
        gemini_1_5_pro.name: gemini_1_5_pro,
        gemini_1_5_flash.name: gemini_1_5_flash,
        gemini_2_0_flash.name: gemini_2_0_flash,
        gemini_2_0_flash_thinking.name: gemini_2_0_flash_thinking,

        ### Anthropic ###
        # claude 3
        claude_3_opus.name: claude_3_opus,
        claude_3_sonnet.name: claude_3_sonnet,
        claude_3_haiku.name: claude_3_haiku,

        # claude 3.5
        claude_3_5_sonnet.name: claude_3_5_sonnet,

        ### Reka AI ###
        reka_core.name: reka_core,

        ### Blackbox AI ###
        blackboxai.name: blackboxai,
        blackboxai_pro.name: blackboxai_pro,

        ### CohereForAI ###
        command_r.name: command_r,
        command_r_plus.name: command_r_plus,
        command_r7b.name: command_r7b,

        ### GigaChat ###
        gigachat.name: gigachat,

        ### Qwen ###
        qwen_1_5_7b.name: qwen_1_5_7b,
        qwen_2_72b.name: qwen_2_72b,
        qwen_2_vl_7b.name: qwen_2_vl_7b,
        qwen_2_5_72b.name: qwen_2_5_72b,
        qwen_2_5_coder_32b.name: qwen_2_5_coder_32b,
        qwen_2_5_1m.name: qwen_2_5_1m,

        # qwq/qvq
        qwq_32b.name: qwq_32b,
        qvq_72b.name: qvq_72b,

        ### Inflection ###
        pi.name: pi,

        ### x.ai ###
        grok_2.name: grok_2,

        ### Perplexity AI ###
        sonar.name: sonar,
        sonar_pro.name: sonar_pro,
        sonar_reasoning.name: sonar_reasoning,
        
        ### DeepSeek ###
        deepseek_chat.name: deepseek_chat,
        deepseek_v3.name: deepseek_v3,
        deepseek_r1.name: deepseek_r1,

        nemotron_70b.name: nemotron_70b, ### Nvidia ###
        dbrx_instruct.name: dbrx_instruct, ### Databricks ###
        p1.name: p1, ### PollinationsAI ### 
        cably_80b.name: cably_80b, ### CablyAI ###
        glm_4.name: glm_4, ### THUDM ###
        mini_max.name: mini_max, ## MiniMax
        evil.name: evil, ### Uncensored AI ###
        
        #############
        ### Image ###
        #############

        ### Stability AI ###
        sdxl_turbo.name: sdxl_turbo,
        sd_3_5.name: sd_3_5,

        ### Flux AI ###
        flux.name: flux,
        flux_pro.name: flux_pro,
        flux_dev.name: flux_dev,
        flux_schnell.name: flux_schnell,

        ### OpenAI ###
        dall_e_3.name: dall_e_3,
        
        ### Midjourney ###
        midjourney.name: midjourney,
    }


demo_models = {
    gpt_4o.name: [gpt_4o, [PollinationsAI, Blackbox]],
    "default": [llama_3_2_11b, [HuggingFaceAPI]],
    qwen_2_vl_7b.name: [qwen_2_vl_7b, [HuggingFaceAPI]],
    qvq_72b.name: [qvq_72b, [HuggingSpace, HuggingFaceAPI]],
    deepseek_r1.name: [deepseek_r1, [HuggingFace, HuggingFaceAPI]],
    claude_3_haiku.name: [claude_3_haiku, [DDG, Jmuz]],
    command_r.name: [command_r, [HuggingSpace]],
    command_r_plus.name: [command_r_plus, [HuggingSpace]],
    command_r7b.name: [command_r7b, [HuggingSpace]],
    qwen_2_72b.name: [qwen_2_72b, [HuggingSpace]],
    qwen_2_5_coder_32b.name: [qwen_2_5_coder_32b, [HuggingFace]],
    qwq_32b.name: [qwq_32b, [HuggingFace]],
    llama_3_3_70b.name: [llama_3_3_70b, [HuggingFace]],
    sd_3_5.name: [sd_3_5, [HuggingSpace, HuggingFace]],
    flux_dev.name: [flux_dev, [HuggingSpace, HuggingFace]],
    flux_schnell.name: [flux_schnell, [HuggingFace]],
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
        if providers
    }
_all_models = list(__models__.keys())
