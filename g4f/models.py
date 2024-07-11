from __future__  import annotations

from dataclasses import dataclass

from .Provider import IterListProvider, ProviderType
from .Provider import (
    AI365VIP,
	Bing,
	Blackbox,
	Chatgpt4o,
	ChatgptFree,
	DDG,
	DeepInfra,
	DeepInfraImage,
	FreeChatgpt,
	FreeGpt,
	Gemini,
	GeminiPro,
	GeminiProChat,
	GigaChat,
	HuggingChat,
	HuggingFace,
	Koala,
	Liaobots,
	MetaAI,
	OpenaiChat,
	PerplexityLabs,
	Pi,
	Pizzagpt,
	Reka,
	Replicate,
	ReplicateHome,
	Vercel,
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
        OpenaiChat,
        Koala,
        ChatgptFree,
        FreeChatgpt,
        DDG,
        AI365VIP,
        Pizzagpt,
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
        OpenaiChat,
        ChatgptFree,
        FreeChatgpt,
        DDG,
        AI365VIP,
        Pizzagpt,
    ])
)

gpt_35_turbo_16k = Model(
    name          = 'gpt-3.5-turbo-16k',
    base_provider = 'openai',
    best_provider = gpt_35_long.best_provider
)

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

# gpt-4
gpt_4 = Model(
    name          = 'gpt-4',
    base_provider = 'openai',
    best_provider = IterListProvider([
        Bing, Liaobots, 
    ])
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

gpt_4_turbo = Model(
    name          = 'gpt-4-turbo',
    base_provider = 'openai',
    best_provider = Bing
)

gpt_4o = Model(
    name          = 'gpt-4o',
    base_provider = 'openai',
    best_provider = IterListProvider([
        You, Liaobots, Chatgpt4o, AI365VIP
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

llama_2_70b_chat = Model(
    name          = "meta/llama-2-70b-chat",
    base_provider = "meta",
    best_provider = IterListProvider([ReplicateHome])
)

llama3_8b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-8B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate])
)

llama3_70b_instruct = Model(
    name          = "meta-llama/Meta-Llama-3-70B-Instruct",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra, PerplexityLabs, Replicate, HuggingChat, DDG])
)

codellama_34b_instruct = Model(
    name          = "codellama/CodeLlama-34b-Instruct-hf",
    base_provider = "meta",
    best_provider = HuggingChat
)

codellama_70b_instruct = Model(
    name          = "codellama/CodeLlama-70b-Instruct-hf",
    base_provider = "meta",
    best_provider = IterListProvider([DeepInfra])
)


### Mistral ###
mixtral_8x7b = Model(
    name          = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, PerplexityLabs, HuggingChat, DDG])
)

mistral_7b_v02 = Model(
    name          = "mistralai/Mistral-7B-Instruct-v0.2",
    base_provider = "huggingface",
    best_provider = IterListProvider([DeepInfra, HuggingFace, HuggingChat, ReplicateHome])
)


### NousResearch ###
Nous_Hermes_2_Mixtral_8x7B_DPO = Model(
    name          = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    base_provider = "NousResearch",
    best_provider = IterListProvider([HuggingFace, HuggingChat])
)


### 01-ai ###
Yi_1_5_34B_Chat = Model(
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
    best_provider = IterListProvider([GeminiPro, You, GeminiProChat])
)

# gemma
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
claude_v2 = Model(
    name          = 'claude-v2',
    base_provider = 'anthropic',
    best_provider = IterListProvider([Vercel])
)

claude_3_opus = Model(
    name          = 'claude-3-opus',
    base_provider = 'anthropic',
    best_provider = You
)

claude_3_sonnet = Model(
    name          = 'claude-3-sonnet',
    base_provider = 'anthropic',
    best_provider = You
)

claude_3_haiku = Model(
    name          = 'claude-3-haiku',
    base_provider = 'anthropic',
    best_provider = IterListProvider([DDG, AI365VIP])
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
    best_provider = IterListProvider([ReplicateHome, DeepInfraImage])
    
)

### AI Forever ###
kandinsky_2_2 = Model(
    name = 'ai-forever/kandinsky-2.2',
    base_provider = 'AI Forever',
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
        'gpt-3.5-turbo'          : gpt_35_turbo,
        'gpt-3.5-turbo-0613'     : gpt_35_turbo_0613,
        'gpt-3.5-turbo-16k'      : gpt_35_turbo_16k,
        'gpt-3.5-turbo-16k-0613' : gpt_35_turbo_16k_0613,
        'gpt-3.5-long': gpt_35_long,

        # gpt-4
        'gpt-4o'         : gpt_4o,
        'gpt-4'          : gpt_4,
        'gpt-4-0613'     : gpt_4_0613,
        'gpt-4-32k'      : gpt_4_32k,
        'gpt-4-32k-0613' : gpt_4_32k_0613,
        'gpt-4-turbo'    : gpt_4_turbo,
		
		
		### Meta ###
        "meta-ai": meta,
        
        'llama-2-70b-chat': llama_2_70b_chat,
        'llama3-8b': llama3_8b_instruct, # alias
        'llama3-70b': llama3_70b_instruct, # alias
        'llama3-8b-instruct' : llama3_8b_instruct,
        'llama3-70b-instruct': llama3_70b_instruct,

        'codellama-34b-instruct': codellama_34b_instruct,
        'codellama-70b-instruct': codellama_70b_instruct,


        ### Mistral (Opensource) ###
        'mixtral-8x7b': mixtral_8x7b,
        'mistral-7b-v02': mistral_7b_v02,
        
        
        ### NousResearch ###
		'Nous-Hermes-2-Mixtral-8x7B-DPO': Nous_Hermes_2_Mixtral_8x7B_DPO,


		### 01-ai ###
		'Yi-1.5-34B-Chat': Yi_1_5_34B_Chat,
		
		
		### Microsoft ###
		'Phi-3-mini-4k-instruct': Phi_3_mini_4k_instruct,


        ### Google ###
        # gemini
        'gemini': gemini,
        'gemini-pro': gemini_pro,
        
        # gemma
        'gemma-2-9b-it': gemma_2_9b_it,
        'gemma-2-27b-it': gemma_2_27b_it,


        ### Anthropic ###
        'claude-v2': claude_v2,
        'claude-3-opus': claude_3_opus,
        'claude-3-sonnet': claude_3_sonnet,
        'claude-3-haiku': claude_3_haiku,


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
        
        
        # Other
        'pi': pi,     
        
        
        
        #############
		### Image ###
		#############
		
		### Stability AI ###
        'sdxl': sdxl,
        
        ### AI Forever ###
        'kandinsky-2.2': kandinsky_2_2,
    }

_all_models = list(ModelUtils.convert.keys())
