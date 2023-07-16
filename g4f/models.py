from g4f import Provider


class Model:
    class model:
        name: str
        base_provider: str
        best_provider: str

    class gpt_35_turbo:
        name: str = 'gpt-3.5-turbo'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Forefront

    class gpt_4:
        name: str = 'gpt-4'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Bing
        best_providers: list = [Provider.Bing, Provider.Lockchat]

    class claude_instant_v1_100k:
        name: str = 'claude-instant-v1-100k'
        base_provider: str = 'anthropic'
        best_provider: Provider.Provider = Provider.Vercel

    class claude_instant_v1:
        name: str = 'claude-instant-v1'
        base_provider: str = 'anthropic'
        best_provider: Provider.Provider = Provider.Vercel

    class claude_v1_100k:
        name: str = 'claude-v1-100k'
        base_provider: str = 'anthropic'
        best_provider: Provider.Provider = Provider.Vercel

    class claude_v1:
        name: str = 'claude-v1'
        base_provider: str = 'anthropic'
        best_provider: Provider.Provider = Provider.Vercel

    class alpaca_7b:
        name: str = 'alpaca-7b'
        base_provider: str = 'replicate'
        best_provider: Provider.Provider = Provider.Vercel

    class stablelm_tuned_alpha_7b:
        name: str = 'stablelm-tuned-alpha-7b'
        base_provider: str = 'replicate'
        best_provider: Provider.Provider = Provider.Vercel

    class bloom:
        name: str = 'bloom'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class bloomz:
        name: str = 'bloomz'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class flan_t5_xxl:
        name: str = 'flan-t5-xxl'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class flan_ul2:
        name: str = 'flan-ul2'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class gpt_neox_20b:
        name: str = 'gpt-neox-20b'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class oasst_sft_4_pythia_12b_epoch_35:
        name: str = 'oasst-sft-4-pythia-12b-epoch-3.5'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class santacoder:
        name: str = 'santacoder'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.Vercel

    class command_medium_nightly:
        name: str = 'command-medium-nightly'
        base_provider: str = 'cohere'
        best_provider: Provider.Provider = Provider.Vercel

    class command_xlarge_nightly:
        name: str = 'command-xlarge-nightly'
        base_provider: str = 'cohere'
        best_provider: Provider.Provider = Provider.Vercel

    class code_cushman_001:
        name: str = 'code-cushman-001'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class code_davinci_002:
        name: str = 'code-davinci-002'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class text_ada_001:
        name: str = 'text-ada-001'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class text_babbage_001:
        name: str = 'text-babbage-001'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class text_curie_001:
        name: str = 'text-curie-001'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class text_davinci_002:
        name: str = 'text-davinci-002'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel

    class text_davinci_003:
        name: str = 'text-davinci-003'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.Vercel
        
    class palm:
        name: str = 'palm'
        base_provider: str = 'google'
        best_provider: Provider.Provider = Provider.Bard
        
            
    """    'falcon-40b': Model.falcon_40b,
    'falcon-7b': Model.falcon_7b,
    'llama-13b': Model.llama_13b,"""
    
    class falcon_40b:
        name: str = 'falcon-40b'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.H2o
    
    class falcon_7b:
        name: str = 'falcon-7b'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.H2o
        
    class llama_13b:
        name: str = 'llama-13b'
        base_provider: str = 'huggingface'
        best_provider: Provider.Provider = Provider.H2o
        
    class gpt_35_turbo_16k:
        name: str = 'gpt-3.5-turbo-16k'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.EasyChat
        
    class gpt_35_turbo_0613:
        name: str = 'gpt-3.5-turbo-0613'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.EasyChat
        
    class gpt_35_turbo_16k_0613:
        name: str = 'gpt-3.5-turbo-16k-0613'
        base_provider: str = 'openai'
        best_provider: Provider.Provider = Provider.EasyChat
        
    class gpt_4_32k:
        name: str = 'gpt-4-32k'
        base_provider: str = 'openai'
        best_provider =  None
        
    class gpt_4_0613:
        name: str = 'gpt-4-0613'
        base_provider: str = 'openai'
        best_provider = None
    
class ModelUtils:
    convert: dict = {
        'gpt-3.5-turbo': Model.gpt_35_turbo,
        'gpt-3.6-turbo-16k': Model.gpt_35_turbo_16k,
        'gpt-3.5-turbo-0613': Model.gpt_35_turbo_0613,
        'gpt-3.5-turbo-16k-0613': Model.gpt_35_turbo_16k_0613,
        
        'gpt-4': Model.gpt_4,
        'gpt-4-32k': Model.gpt_4_32k,
        'gpt-4-0613': Model.gpt_4_0613,
        
        'claude-instant-v1-100k': Model.claude_instant_v1_100k,
        'claude-v1-100k': Model.claude_v1_100k,
        'claude-instant-v1': Model.claude_instant_v1,
        'claude-v1': Model.claude_v1,
        
        'alpaca-7b': Model.alpaca_7b,
        'stablelm-tuned-alpha-7b': Model.stablelm_tuned_alpha_7b,
        
        'bloom': Model.bloom,
        'bloomz': Model.bloomz,
        
        'flan-t5-xxl': Model.flan_t5_xxl,
        'flan-ul2': Model.flan_ul2,
        
        'gpt-neox-20b': Model.gpt_neox_20b,
        'oasst-sft-4-pythia-12b-epoch-3.5': Model.oasst_sft_4_pythia_12b_epoch_35,
        'santacoder': Model.santacoder,
        
        'command-medium-nightly': Model.command_medium_nightly,
        'command-xlarge-nightly': Model.command_xlarge_nightly,
        
        'code-cushman-001': Model.code_cushman_001,
        'code-davinci-002': Model.code_davinci_002,
        
        'text-ada-001': Model.text_ada_001,
        'text-babbage-001': Model.text_babbage_001,
        'text-curie-001': Model.text_curie_001,
        'text-davinci-002': Model.text_davinci_002,
        'text-davinci-003': Model.text_davinci_003,
        
        'palm2': Model.palm,
        'palm': Model.palm,
        'google': Model.palm,
        'google-bard': Model.palm,
        'google-palm': Model.palm,
        'bard': Model.palm,
        
        'falcon-40b': Model.falcon_40b,
        'falcon-7b': Model.falcon_7b,
        'llama-13b': Model.llama_13b,
    }