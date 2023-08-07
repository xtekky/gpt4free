from types import ModuleType
from . import Provider
from dataclasses import dataclass


@dataclass
class Model:
    name: str
    base_provider: str
    best_provider: ModuleType | None


gpt_35_turbo = Model(
    name="gpt-3.5-turbo",
    base_provider="openai",
    best_provider=Provider.Forefront,
)

gpt_4 = Model(
    name="gpt-4",
    base_provider="openai",
    best_provider=Provider.Bing,
)

claude_instant_v1_100k = Model(
    name="claude-instant-v1-100k",
    base_provider="anthropic",
    best_provider=Provider.Vercel,
)

claude_instant_v1 = Model(
    name="claude-instant-v1",
    base_provider="anthropic",
    best_provider=Provider.Vercel,
)

claude_v1_100k = Model(
    name="claude-v1-100k",
    base_provider="anthropic",
    best_provider=Provider.Vercel,
)

claude_v1 = Model(
    name="claude-v1",
    base_provider="anthropic",
    best_provider=Provider.Vercel,
)

alpaca_7b = Model(
    name="alpaca-7b",
    base_provider="replicate",
    best_provider=Provider.Vercel,
)

stablelm_tuned_alpha_7b = Model(
    name="stablelm-tuned-alpha-7b",
    base_provider="replicate",
    best_provider=Provider.Vercel,
)

bloom = Model(
    name="bloom",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

bloomz = Model(
    name="bloomz",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

flan_t5_xxl = Model(
    name="flan-t5-xxl",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

flan_ul2 = Model(
    name="flan-ul2",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

gpt_neox_20b = Model(
    name="gpt-neox-20b",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

oasst_sft_4_pythia_12b_epoch_35 = Model(
    name="oasst-sft-4-pythia-12b-epoch-3.5",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

santacoder = Model(
    name="santacoder",
    base_provider="huggingface",
    best_provider=Provider.Vercel,
)

command_medium_nightly = Model(
    name="command-medium-nightly",
    base_provider="cohere",
    best_provider=Provider.Vercel,
)

command_xlarge_nightly = Model(
    name="command-xlarge-nightly",
    base_provider="cohere",
    best_provider=Provider.Vercel,
)

code_cushman_001 = Model(
    name="code-cushman-001",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

code_davinci_002 = Model(
    name="code-davinci-002",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

text_ada_001 = Model(
    name="text-ada-001",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

text_babbage_001 = Model(
    name="text-babbage-001",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

text_curie_001 = Model(
    name="text-curie-001",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

text_davinci_002 = Model(
    name="text-davinci-002",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

text_davinci_003 = Model(
    name="text-davinci-003",
    base_provider="openai",
    best_provider=Provider.Vercel,
)

palm = Model(
    name="palm",
    base_provider="google",
    best_provider=Provider.Bard,
)

falcon_40b = Model(
    name="falcon-40b",
    base_provider="huggingface",
    best_provider=Provider.H2o,
)

falcon_7b = Model(
    name="falcon-7b",
    base_provider="huggingface",
    best_provider=Provider.H2o,
)

llama_13b = Model(
    name="llama-13b",
    base_provider="huggingface",
    best_provider=Provider.H2o,
)

gpt_35_turbo_16k = Model(
    name="gpt-3.5-turbo-16k",
    base_provider="openai",
    best_provider=Provider.EasyChat,
)

gpt_35_turbo_0613 = Model(
    name="gpt-3.5-turbo-0613",
    base_provider="openai",
    best_provider=Provider.EasyChat,
)

gpt_35_turbo_16k_0613 = Model(
    name="gpt-3.5-turbo-16k-0613",
    base_provider="openai",
    best_provider=Provider.EasyChat,
)

gpt_4_32k = Model(name="gpt-4-32k", base_provider="openai", best_provider=None)

gpt_4_0613 = Model(name="gpt-4-0613", base_provider="openai", best_provider=None)


class ModelUtils:
    convert: dict[str, Model] = {
        "gpt-3.5-turbo": gpt_35_turbo,
        "gpt-3.5-turbo-16k": gpt_35_turbo_16k,
        "gpt-3.5-turbo-0613": gpt_35_turbo_0613,
        "gpt-3.5-turbo-16k-0613": gpt_35_turbo_16k_0613,
        "gpt-4": gpt_4,
        "gpt-4-32k": gpt_4_32k,
        "gpt-4-0613": gpt_4_0613,
        "claude-instant-v1-100k": claude_instant_v1_100k,
        "claude-v1-100k": claude_v1_100k,
        "claude-instant-v1": claude_instant_v1,
        "claude-v1": claude_v1,
        "alpaca-7b": alpaca_7b,
        "stablelm-tuned-alpha-7b": stablelm_tuned_alpha_7b,
        "bloom": bloom,
        "bloomz": bloomz,
        "flan-t5-xxl": flan_t5_xxl,
        "flan-ul2": flan_ul2,
        "gpt-neox-20b": gpt_neox_20b,
        "oasst-sft-4-pythia-12b-epoch-3.5": oasst_sft_4_pythia_12b_epoch_35,
        "santacoder": santacoder,
        "command-medium-nightly": command_medium_nightly,
        "command-xlarge-nightly": command_xlarge_nightly,
        "code-cushman-001": code_cushman_001,
        "code-davinci-002": code_davinci_002,
        "text-ada-001": text_ada_001,
        "text-babbage-001": text_babbage_001,
        "text-curie-001": text_curie_001,
        "text-davinci-002": text_davinci_002,
        "text-davinci-003": text_davinci_003,
        "palm2": palm,
        "palm": palm,
        "google": palm,
        "google-bard": palm,
        "google-palm": palm,
        "bard": palm,
        "falcon-40b": falcon_40b,
        "falcon-7b": falcon_7b,
        "llama-13b": llama_13b,
    }
