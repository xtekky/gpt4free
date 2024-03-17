models = {
    "mistral-7b": {
        "path": "mistral-7b-openorca.gguf2.Q4_0.gguf",
        "ram": "8",
        "prompt": "<|im_start|>user\n%1<|im_end|>\n<|im_start|>assistant\n",
        "system": "<|im_start|>system\nYou are MistralOrca, a large language model trained by Alignment Lab AI. For multi-step problems, write out your reasoning for each step.\n<|im_end|>"
    },
    "mistral-7b-instruct": {
        "path": "mistral-7b-instruct-v0.1.Q4_0.gguf",
        "ram": "8",
        "prompt": "[INST] %1 [/INST]",
        "system": None
    },
    "gpt4all-falcon": {
        "path": "gpt4all-falcon-newbpe-q4_0.gguf",
        "ram": "8",
        "prompt": "### Instruction:\n%1\n### Response:\n",
        "system": None
    },
    "orca-2": {
        "path": "orca-2-13b.Q4_0.gguf",
        "ram": "16",
        "prompt": None,
        "system": None
    },
    "wizardlm-13b": {
        "path": "wizardlm-13b-v1.2.Q4_0.gguf",
        "ram": "16",
        "prompt": None,
        "system": None
    },
    "nous-hermes-llama2": {
        "path": "nous-hermes-llama2-13b.Q4_0.gguf",
        "ram": "16",
        "prompt": "### Instruction:\n%1\n### Response:\n",
        "system": None
    },
    "gpt4all-13b-snoozy": {
        "path": "gpt4all-13b-snoozy-q4_0.gguf",
        "ram": "16",
        "prompt": None,
        "system": None
    },
    "mpt-7b-chat": {
        "path": "mpt-7b-chat-newbpe-q4_0.gguf",
        "ram": "8",
        "prompt": "<|im_start|>user\n%1<|im_end|>\n<|im_start|>assistant\n",
        "system": "<|im_start|>system\n- You are a helpful assistant chatbot trained by MosaicML.\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>"
    },
    "orca-mini-3b": {
        "path": "orca-mini-3b-gguf2-q4_0.gguf",
        "ram": "4",
        "prompt": "### User:\n%1\n### Response:\n",
        "system": "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    },
    "replit-code-3b": {
        "path": "replit-code-v1_5-3b-newbpe-q4_0.gguf",
        "ram": "4",
        "prompt": "%1",
        "system": None
    },
    "starcoder": {
        "path": "starcoder-newbpe-q4_0.gguf",
        "ram": "4",
        "prompt": "%1",
        "system": None
    },
    "rift-coder-7b": {
        "path": "rift-coder-v0-7b-q4_0.gguf",
        "ram": "8",
        "prompt": "%1",
        "system": None
    },
    "all-MiniLM-L6-v2": {
        "path": "all-MiniLM-L6-v2-f16.gguf",
        "ram": "1",
        "prompt": None,
        "system": None
    },
    "mistral-7b-german": {
        "path": "em_german_mistral_v01.Q4_0.gguf",
        "ram": "8",
        "prompt": "USER: %1 ASSISTANT: ",
        "system": "Du bist ein hilfreicher Assistent. "
    }
}