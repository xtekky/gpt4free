
### G4F - Local Usage Guide
 

### Table of Contents
1. [Introduction](#introduction)
2. [Required Dependencies](#required-dependencies)
3. [Basic Usage Example](#basic-usage-example)
4. [Supported Models](#supported-models)
5. [Performance Considerations](#performance-considerations)
6. [Troubleshooting](#troubleshooting)

#### Introduction
This guide explains how to use g4f to run language models locally. G4F (GPT4Free) allows you to interact with various language models on your local machine, providing a flexible and private solution for natural language processing tasks.

## Usage
 
#### Local inference
How to use g4f to run language models locally
  
#### Required dependencies
**Make sure to install the required dependencies by running:**
```bash
pip install g4f[local]
```
or
```bash
pip install -U gpt4all
```

  

#### Basic usage example 
```python
from g4f.local import LocalClient

client   = LocalClient()
response = client.chat.completions.create(
    model    = 'orca-mini-3b',
    messages = [{"role": "user", "content": "hi"}],
    stream   = True
)

for token in response:
    print(token.choices[0].delta.content or "")
```

Upon first use, there will be a prompt asking you if you wish to download the model. If you respond with `y`, g4f will go ahead and download the model for you.

You can also manually place supported models into `./g4f/local/models/`
  

**You can get a list of the current supported models by running:**
```python
from g4f.local import LocalClient

client   = LocalClient()
client.list_models()
``` 

```json
{
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
```

#### Performance Considerations
**When running language models locally, consider the following:**
   - RAM requirements vary by model size (see the 'ram' field in the model list).
   - CPU/GPU capabilities affect inference speed.
   - Disk space is needed to store the model files.

#### Troubleshooting
**Common issues and solutions:**
   1. **Model download fails**: Check your internet connection and try again.
   2. **Out of memory error**: Choose a smaller model or increase your system's RAM.
   3. **Slow inference**: Consider using a GPU or a more powerful CPU.



[Return to Home](/)
