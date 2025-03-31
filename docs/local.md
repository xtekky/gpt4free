### G4F - 本地使用指南

### 目录
1. [介绍](#介绍)
2. [所需依赖](#所需依赖)
3. [基本使用示例](#基本使用示例)
4. [支持的模型](#支持的模型)
5. [性能考虑](#性能考虑)
6. [故障排除](#故障排除)

#### 介绍
本指南解释了如何使用 g4f 在本地运行语言模型。G4F (GPT4Free) 允许您在本地计算机上与各种语言模型进行交互，为自然语言处理任务提供灵活和私密的解决方案。

## 使用

#### 本地推理
如何使用 g4f 在本地运行语言模型

#### 所需依赖
**请确保通过运行以下命令安装所需的依赖项：**
```bash
pip install g4f[local]
```
或
```bash
pip install -U gpt4all
```

#### 基本使用示例
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

首次使用时，会提示您是否希望下载模型。如果您回答 `y`，g4f 将继续为您下载模型。

您也可以手动将支持的模型放置到 `./g4f/local/models/` 目录中。

**您可以通过运行以下命令获取当前支持的模型列表：**
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
        "prompt": "<|im_end|>user\n%1<|im_end|>\n<|im_end|>assistant\n",
        "system": "<|im_end|>system\nYou are MistralOrca, a large language model trained by Alignment Lab AI. For multi-step problems, write out your reasoning for each step.\n<|im_end|>"
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
        "prompt": "<|im_end|>user\n%1<|im_end|>\n<|im_end|>assistant\n",
        "system": "<|im_end|>system\n- You are a helpful assistant chatbot trained by MosaicML.\n- You answer questions.\n- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>"
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

#### 性能考虑
**在本地运行语言模型时，请考虑以下因素：**
   - RAM 要求因模型大小而异（请参见模型列表中的 'ram' 字段）。
   - CPU/GPU 能力会影响推理速度。
   - 需要磁盘空间来存储模型文件。

#### 故障排除
**常见问题及解决方案：**
   1. **模型下载失败**：检查您的互联网连接并重试。
   2. **内存不足错误**：选择较小的模型或增加系统的 RAM。
   3. **推理速度慢**：考虑使用 GPU 或更强大的 CPU。

[返回首页](/)
