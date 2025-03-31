# G4F 客户端 API 指南

## 目录
   - [介绍](#介绍)
   - [入门](#入门)
   - [切换到 G4F 客户端](#切换到-g4f-客户端)
   - [初始化客户端](#初始化客户端)
   - [创建聊天补全](#创建聊天补全)
   - [配置](#配置)
   - [参数解释](#参数解释)
   - [使用示例](#使用示例)
   - [文本补全](#文本补全)
   - [流式补全](#流式补全)
   - [使用视觉模型](#使用视觉模型)
   - [图像生成](#图像生成)
   - [创建图像变体](#创建图像变体)
   - [高级用法](#高级用法)
   - [对话记忆](#对话记忆)
   - [搜索工具支持](#搜索工具支持)
   - [使用带有 RetryProvider 的提供商列表](#使用带有-retryprovider-的提供商列表)
   - [使用视觉模型](#使用视觉模型)
   - [命令行聊天程序](#命令行聊天程序)

## 介绍
欢迎使用 G4F 客户端 API，这是一款尖端工具，可将高级 AI 功能无缝集成到您的 Python 应用程序中。本指南旨在帮助您从使用 OpenAI 客户端过渡到 G4F 客户端，提供增强的功能，同时保持与现有 OpenAI API 的兼容性。

---

## 入门

### 切换到 G4F 客户端
**要开始使用 G4F 客户端，只需在您的 Python 代码中更新导入语句：**

**旧导入：**
```python
from openai import OpenAI
```

**新导入：**
```python
from g4f.client import Client as OpenAI
```

G4F 客户端保留了与 OpenAI 相同的熟悉 API 接口，确保平滑的过渡过程。

---

## 初始化客户端
要使用 G4F 客户端，请创建一个新实例。**以下是展示自定义提供商的示例：**
```python
from g4f.client import Client
from g4f.Provider import BingCreateImages, OpenaiChat, Gemini

client = Client(
    provider=OpenaiChat,
    image_provider=Gemini,
    # 添加任何其他必要的参数
)
```

---

## 创建聊天补全
**以下是创建聊天补全的改进示例：**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ]
    # 添加任何其他必要的参数
)
```

**此示例：**
   - 提出特定问题 `Say this is a test`
   - 配置各种参数，如 temperature 和 max_tokens，以更好地控制输出
   - 禁用流式传输以获得完整响应

您可以根据具体需求调整这些参数。

## 配置
**您可以在客户端中为提供商设置 `api_key` 并定义所有传出请求的代理：**
```python
from g4f.client import Client

client = Client(
    api_key="your_api_key_here",
    proxies="http://user:pass@host",
    # 添加任何其他必要的参数
)
```

---

## 参数解释
**使用 G4F 创建聊天补全或执行相关任务时，您可以配置以下参数：**
- **`model`**:  
  指定用于任务的 AI 模型。示例包括 `"gpt-4o"` 表示 GPT-4 优化版或 `"gpt-4o-mini"` 表示轻量版。模型的选择决定了响应的质量和速度。请确保所选模型受提供商支持。

- **`messages`**:  
  **表示对话上下文的字典列表。每个字典包含两个键：**
      - `role`: 定义消息发送者的角色，如 `"user"`（用户输入）或 `"system"`（对 AI 的指令）。  
      - `content`: 消息的实际文本。  
  **示例：**
  ```python
  [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What day is it today?"}
  ]
  ```

- **`provider`**:
*(可选)* 指定 API 的后端提供商。示例包括 `g4f.Provider.Blackbox` 或 `g4f.Provider.OpenaiChat`。每个提供商可能支持不同的模型和功能，因此请选择符合您需求的提供商。

- **`web_search`** (可选):  
  布尔标志，指示是否启用基于互联网的搜索功能。这对于获取模型训练数据中未包含的实时或特定详细信息非常有用。

#### 提供商限制
`web_search` 参数 **仅限于特定提供商**，包括：
  - ChatGPT
  - HuggingChat
  - Blackbox
  - RubiksAI

如果您选择的提供商不支持 `web_search`，则它将无法按预期工作。  

**替代解决方案：**  
与其依赖 `web_search` 参数，您可以使用更通用的 **搜索工具支持**，它允许高度自定义的网络搜索操作。搜索工具使您能够定义查询、结果数量、字数限制和超时等参数，提供更大的搜索功能控制。

---

## 使用示例

### 文本补全
**使用 `ChatCompletions` 端点生成文本补全：** 
```python
from g4f.client import Client

client = Client()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ],
    web_search = False
)

print(response.choices[0].message.content)
```

### 流式补全
**在生成响应时逐步处理响应：**
```python
from g4f.client import Client

client = Client()

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test"
        }
    ],
    stream=True,
    web_search = False
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content or "", end="")
```

---
### 使用视觉模型
**分析图像并生成描述：**
```python
import g4f
import requests

from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro

# 使用所需提供商和 api key 初始化 GPT 客户端
client = Client(
    api_key="your_api_key_here",
    provider=GeminiPro
)

image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).raw
# 或：image = open("docs/images/cat.jpeg", "rb")

response = client.chat.completions.create(
    model=g4f.models.default,
    messages=[
        {
            "role": "user",
            "content": "What's in this image?"
        }
    ],
    image=image
    # 添加任何其他必要的参数
)

print(response.choices[0].message.content)
```

---

### 图像生成
**`response_format` 参数是可选的，可以具有以下值：**
- **如果未指定（默认）：** 图像将本地保存，并返回本地路径（例如，"/images/1733331238_cf9d6aa9-f606-4fea-ba4b-f06576cba309.jpg"）。
- **"url":** 返回生成图像的 URL。
- **"b64_json":** 以 base64 编码的 JSON 字符串返回图像。

**使用指定的提示生成图像：**
```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="flux",
    prompt="a white siamese cat",
    response_format="url"
    # 添加任何其他必要的参数
)

image_url = response.data[0].url

print(f"Generated image URL: {image_url}")
```

#### Base64 响应格式
```python
from g4f.client import Client

client = Client()

response = client.images.generate(
    model="flux",
    prompt="a white siamese cat",
    response_format="b64_json"
    # 添加任何其他必要的参数
)

base64_text = response.data[0].b64_json
print(base64_text)
```

### 创建图像变体
**创建现有图像的变体：**
```python
from g4f.client import Client
from g4f.Provider import OpenaiChat

client = Client(
    image_provider=OpenaiChat
)

response = client.images.create_variation(
    image=open("docs/images/cat.jpg", "rb"),
    model="dall-e-3",
    # 添加任何其他必要的参数
)

image_url = response.data[0].url

print(f"Generated image URL: {image_url}")
```

---

## 高级用法

### 对话记忆
为了保持连贯的对话，重要的是存储对话的上下文或历史记录。这可以通过将用户的输入和机器人的响应附加到消息列表中来实现。这使得模型在生成响应时可以参考过去的交流。

**对话历史记录由具有不同角色的消息组成：**
- `system`: 定义 AI 行为的初始指令
- `user`: 来自用户的消息
- `assistant`: 来自 AI 的响应

**以下示例演示了如何使用 G4F 实现对话记忆：**
```python
from g4f.client import Client

class Conversation:
    def __init__(self):
        self.client = Client()
        self.history = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]
    
    def add_message(self, role, content):
        self.history.append({
            "role": role,
            "content": content
        })
    
    def get_response(self, user_message):
        # 将用户消息添加到历史记录中
        self.add_message("user", user_message)
        
        # 从 AI 获取响应
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            web_search=False
        )
        
        # 将 AI 响应添加到历史记录中
        assistant_response = response.choices[0].message.content
        self.add_message("assistant", assistant_response)
        
        return assistant_response

def main():
    conversation = Conversation()
    
    print("=" * 50)
    print("G4F 聊天已启动（输入 'exit' 结束）".center(50))
    print("=" * 50)
    print("\nAI: 你好！今天我能帮你什么？")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() == 'exit':
            print("\n再见！")
            break
            
        response = conversation.get_response(user_input)
        print("\nAI:", response)

if __name__ == "__main__":
    main()
```

**主要功能：**
- 通过消息历史记录维护对话上下文
- 包含 AI 行为的系统指令
- 自动存储用户输入和 AI 响应
- 使用基于类的方法实现简单且干净

**使用示例：**
```python
conversation = Conversation()
response = conversation.get_response("Hello, how are you?")
print(response)
```

**注意：**
对话历史记录会随着每次交互而增长。对于长对话，您可能需要实现一种方法来限制历史记录的大小或清除旧消息以管理令牌使用。

---

## 搜索工具支持

**搜索工具支持** 功能允许在聊天补全期间触发网络搜索。这对于检索实时或特定数据非常有用，提供比 `web_search` 更灵活的解决方案。

**使用示例**：
```python
from g4f.client import Client

client = Client()

tool_calls = [
    {
        "function": {
            "arguments": {
                "query": "Latest advancements in AI",
                "max_results": 5,
                "max_words": 2500,
                "backend": "auto",
                "add_text": True,
                "timeout": 5
            },
            "name": "search_tool"
        },
        "type": "function"
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Tell me about recent advancements in AI."}
    ],
    tool_calls=tool_calls
)

print(response.choices[0].message.content)
```

**`search_tool` 的参数：**
- **`query`**: 搜索查询字符串。
- **`max_results`**: 要检索的搜索结果数量。
- **`max_words`**: 响应中的最大字数。
- **`backend`**: 用于搜索的后端（例如，`"api"`）。
- **`add_text`**: 是否在响应中包含文本片段。
- **`timeout`**: 搜索操作的最大时间（以秒为单位）。

**搜索工具支持的优势：**
- 适用于任何提供商，无论是否支持 `web_search`。
- 提供更多的自定义和控制搜索过程。
- 绕过提供商特定的限制。

---

### 使用带有 RetryProvider 的提供商列表
```python
from g4f.client import Client
from g4f.Provider import RetryProvider, Phind, FreeChatgpt, Liaobots
import g4f.debug

g4f.debug.logging = True
g4f.debug.version_check = False

client = Client(
    provider=RetryProvider([Phind, FreeChatgpt, Liaobots], shuffle=False)
)

response = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(response.choices[0].message.content)
```
  
## 命令行聊天程序
**以下是使用 G4F 客户端的简单命令行聊天程序示例：**
```python
import g4f
from g4f.client import Client

# 使用所需提供商初始化 GPT 客户端
client = Client()

# 初始化空的对话历史记录
messages = []

while True:
    # 获取用户输入
    user_input = input("你: ")

    # 检查用户是否想退出聊天
    if user_input.lower() == "exit":
        print("退出聊天...")
        break  # 退出循环以结束对话

    # 更新对话历史记录中的用户消息
    messages.append({"role": "user", "content": user_input})

    try:
        # 获取 GPT 的响应
        response = client.chat.completions.create(
            messages=messages,
            model=g4f.models.default,
        )

        # 提取 GPT 响应并打印
        gpt_response = response.choices[0].message.content
        print(f"Bot: {gpt_response}")

        # 更新对话历史记录中的 GPT 响应
        messages.append({"role": "assistant", "content": gpt_response})

    except Exception as e:
        print(f"发生错误：{e}")
```
 
本指南提供了 G4F 客户端 API 的全面概述，展示了其在处理各种 AI 任务（从文本生成到图像分析和创建）方面的多功能性。通过利用这些功能，您可以构建强大且响应迅速的应用程序，充分利用高级 AI 模型的能力。


---  
[返回首页](/)
