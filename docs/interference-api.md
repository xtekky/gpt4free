# G4F - 干扰 API 使用指南

## 目录
   - [介绍](#介绍)
   - [运行干扰 API](#运行干扰-api)
   - [从 PyPI 包运行](#从-pypi-包运行)
   - [从仓库运行](#从仓库运行)
   - [使用干扰 API](#使用干扰-api)
   - [基本用法](#基本用法)
   - [使用 OpenAI 库](#使用-openai-库)
   - [使用 Requests 库](#使用-requests-库)
   - [选择提供商](#选择提供商)
   - [关键点](#关键点)
   - [结论](#结论)

## 介绍
G4F 干扰 API 是一个强大的工具，它允许您使用 G4F (Gpt4free) 为其他 OpenAI 集成提供服务。它充当代理，将针对 OpenAI API 的请求转换为与 G4F 提供商兼容的请求。本指南将引导您完成设置、运行和有效使用干扰 API 的过程。

## 运行干扰 API
**您可以通过两种方式运行干扰 API：** 使用 PyPI 包或从仓库运行。

### 从 PyPI 包运行
**要直接从 G4F PyPI 包运行干扰 API，请使用以下 Python 代码：**

```python
from g4f.api import run_api

run_api()
```

### 从仓库运行
**如果您更喜欢从克隆的仓库运行干扰 API，您有两种选择：**

1. **使用命令行：**
```bash
g4f api
```

2. **使用 Python：**
```bash
python -m g4f.api.run
```

**运行后，API 将可通过以下地址访问：** `http://localhost:1337/v1`

**（高级）绑定到自定义端口：**
```bash
python -m g4f.cli api --bind "0.0.0.0:2400" 
```

## 使用干扰 API

### 基本用法
**您可以使用 curl 命令与干扰 API 进行交互，进行文本和图像生成：**

**文本生成：**
```bash
curl -X POST "http://localhost:1337/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {
               "role": "user",
               "content": "Hello"
             }
           ],
           "model": "gpt-4o-mini"
         }'
```

**图像生成：**
1. **url：**
```bash
curl -X POST "http://localhost:1337/v1/images/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a white siamese cat",
           "model": "flux",
           "response_format": "url"
         }'
```

2. **b64_json**
```bash
curl -X POST "http://localhost:1337/v1/images/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a white siamese cat",
           "model": "flux",
           "response_format": "b64_json"
         }'
```

---

### 使用 OpenAI 库

**要使用 OpenAI Python 库与干扰 API 进行交互，您可以将 `base_url` 指向您的端点：**

```python
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="secret",  # 设置 API 密钥（如果您的提供商不需要，可以使用 "secret"）
    base_url="http://localhost:1337/v1"  # 指向您的本地或自定义 API 端点
)

# 创建聊天补全请求
response = client.chat.completions.create(
    model="gpt-4o-mini",  # 指定要使用的模型
    messages=[{"role": "user", "content": "Write a poem about a tree"}],  # 定义输入消息
    stream=True,  # 启用流式传输以实现实时响应
)

# 处理响应
if isinstance(response, dict):
    # 非流式响应
    print(response.choices[0].message.content)
else:
    # 流式响应
    for token in response:
        content = token.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
```

**注意：**
- OpenAI Python 库需要 `api_key`。如果您的提供商不需要 API 密钥，您可以将其设置为 `"secret"`。G4F 中的提供商将忽略此值。
- 将 `"http://localhost:1337/v1"` 替换为适当的 URL，以指向您的自定义或本地干扰 API。

---

### 使用 Requests 库

**您还可以使用 `requests` 库直接向干扰 API 发送请求：**
```python
import requests

url = "http://localhost:1337/v1/chat/completions"

body = {
    "model": "gpt-4o-mini",
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}

json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```

## 选择提供商

**提供商选择**： [如何指定提供商？](selecting_a_provider.md)

选择合适的提供商是配置 G4F 干扰 API 以满足您需求的关键步骤。请参阅上面链接的指南，了解选择和指定提供商的详细说明。

## 关键点
   - 干扰 API 将 OpenAI API 请求转换为 G4F 提供商请求。
   - 它可以从 PyPI 包或克隆的仓库运行。
   - 通过更改 `base_url`，API 支持与 OpenAI Python 库一起使用。
   - 可以使用 `requests` 等库直接向 API 端点发送请求。
   - 支持文本和图像生成。

## 结论
G4F 干扰 API 提供了一种无缝的方式，将 G4F 集成到现有的基于 OpenAI 的应用程序和工具中。通过遵循本指南，您现在应该能够有效地设置、运行和使用干扰 API。无论您是将其用于文本生成、图像创建，还是作为 OpenAI 在项目中的替代品，干扰 API 都为您的 AI 驱动应用程序提供了灵活性和强大功能。

---

[返回首页](/)
