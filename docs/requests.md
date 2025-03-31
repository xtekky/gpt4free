# G4F 请求 API 指南

## 目录
- [介绍](#介绍)
- [入门](#入门)
  - [安装依赖](#安装依赖)
- [进行 API 请求](#进行-api-请求)
- [文本生成](#文本生成)
  - [使用聊天补全端点](#使用聊天补全端点)
  - [流式文本生成](#流式文本生成)
- [模型检索](#模型检索)
  - [获取可用模型](#获取可用模型)
- [图像生成](#图像生成)
  - [使用 AI 创建图像](#使用-ai-创建图像)
- [高级用法](#高级用法)

## 介绍

欢迎使用 G4F 请求 API 指南，这是一个强大的工具，可以通过 HTTP 请求直接从您的 Python 应用程序中利用 AI 功能。本指南将带您完成设置请求以与 AI 模型进行交互的步骤，包括文本生成和图像创建。

## 入门

### 安装依赖

确保您的环境中已安装 `requests` 库。如果需要，您可以通过 `pip` 安装它：

```bash
pip install requests
```

本指南提供了如何使用 Python 的 `requests` 库进行 API 请求的示例，重点是文本和图像生成以及检索可用模型。

## 进行 API 请求

在深入了解具体功能之前，了解如何构建 API 请求是至关重要的。所有端点假设您的服务器在 `http://localhost` 本地运行。如果您的服务器在不同的端口上运行，请相应地调整 URL（例如，`http://localhost:8000`）。

## 文本生成

### 使用聊天补全端点

要使用聊天补全端点生成文本响应，请按照以下示例操作：

```python
import requests

# 定义负载
payload = {
    "model": "gpt-4o",
    "temperature": 0.9,
    "messages": [{"role": "system", "content": "Hello, how are you?"}]
}

# 发送 POST 请求到聊天补全端点
response = requests.post("http://localhost/v1/chat/completions", json=payload)

# 检查请求是否成功
if response.status_code == 200:
    # 打印响应文本
    print(response.text)
else:
    print(f"请求失败，状态码 {response.status_code}")
    print("响应:", response.text)
```

**解释:**
- 此请求将对话上下文发送到模型，模型生成并返回响应。
- `temperature` 参数控制输出的随机性。

### 流式文本生成

对于希望在生成时接收部分响应或流式数据的场景，您可以利用 API 的流式功能。以下是如何使用 Python 的 `requests` 库实现流式文本生成：

```python
import requests
import json

def fetch_response(url, model, messages):
    """
    发送 POST 请求到流式聊天补全端点。

    参数:
        url (str): API 端点 URL。
        model (str): 用于文本生成的模型。
        messages (list): 消息字典列表。

    返回:
        requests.Response: 流式响应对象。
    """
    payload = {"model": model, "messages": messages, "stream": True}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    if response.status_code != 200:
        raise Exception(
            f"发送消息失败: {response.status_code} {response.text}"
        )
    return response

def process_stream(response):
    """
    处理流式响应并提取消息。

    参数:
        response (requests.Response): 流式响应对象。
    """
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line == "data: [DONE]":
                print("\n\n对话完成。")
                break
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    message = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if message:
                        print(message, end="", flush=True)
                except json.JSONDecodeError as e:
                    print(f"解码 JSON 时出错: {e}")
                    continue

# 定义 API 端点
chat_url = "http://localhost:8080/v1/chat/completions"

# 定义负载
model = ""
messages = [{"role": "user", "content": "Hello, how are you?"}]

try:
    # 获取流式响应
    response = fetch_response(chat_url, model, messages)
    
    # 处理流式响应
    process_stream(response)

except Exception as e:
    print(f"发生错误: {e}")
```

**解释:**
- **`fetch_response` 函数:**
  - 发送 POST 请求到流式聊天补全端点，指定模型和消息。
  - 设置 `stream` 参数为 `true` 以启用流式传输。
  - 如果请求失败，则引发异常。

- **`process_stream` 函数:**
  - 迭代流式响应中的每一行。
  - 解码行并检查终止信号 `"data: [DONE]"`。
  - 解析以 `"data: "` 开头的行以提取消息内容。

- **主执行:**
  - 定义 API 端点、模型和消息。
  - 获取并处理流式响应。
  - 检索并打印消息。

**使用提示:**
- 确保您的本地服务器支持流式传输。
- 如果您的本地服务器在不同的端口或路径上运行，请调整 `chat_url`。
- 使用线程或异步编程处理实时应用程序中的流。

## 模型检索

### 获取可用模型

要检索可用模型列表，您可以使用以下函数：

```python
import requests

def fetch_models():
    """
    从 API 检索可用模型列表。

    返回:
        dict: 包含可用模型或错误消息的字典。
    """
    url = "http://localhost/v1/models/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 处理 HTTP 问题
        return response.json()  # 解析并返回 JSON 响应
    except Exception as e:
        return {"error": str(e)}  # 如果出现问题，返回错误消息

models = fetch_models()

print(models)
```

**解释:**
- `fetch_models` 函数向模型端点发出 GET 请求。
- 它处理 HTTP 错误并返回包含可用模型或错误消息的解析 JSON 响应。

## 图像生成

### 使用 AI 创建图像

以下函数演示了如何使用指定模型生成图像：

```python
import requests

def generate_image(prompt: str, model: str = "flux-4o"):
    """
    根据提供的文本提示生成图像。

    参数:
        prompt (str): 图像生成的文本提示。
        model (str, optional): 用于图像生成的模型。默认为 "flux-4o"。

    返回:
        tuple: 包含图像 URL、标题和完整响应的元组。
    """
    payload = {
        "model": model,
        "temperature": 0.9,
        "prompt": prompt.replace(" ", "+"),
    }

    try:
        response = requests.post("http://localhost/v1/images/generate", json=payload)
        response.raise_for_status()
        res = response.json()

        data = res.get("data")
        if not data or not isinstance(data, list):
            raise ValueError("响应中的 'data' 无效")

        image_url = data[0].get("url")
        if not image_url:
            raise ValueError("响应数据中未找到 'url'")

        timestamp = res.get("created")
        caption = f"提示: {prompt}\n创建时间: {timestamp}\n模型: {model}"
        return image_url, caption, res

    except Exception as e:
        return None, f"错误: {e}", None

prompt = "A tiger in a forest"

image_url, caption, res = generate_image(prompt)

print("API 响应:", res)
print("图像 URL:", image_url)
print("标题:", caption)
```

**解释:**
- `generate_image` 函数构建请求以根据文本提示创建图像。
- 它处理响应和可能的错误，确保成功时返回 URL 和标题。

## 高级用法

本指南演示了 G4F 请求 API 的基本用法。该 API 提供了将高级 AI 集成到您的应用程序中的强大功能。您可以扩展这些示例以适应更复杂的工作流和任务，确保您的应用程序具有最先进的 AI 功能。

### 处理并发和异步请求

对于需要高性能和非阻塞操作的应用程序，请考虑使用异步编程库，如 `aiohttp` 或 `httpx`。以下是使用 `aiohttp` 的示例：

```python
import aiohttp
import asyncio
import json
from queue import Queue

async def fetch_response_async(url, model, messages, output_queue):
    """
    异步发送 POST 请求到流式聊天补全端点并处理流。

    参数:
        url (str): API 端点 URL。
        model (str): 用于文本生成的模型。
        messages (list): 消息字典列表。
        output_queue (Queue): 用于存储提取消息的队列。
    """
    payload = {"model": model, "messages": messages, "stream": True}
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"发送消息失败: {resp.status} {text}")
            
            async for line in resp.content:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line == "data: [DONE]":
                    break
                if decoded_line.startswith("data: "):
                    try:
                        data = json.loads(decoded_line[6:])
                        message = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if message:
                            output_queue.put(message)
                    except json.JSONDecodeError:
                        continue

async def main():
    chat_url = "http://localhost/v1/chat/completions"
    model = "gpt-4o"
    messages = [{"role": "system", "content": "Hello, how are you?"}]
    output_queue = Queue()

    try:
        await fetch_response_async(chat_url, model, messages, output_queue)
        
        while not output_queue.empty():
            msg = output_queue.get()
            print(msg)

    except Exception as e:
        print(f"发生错误: {e}")

# 运行异步主函数
asyncio.run(main())
```

**解释:**
- **`aiohttp` 库:** 便于异步 HTTP 请求，允许您的应用程序高效地处理多个请求而不阻塞。
- **`fetch_response_async` 函数:**
  - 异步发送 POST 请求到流式聊天补全端点。
  - 逐行处理流式响应。
  - 提取消息并将其排入 `output_queue`。
- **`main` 函数:**
  - 定义 API 端点、模型和消息。
  - 初始化一个 `Queue` 以存储传入消息。
  - 调用异步获取函数并处理消息。

**优点:**
- **性能:** 高效处理多个请求，减少高吞吐量应用程序中的延迟。
- **可扩展性:** 随着需求的增加轻松扩展，适用于生产环境。

**注意:** 确保已安装 `aiohttp`：

```bash
pip install aiohttp
```

## 结论

通过遵循本指南，您可以有效地将 G4F 请求 API 集成到您的 Python 应用程序中，实现强大的 AI 驱动功能，如文本和图像生成、模型检索和处理流式数据。无论您是构建简单脚本还是复杂的高性能应用程序，提供的示例都为利用 AI 在项目中的全部潜力提供了坚实的基础。

请随意自定义和扩展这些示例以适应您的特定需求。如果您遇到任何问题或有进一步的问题，请不要犹豫，寻求帮助或参考其他资源。

---

# 附加说明

1. **调整基本 URL:**
   - 本指南假设您的 API 服务器可通过 `http://localhost` 访问。如果您的服务器在不同的端口（例如 `8000`）上运行，请相应地更新 URL：
     ```python
     # 端口 8000 的示例
     chat_url = "http://localhost:8000/v1/chat/completions"
     ```
   
2. **环境变量（可选）:**
   - 为了更好的灵活性和安全性，请考虑使用环境变量来存储您的基本 URL 和其他敏感信息。
     ```python
     import os

     BASE_URL = os.getenv("API_BASE_URL", "http://localhost")
     chat_url = f"{BASE_URL}/v1/chat/completions"
     ```

3. **错误处理:**
   - 始终实现健壮的错误处理，以优雅地管理意外情况，如网络故障或无效响应。

4. **安全考虑:**
   - 确保您的本地 API 服务器是安全的，特别是如果可以通过网络访问。必要时实施认证机制。

5. **测试:**
   - 在将 API 端点集成到代码中之前，使用 [Postman](https://www.postman.com/) 或 [Insomnia](https://insomnia.rest/) 等工具测试它们。

6. **日志记录:**
   - 实现日志记录以监控应用程序的行为，这对于调试和维护系统至关重要。

---

[返回首页](/)
