## 视觉支持在聊天补全中的应用

本文档提供了如何在聊天补全中集成视觉支持的概述，包括使用 API 和客户端的示例。

### 使用 API 的示例

要在聊天补全中使用视觉支持，请按照以下示例操作：

```python
import requests
import json
from g4f.image import to_data_uri
from g4f.requests.raise_for_status import raise_for_status

url = "http://localhost:8080/v1/chat/completions"
body = {
    "model": "",
    "provider": "Copilot",
    "messages": [
        {"role": "user", "content": "what are on this image?"}
    ],
    "images": [
        ["data:image/jpeg;base64,...", "cat.jpeg"]
    ]
}
response = requests.post(url, json=body, headers={"g4f-api-key": "secret"})
raise_for_status(response)
print(response.json())
```

在此示例中：
- `url` 是聊天补全 API 的端点。
- `body` 包含模型、提供商、消息和图像。
- `messages` 是包含角色和内容的消息对象列表。
- `images` 是 Data URI 格式的图像数据和可选文件��的列表。
- `response` 存储 API 响应。

### 使用客户端的示例

要在聊天补全中使用视觉支持，请按照以下示例操作：

```python
import g4f
import g4f.Provider

def chat_completion(prompt):
    client = g4f.Client(provider=g4f.Provider.Blackbox)
    images = [
        [open("docs/images/waterfall.jpeg", "rb"), "waterfall.jpeg"],
        [open("docs/images/cat.webp", "rb"), "cat.webp"]
    ]
    response = client.chat.completions.create([{"content": prompt, "role": "user"}], "", images=images)
    print(response.choices[0].message.content)

prompt = "what are on this images?"
chat_completion(prompt)
```

```
**Image 1**

* A waterfall with a rainbow
* Lush greenery surrounding the waterfall
* A stream flowing from the waterfall

**Image 2**

* A white cat with blue eyes
* A bird perched on a window sill
* Sunlight streaming through the window
```

在此示例中：
- `client` 初始化一个带有指定提供商的新客户端。
- `images` 是图像数据和可选文件名的列表。
- `response` 存储来自客户端的响应。
- `chat_completion` 函数打印聊天补全输出。

### 注意事项

- 可以发送多张图像。每张图像有两个数据部分：图像数据（API 使用 Data URI 格式）和可选文件名。
- 客户端支持字节、IO 对象和 PIL 图像作为输入。
- 确保使用支持视觉和多图像的提供商。
