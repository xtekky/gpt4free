**# G4F - 认证指南**  
本文档解释了如何使用 G4F 提供商进行认证并配置 GUI 安全性。它涵盖了 API 密钥管理、基于 Cookie 的认证、速率限制和 GUI 访问控制。

---

## **目录**  
1. **[提供商认证](#提供商认证)**  
   - [先决条件](#先决条件)  
   - [API 密钥设置](#api-密钥设置)  
   - [同步使用](#同步使用)  
   - [异步使用](#异步使用)  
   - [多个提供商](#多个提供商与api密钥)  
   - [基于 Cookie 的认证](#基于-cookie-的认证)  
   - [速率限制](#速率限制)  
   - [错误处理](#错误处理)  
   - [支持的提供商](#支持的提供商)  
2. **[GUI 认证](#gui-认证)**  
   - [服务器设置](#服务器设置)  
   - [浏览器访问](#浏览器访问)  
   - [编程访问](#编程访问)  
3. **[最佳实践](#最佳实践)**  
4. **[故障排除](#故障排除)**  

---

## **提供商认证**  

### **先决条件**  
- Python 3.7+  
- 安装 `g4f` 包:  
  ```bash
  pip install g4f
  ```  
- 提供商的 API 密钥或 Cookie（如果需要）。  

---

### **API 密钥设置**  
#### **步骤 1: 设置环境变量**  
**对于 Linux/macOS (终端)**:  
```bash
# Anthropic 示例
export ANTHROPIC_API_KEY="your_key_here"

# HuggingFace 示例
export HUGGINGFACE_API_KEY="another_key_here"
```

**对于 Windows (命令提示符)**:  
```cmd
:: Anthropic 示例
set ANTHROPIC_API_KEY=your_key_here

:: HuggingFace 示例
set HUGGINGFACE_API_KEY=another_key_here
```

**对于 Windows (PowerShell)**:  
```powershell
# Anthropic 示例
$env:ANTHROPIC_API_KEY = "your_key_here"

# HuggingFace 示例
$env:HUGGINGFACE_API_KEY = "another_key_here"
```

#### **步骤 2: 初始化客户端**  
```python
from g4f.client import Client

# Anthropic 示例
client = Client(
    provider="g4f.Provider.Anthropic",
    api_key="your_key_here"  # 或使用 os.getenv("ANTHROPIC_API_KEY")
)
```

---

### **同步使用**  
```python
from g4f.client import Client

# 使用 Anthropic 初始化
client = Client(provider="g4f.Provider.Anthropic", api_key="your_key_here")

# 简单请求
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

### **异步使用**  
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    # 使用 Groq 初始化
    client = AsyncClient(provider="g4f.Provider.Groq", api_key="your_key_here")
    
    response = await client.chat.completions.create(
        model="mixtral-8x7b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

### **多个提供商与 API 密钥**  
```python
import os
import g4f.Provider
from g4f.client import Client

# 使用环境变量
providers = {
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "Groq": os.getenv("GROQ_API_KEY")
}

for provider_name, api_key in providers.items():
    client = Client(provider=getattr(g4f.Provider, provider_name), api_key=api_key)
    response = client.chat.completions.create(
        model="claude-3.5-sonnet",
        messages=[{"role": "user", "content": f"Hello to {provider_name}!"}]
    )
    print(f"{provider_name}: {response.choices[0].message.content}")
```

---

### **基于 Cookie 的认证**  
**对于像 Gemini/Bing 这样的提供商**:  
1. 打开浏览器并登录到提供商的网站。  
2. 使用开发者工具 (F12) 复制 Cookie:  
   - Chrome/Edge: **应用程序** → **Cookie**  
   - Firefox: **存储** → **Cookie**  

```python
from g4f.client import Client
from g4f.Provider import Gemini

# 使用 Cookie
client = Client(
    provider=Gemini,
)
response = client.chat.completions.create(
    model="", # 默认模型
    messages="Hello Google",
    cookies={
        "__Secure-1PSID": "your_cookie_value_here",
        "__Secure-1PSIDTS": "your_cookie_value_here"
    }
)
print(f"Gemini: {response.choices[0].message.content}")
```

---

### **速率限制**  
```python
from aiolimiter import AsyncLimiter

# 限制为每秒 5 个请求
rate_limiter = AsyncLimiter(max_rate=5, time_period=1)

async def make_request():
    async with rate_limiter:
        return await client.chat.completions.create(...)
```

---

### **错误处理**  
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_request():
    try:
        return client.chat.completions.create(...)
    except Exception as e:
        print(f"Attempt failed: {str(e)}")
        raise
```

---

### **支持的提供商**  
| 提供商       | 认证类型       | 示例模型       |  
|----------------|-----------------|----------------------|  
| Anthropic      | API 密钥         | `claude-3.5-sonnet`  |  
| Gemini         | Cookie         | `gemini-1.5-pro`     |  
| Groq           | API 密钥         | `mixtral-8x7b`       |  
| HuggingFace    | API 密钥         | `llama-3.1-70b`      |  

*完整列表: [提供商和模型](providers-and-models.md)*  

---

## **GUI 认证**  

### **服务器设置**  
1. 创建密码:  
   ```bash
   # Linux/macOS
   export G4F_API_KEY="your_password_here"

   # Windows (命令提示符)
   set G4F_API_KEY=your_password_here

   # Windows (PowerShell)
   $env:G4F_API_KEY = "your_password_here"
   ```  
2. 启动服务器:  
   ```bash
   python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
   ```  

---

### **浏览器访问**  
1. 访问 `http://localhost:8080/chat/`。  
2. 使用凭据:  
   - **用户名**: 任意值 (例如 `admin`)。  
   - **密码**: 您的 `G4F_API_KEY`。  

---

### **编程访问**  
```python
import requests

response = requests.get(
    "http://localhost:8080/chat/",
    auth=("admin", "your_password_here")
)
print("Success!" if response.status_code == 200 else f"Failed: {response.status_code}")
```

---

## **最佳实践**  
1. 🔒 **永远不要硬编码密钥**  
   - 使用 `.env` 文件或像 AWS Secrets Manager 这样的秘密管理器。  
2. 🔄 **每 90 天轮换一次密钥**  
   - 对于生产环境尤为重要。  
3. 📊 **监控 API 使用情况**  
   - 使用像 Prometheus/Grafana 这样的工具进行跟踪。  
4. ♻️ **重试临时错误**  
   - 使用 `tenacity` 库进行健壮的重试逻辑。  

---

## **故障排除**  
| 问题                     | 解决方案                                  |  
|---------------------------|-------------------------------------------|  
| **"无效的 API 密钥"**     | 1. 验证密钥拼写<br>2. 在提供商仪表板中重新生成密钥 |  
| **"Cookie 过期"**      | 1. 重新登录到提供商网站<br>2. 更新 Cookie 值 |  
| **"超出速率限制"** | 1. 实施速率限制<br>2. 升级提供商计划 |  
| **"找不到提供商"**  | 1. 检查提供商名称拼写<br>2. 验证提供商兼容性 |  

---

**[⬆ 返回顶部](#目录)** | **[提供商和模型 →](providers-and-models.md)**
