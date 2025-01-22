**# G4F - Authentication Guide**  
This documentation explains how to authenticate with G4F providers and configure GUI security. It covers API key management, cookie-based authentication, rate limiting, and GUI access controls.

---

## **Table of Contents**  
1. **[Provider Authentication](#provider-authentication)**  
   - [Prerequisites](#prerequisites)  
   - [API Key Setup](#api-key-setup)  
   - [Synchronous Usage](#synchronous-usage)  
   - [Asynchronous Usage](#asynchronous-usage)  
   - [Multiple Providers](#multiple-providers-with-api-keys)  
   - [Cookie-Based Authentication](#cookie-based-authentication)  
   - [Rate Limiting](#rate-limiting)  
   - [Error Handling](#error-handling)  
   - [Supported Providers](#supported-providers)  
2. **[GUI Authentication](#gui-authentication)**  
   - [Server Setup](#server-setup)  
   - [Browser Access](#browser-access)  
   - [Programmatic Access](#programmatic-access)  
3. **[Best Practices](#best-practices)**  
4. **[Troubleshooting](#troubleshooting)**  

---

## **Provider Authentication**  

### **Prerequisites**  
- Python 3.7+  
- Installed `g4f` package:  
  ```bash
  pip install g4f
  ```  
- API keys or cookies from providers (if required).  

---

### **API Key Setup**  
#### **Step 1: Set Environment Variables**  
**For Linux/macOS (Terminal)**:  
```bash
# Example for Anthropic
export ANTHROPIC_API_KEY="your_key_here"

# Example for HuggingFace
export HUGGINGFACE_API_KEY="another_key_here"
```

**For Windows (Command Prompt)**:  
```cmd
:: Example for Anthropic
set ANTHROPIC_API_KEY=your_key_here

:: Example for HuggingFace
set HUGGINGFACE_API_KEY=another_key_here
```

**For Windows (PowerShell)**:  
```powershell
# Example for Anthropic
$env:ANTHROPIC_API_KEY = "your_key_here"

# Example for HuggingFace
$env:HUGGINGFACE_API_KEY = "another_key_here"
```

#### **Step 2: Initialize Client**  
```python
from g4f.client import Client

# Example for Anthropic
client = Client(
    provider="g4f.Provider.Anthropic",
    api_key="your_key_here"  # Or use os.getenv("ANTHROPIC_API_KEY")
)
```

---

### **Synchronous Usage**  
```python
from g4f.client import Client

# Initialize with Anthropic
client = Client(provider="g4f.Provider.Anthropic", api_key="your_key_here")

# Simple request
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

---

### **Asynchronous Usage**  
```python
import asyncio
from g4f.client import AsyncClient

async def main():
    # Initialize with Groq
    client = AsyncClient(provider="g4f.Provider.Groq", api_key="your_key_here")
    
    response = await client.chat.completions.create(
        model="mixtral-8x7b",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

### **Multiple Providers with API Keys**  
```python
import os
from g4f.client import Client

# Using environment variables
providers = {
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "Groq": os.getenv("GROQ_API_KEY")
}

for provider_name, api_key in providers.items():
    client = Client(provider=f"g4f.Provider.{provider_name}", api_key=api_key)
    response = client.chat.completions.create(
        model="claude-3.5-sonnet",
        messages=[{"role": "user", "content": f"Hello from {provider_name}!"}]
    )
    print(f"{provider_name}: {response.choices[0].message.content}")
```

---

### **Cookie-Based Authentication**  
**For Providers Like Gemini/Bing**:  
1. Open your browser and log in to the provider's website.  
2. Use developer tools (F12) to copy cookies:  
   - Chrome/Edge: **Application** → **Cookies**  
   - Firefox: **Storage** → **Cookies**  

```python
from g4f.Provider import Gemini

# Initialize with cookies
client = Client(
    provider=Gemini,
    cookies={
        "__Secure-1PSID": "your_cookie_value_here",
        "__Secure-1PSIDTS": "timestamp_value_here"
    }
)
```

---

### **Rate Limiting**  
```python
from aiolimiter import AsyncLimiter

# Limit to 5 requests per second
rate_limiter = AsyncLimiter(max_rate=5, time_period=1)

async def make_request():
    async with rate_limiter:
        return await client.chat.completions.create(...)
```

---

### **Error Handling**  
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

### **Supported Providers**  
| Provider       | Auth Type       | Example Models       |  
|----------------|-----------------|----------------------|  
| Anthropic      | API Key         | `claude-3.5-sonnet`  |  
| Gemini         | Cookies         | `gemini-1.5-pro`     |  
| Groq           | API Key         | `mixtral-8x7b`       |  
| HuggingFace    | API Key         | `llama-3.1-70b`      |  

*Full list: [Providers and Models](providers-and-models.md)*  

---

## **GUI Authentication**  

### **Server Setup**  
1. Create a password:  
   ```bash
   # Linux/macOS
   export G4F_API_KEY="your_password_here"

   # Windows (Command Prompt)
   set G4F_API_KEY=your_password_here

   # Windows (PowerShell)
   $env:G4F_API_KEY = "your_password_here"
   ```  
2. Start the server:  
   ```bash
   python -m g4f --debug --port 8080 --g4f-api-key $G4F_API_KEY
   ```  

---

### **Browser Access**  
1. Navigate to `http://localhost:8080/chat/`.  
2. Use credentials:  
   - **Username**: Any value (e.g., `admin`).  
   - **Password**: Your `G4F_API_KEY`.  

---

### **Programmatic Access**  
```python
import requests

response = requests.get(
    "http://localhost:8080/chat/",
    auth=("admin", "your_password_here")
)
print("Success!" if response.status_code == 200 else f"Failed: {response.status_code}")
```

---

## **Best Practices**  
1. 🔒 **Never hardcode keys**  
   - Use `.env` files or secret managers like AWS Secrets Manager.  
2. 🔄 **Rotate keys every 90 days**  
   - Especially critical for production environments.  
3. 📊 **Monitor API usage**  
   - Use tools like Prometheus/Grafana for tracking.  
4. ♻️ **Retry transient errors**  
   - Use the `tenacity` library for robust retry logic.  

---

## **Troubleshooting**  
| Issue                     | Solution                                  |  
|---------------------------|-------------------------------------------|  
| **"Invalid API Key"**     | 1. Verify key spelling<br>2. Regenerate key in provider dashboard |  
| **"Cookie Expired"**      | 1. Re-login to provider website<br>2. Update cookie values |  
| **"Rate Limit Exceeded"** | 1. Implement rate limiting<br>2. Upgrade provider plan |  
| **"Provider Not Found"**  | 1. Check provider name spelling<br>2. Verify provider compatibility |  

---

**[⬆ Back to Top](#table-of-contents)** | **[Providers and Models →](providers-and-models.md)**
