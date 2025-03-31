### G4F - 配置

## 目录
- [认证](#认证)
- [Cookies 配置](#cookies-配置)
- [HAR 和 Cookie 文件](#har-和-cookie-文件) 
- [调试模式](#调试模式)
- [代理配置](#代理配置)

#### 认证

请参阅 [G4F 认证设置指南](authentication.md) 以获取详细的认证设置说明。

### Cookies 配置
Cookies 对于使用 Meta AI 和 Microsoft Designer 创建图像是必需的。
此外，Google Gemini 和 WhiteRabbitNeo 提供商也需要 Cookies。
从 Bing 获取 "\_U" cookie，从 Google 获取所有以 "\_\_Secure-1PSID" 开头的 cookies。

**您可以将这些 cookies 直接传递给 create 函数，或者在运行 G4F 之前使用 `set_cookies` 方法进行设置：**
```python
from g4f.cookies import set_cookies

set_cookies(".bing.com", {
  "_U": "cookie value"
})

set_cookies(".google.com", {
  "__Secure-1PSID": "cookie value"
})
```
---
### HAR 和 Cookie 文件
**使用 .har 和 Cookie 文件**
您可以将 `.har` 和 cookie 文件 `.json` 放置在默认的 `./har_and_cookies` 目录中。要导出 cookie 文件，请使用 Chrome 网上应用店提供的 [EditThisCookie 扩展](https://chromewebstore.google.com/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg)。

**创建 .har 文件以捕获 Cookies**
要捕获 cookies，您还可以创建 `.har` 文件。有关详细信息，请参阅下一节。

### 更改 Cookies 目录并在 Python 中加载 Cookie 文件
**您可以更改 cookies 目录并在 Python 环境中加载 cookie 文件。要将 cookies 目录设置为相对于您的 Python 文件，请使用以下代码：**
```python
import os.path
from g4f.cookies import set_cookies_dir, read_cookie_files

import g4f.debug
g4f.debug.logging = True

cookies_dir = os.path.join(os.path.dirname(__file__), "har_and_cookies")
set_cookies_dir(cookies_dir)
read_cookie_files(cookies_dir)
```

### 调试模式
**如果您启用调试模式，您将看到类似以下的日志：**

```
Read .har file: ./har_and_cookies/you.com.har
Cookies added: 10 from .you.com
Read cookie file: ./har_and_cookies/google.json
Cookies added: 16 from .google.com
```

#### OpenaiChat 提供商的 .HAR 文件

##### 生成 .HAR 文件

**要使用 OpenaiChat 提供商，需要从 https://chatgpt.com/ 获取 .har 文件。请按照以下步骤创建有效的 .har 文件：**
1. 使用您喜欢的网络浏览器导航到 https://chatgpt.com/ 并使用您的凭据登录。
2. 访问浏览器中的开发者工具。通常可以通过右键单击页面并选择“检查”或按 F12 或 Ctrl+Shift+I（在 Mac 上为 Cmd+Option+I）来完成此操作。
3. 打开开发者工具后，切换到“网络”选项卡。
4. 重新加载网站以在网络选项卡中捕获加载过程。
5. 在聊天中执行一个可以在 .har 文件中捕获的操作。
6. 右键单击列出的任何网络活动并选择“保存所有为 HAR 文件”以导出 .har 文件。

##### 存储 .HAR 文件

- 如果您使用 Docker，请将导出的 .har 文件放置在 `./har_and_cookies` 目录中。或者，如果您从终端使用 Python，可以将其存储在当前工作目录中的 `./har_and_cookies` 目录中。

> **注意：** 确保您的 .har 文件安全存储，因为它可能包含敏感信息。

### 代理配置
**如果您想隐藏或更改提供商的 IP 地址，可以通过环境变量全局设置代理：**

**- 在 macOS 和 Linux 上：**
```bash
export G4F_PROXY="http://host:port"
```

**- 在 Windows 上：**
```bash
set G4F_PROXY=http://host:port
```
