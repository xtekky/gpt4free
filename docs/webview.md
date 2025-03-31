### G4F - Webview GUI

在操作系统的窗口中打开 GUI。运行在本地/静态/SSL 服务器上，并使用 JavaScript API。
支持登录到 OpenAI Chat（.har 文件）、图像上传和流式文本生成。

支持所有平台，但仅在 Linux/Windows 上进行了测试。

1. 使用以下命令安装所有 Python 依赖项：

```bash
pip install g4f[webview]
```

2. *a)* 按照此处的 **操作系统特定** 步骤操作：
 [pywebview 安装](https://pywebview.flowrl.com/guide/installation.html#dependencies)

2. *b)* **Windows 上的 WebView2**：我们的应用程序需要在系统上安装 *WebView2 Runtime*。如果尚未安装，请从 [Microsoft Developer 网站](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) 下载并安装。如果已经安装了 *WebView2 Runtime* 但遇到问题，请导航到 *已安装的 Windows 应用*，选择 *WebView2*，并选择修复选项。

3. 使用以下命令运行应用程序：

```python
from g4f.gui.webview import run_webview
run_webview(debug=True)
```
或执行以下命令：
```bash
python -m g4f.gui.webview -debug
```

[返回首页](/)
