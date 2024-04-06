### G4F - Webview GUI

Open the GUI in a window of your OS. Runs on a local/static/ssl server and use a JavaScript API.
Supports login into the OpenAI Chat (.har files), Image Upload and streamed Text Generation.

Supports all platforms, but only Linux/Windows tested.

1. Install all python requirements with:

```bash
pip install g4f[webview]
```

2. *a)* Follow the **OS specific** steps here:
 [pywebview installation](https://pywebview.flowrl.com/guide/installation.html#dependencies)

2. *b)* **WebView2** on **Windows**: Our application requires the *WebView2 Runtime* to be installed on your system. If you do not have it installed, please download and install it from the [Microsoft Developer Website](https://developer.microsoft.com/en-us/microsoft-edge/webview2/). If you already have *WebView2 Runtime* installed but are encountering issues, navigate to *Installed Windows Apps*, select *WebView2*, and opt for the repair option.

3. Run the app with:

```python
from g4f.gui.webview import run_webview
run_webview(debug=True)
```
or execute the following command:
```bash
python -m g4f.gui.webview -debug
```

[Return to Home](/)