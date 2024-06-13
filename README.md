![248433934-7886223b-c1d1-4260-82aa-da5741f303bb](https://github.com/xtekky/gpt4free/assets/98614666/ea012c87-76e0-496a-8ac4-e2de090cc6c9)

<a href="https://trendshift.io/repositories/1692" target="_blank"><img src="https://trendshift.io/api/badge/repositories/1692" alt="xtekky%2Fgpt4free | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

The **ETA** till (v3 for g4f) where I, [@xtekky](https://github.com/xtekky) will pick this project back up and improve it is **`29` days** (written Tue 28 May), join [t.me/g4f_channel](https://t.me/g4f_channel) in the meanwhile to stay updated.

_____


Written by [@xtekky](https://github.com/xtekky) & maintained by [@hlohaus](https://github.com/hlohaus)


<div id="top"></div>

> By using this repository or any code related to it, you agree to the [legal notice](https://github.com/xtekky/gpt4free/blob/main/LEGAL_NOTICE.md). The author is **not responsible for the usage of this repository nor endorses it**, nor is the author responsible for any copies, forks, re-uploads made by other users, or anything else related to GPT4Free. This is the author's only account and repository. To prevent impersonation or irresponsible actions, please comply with the GNU GPL license this Repository uses.  

> [!Warning]
*"gpt4free"* serves as a **PoC** (proof of concept), demonstrating the development of an API package with multi-provider requests, with features like timeouts, load balance and flow control.

> [!Note]
<sup><strong>Lastet version:</strong></sup> [![PyPI version](https://img.shields.io/pypi/v/g4f?color=blue)](https://pypi.org/project/g4f) [![Docker version](https://img.shields.io/docker/v/hlohaus789/g4f?label=docker&color=blue)](https://hub.docker.com/r/hlohaus789/g4f)  
> <sup><strong>Stats:</strong></sup>  [![Downloads](https://static.pepy.tech/badge/g4f)](https://pepy.tech/project/g4f) [![Downloads](https://static.pepy.tech/badge/g4f/month)](https://pepy.tech/project/g4f)

```sh
pip install -U g4f
```
```sh
docker pull hlohaus789/g4f
```

## 🆕 What's New

- Added `gpt-4o`, simply use `gpt-4o` in `chat.completion.create`.
- Installation Guide for Windows (.exe): 💻 [#installation-guide-for-windows](#installation-guide-for-windows-exe)
- Join our Telegram Channel: 📨 [telegram.me/g4f_channel](https://telegram.me/g4f_channel)
- Join our Discord Group: 💬 [discord.gg/XfybzPXPH5](https://discord.gg/XfybzPXPH5)
- `g4f` now supports 100% local inference: 🧠 [local-docs](https://g4f.mintlify.app/docs/core/usage/local)

## 🔻 Site Takedown
Is your site on this repository and you want to take it down? Send an email to takedown@g4f.ai with proof it is yours and it will be removed as fast as possible. To prevent reproduction please secure your API. 😉

## 🚀  Feedback and Todo
You can always leave some feedback here: https://forms.gle/FeWV9RLEedfdkmFN6

As per the survey, here is a list of improvements to come
- [x] Update the repository to include the new openai library syntax (ex: `Openai()` class) | completed, use `g4f.client.Client`
- [ ] Golang implementation
- [ ] 🚧 Improve Documentation (in /docs & Guides, Howtos, & Do video tutorials)
- [x] Improve the provider status list & updates
- [ ] Tutorials on how to reverse sites to write your own wrapper (PoC only ofc)
- [x] Improve the Bing wrapper. (Wait and Retry or reuse conversation)
- [ ] 🚧 Write a standard provider performance test to improve the stability
- [ ] Potential support and development of local models
- [ ] 🚧 Improve compatibility and error handling

## 📚 Table of Contents

- [🆕 What's New](#-whats-new)
- [📚 Table of Contents](#-table-of-contents)
- [🛠️ Getting Started](#-getting-started)
    + [Docker Container Guide](#docker-container-guide)
    + [Installation Guide for Windows (.exe)](#installation-guide-for-windows-exe)
    + [Use python](#use-python)
      - [Prerequisites](#prerequisites)
      - [Install using PyPI package:](#install-using-pypi-package)
      - [Install from source:](#install-from-source)
      - [Install using Docker:](#install-using-docker)
- [💡 Usage](#-usage)
  * [Text Generation](#text-generation)
  * [Image Generation](#image-generation)
  * [Web UI](#web-ui)
  * [Interference API](#interference-api)
  * [Configuration](#configuration)
- [🚀 Providers and Models](#-providers-and-models)
  * [GPT-4](#gpt-4)
  * [GPT-3.5](#gpt-35)
  * [Other](#other)
  * [Models](#models)
- [🔗 Powered by gpt4free](#-powered-by-gpt4free)
- [🤝 Contribute](#-contribute)
    + [How do i create a new Provider?](#guide-how-do-i-create-a-new-provider)
    + [How can AI help me with writing code?](#guide-how-can-ai-help-me-with-writing-code)
- [🙌 Contributors](#-contributors)
- [©️ Copyright](#-copyright)
- [⭐ Star History](#-star-history)
- [📄 License](#-license)

## 🛠️ Getting Started

#### Docker Container Guide

##### Getting Started Quickly:

1. **Install Docker:** Begin by [downloading and installing Docker](https://docs.docker.com/get-docker/).

2. **Set Up the Container:**
   Use the following commands to pull the latest image and start the container:

```sh
docker pull hlohaus789/g4f
docker run \
  -p 8080:8080 -p 1337:1337 -p 7900:7900 \
  --shm-size="2g" \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_images:/app/generated_images \
  hlohaus789/g4f:latest
```

3. **Access the Client:** 
   - To use the included client, navigate to: [http://localhost:8080/chat/](http://localhost:8080/chat/)
   - Or set the API base for your client to: [http://localhost:1337/v1](http://localhost:1337/v1)

4. **(Optional) Provider Login:**
   If required, you can access the container's desktop here: http://localhost:7900/?autoconnect=1&resize=scale&password=secret for provider login purposes.

#### Installation Guide for Windows (.exe)
To ensure the seamless operation of our application, please follow the instructions below. These steps are designed to guide you through the installation process on Windows operating systems.
### Installation Steps
1. **Download the Application**: Visit our [releases page](https://github.com/xtekky/gpt4free/releases/tag/0.3.1.7) and download the most recent version of the application, named `g4f.exe.zip`.
2. **File Placement**: After downloading, locate the `.zip` file in your Downloads folder. Unpack it to a directory of your choice on your system, then execute the `g4f.exe` file to run the app.
3. **Open GUI**: The app starts a web server with the GUI. Open your favorite browser and navigate to `http://localhost:8080/chat/` to access the application interface.
4. **Firewall Configuration (Hotfix)**: Upon installation, it may be necessary to adjust your Windows Firewall settings to allow the application to operate correctly. To do this, access your Windows Firewall settings and allow the application.

By following these steps, you should be able to successfully install and run the application on your Windows system. If you encounter any issues during the installation process, please refer to our Issue Tracker or try to get contact over Discord for assistance.

Run the **Webview UI** on other Platfroms:
- [/docs/guides/webview](https://github.com/xtekky/gpt4free/blob/main/docs/webview.md)

##### Use your smartphone:

Run the Web UI on Your Smartphone:
- [/docs/guides/phone](https://github.com/xtekky/gpt4free/blob/main/docs/guides/phone.md)

#### Use python

##### Prerequisites:

1. [Download and install Python](https://www.python.org/downloads/) (Version 3.10+ is recommended).
2. [Install Google Chrome](https://www.google.com/chrome/) for providers with webdriver

##### Install using PyPI package:

```
pip install -U g4f[all]
```

How do I install only parts or do disable parts?
Use partial requirements: [/docs/requirements](https://github.com/xtekky/gpt4free/blob/main/docs/requirements.md)

##### Install from source:

How do I load the project using git and installing the project requirements?
Read this tutorial and follow it step by step: [/docs/git](https://github.com/xtekky/gpt4free/blob/main/docs/git.md)


##### Install using Docker:

How do I build and run composer image from source?
Use docker-compose: [/docs/docker](https://github.com/xtekky/gpt4free/blob/main/docs/docker.md)


## 💡 Usage

#### Text Generation

```python
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    ...
)
print(response.choices[0].message.content)
```

```
Hello! How can I assist you today?
```

#### Image Generation

```python
from g4f.client import Client

client = Client()
response = client.images.generate(
  model="gemini",
  prompt="a white siamese cat",
  ...
)
image_url = response.data[0].url
```


[![Image with cat](/docs/cat.jpeg)](https://github.com/xtekky/gpt4free/blob/main/docs/client.md)

**Full Documentation for Python API**

- New AsyncClient API from G4F: [/docs/async_client](https://github.com/xtekky/gpt4free/blob/main/docs/async_client.md)
- Client API like the OpenAI Python library: [/docs/client](https://github.com/xtekky/gpt4free/blob/main/docs/client.md)
- Legacy API with python modules: [/docs/legacy](https://github.com/xtekky/gpt4free/blob/main/docs/legacy.md)

#### Web UI

To start the web interface, type the following codes in python:

```python
from g4f.gui import run_gui
run_gui()
```
or execute the following command:
```bash
python -m g4f.cli gui -port 8080 -debug
```

#### Interference API

You can use the Interference API to serve other OpenAI integrations with G4F.

See docs: [/docs/interference](https://github.com/xtekky/gpt4free/blob/main/docs/interference.md)

Access with: http://localhost:1337/v1

### Configuration

#### Cookies

Cookies are essential for using Meta AI and Microsoft Designer to create images.
Additionally, cookies are required for the Google Gemini and WhiteRabbitNeo Provider.
From Bing, ensure you have the "_U" cookie, and from Google, all cookies starting with "__Secure-1PSID" are needed.

You can pass these cookies directly to the create function or set them using the `set_cookies` method before running G4F:

```python
from g4f.cookies import set_cookies

set_cookies(".bing.com", {
  "_U": "cookie value"
})

set_cookies(".google.com", {
  "__Secure-1PSID": "cookie value"
})
```

#### Using .har and Cookie Files

You can place `.har` and cookie files in the default `./har_and_cookies` directory. To export a cookie file, use the [EditThisCookie Extension](https://chromewebstore.google.com/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg) available on the Chrome Web Store.

#### Creating .har Files to Capture Cookies

To capture cookies, you can also create `.har` files. For more details, refer to the next section.

#### Changing the Cookies Directory and Loading Cookie Files in Python

You can change the cookies directory and load cookie files in your Python environment. To set the cookies directory relative to your Python file, use the following code:

```python
import os.path
from g4f.cookies import set_cookies_dir, read_cookie_files

import g4f.debug
g4f.debug.logging = True

cookies_dir = os.path.join(os.path.dirname(__file__), "har_and_cookies")
set_cookies_dir(cookies_dir)
read_cookie_files(cookies_dir)
```

### Debug Mode

If you enable debug mode, you will see logs similar to the following:

```
Read .har file: ./har_and_cookies/you.com.har
Cookies added: 10 from .you.com
Read cookie file: ./har_and_cookies/google.json
Cookies added: 16 from .google.com
```

#### .HAR File for OpenaiChat Provider

##### Generating a .HAR File

To utilize the OpenaiChat provider, a .har file is required from https://chatgpt.com/. Follow the steps below to create a valid .har file:

1. Navigate to https://chatgpt.com/ using your preferred web browser and log in with your credentials.
2. Access the Developer Tools in your browser. This can typically be done by right-clicking the page and selecting "Inspect," or by pressing F12 or Ctrl+Shift+I (Cmd+Option+I on a Mac).
3. With the Developer Tools open, switch to the "Network" tab.
4. Reload the website to capture the loading process within the Network tab.
5. Initiate an action in the chat which can be captured in the .har file.
6. Right-click any of the network activities listed and select "Save all as HAR with content" to export the .har file.

##### Storing the .HAR File

- Place the exported .har file in the `./har_and_cookies` directory if you are using Docker. Alternatively, you can store it in any preferred location within your current working directory.

Note: Ensure that your .har file is stored securely, as it may contain sensitive information.

#### Using Proxy

If you want to hide or change your IP address for the providers, you can set a proxy globally via an environment variable:

- On macOS and Linux:
```bash
export G4F_PROXY="http://host:port"
```

- On Windows:
```bash
set G4F_PROXY=http://host:port
```

## 🚀 Providers and Models

### GPT-4

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bing.com](https://bing.com/chat) | `g4f.Provider.Bing` | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatgpt.ai](https://chatgpt.ai) | `g4f.Provider.ChatgptAi` | ❌ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [liaobots.site](https://liaobots.site) | `g4f.Provider.Liaobots` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt.com](https://chatgpt.com) | `g4f.Provider.OpenaiChat` | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌+✔️ |
| [raycast.com](https://raycast.com) | `g4f.Provider.Raycast` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [beta.theb.ai](https://beta.theb.ai) | `g4f.Provider.Theb` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [you.com](https://you.com) | `g4f.Provider.You` | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |

## Best OpenSource Models
While we wait for gpt-5, here is a list of new models that are at least better than gpt-3.5-turbo. **Some are better than gpt-4**. Expect this list to grow.

| Website | Provider |  parameters | better than |
| ------  | -------  |  ------ |  ------ |
| [claude-3-opus](https://anthropic.com/) | `g4f.Provider.You` | ?B | gpt-4-0125-preview |
| [command-r+](https://txt.cohere.com/command-r-plus-microsoft-azure/) | `g4f.Provider.HuggingChat` | 104B | gpt-4-0314 |
| [llama-3-70b](https://meta.ai/) | `g4f.Provider.Llama` or `DeepInfra` | 70B | gpt-4-0314 |
| [claude-3-sonnet](https://anthropic.com/) | `g4f.Provider.You` | ?B | gpt-4-0314 |
| [reka-core](https://chat.reka.ai/) | `g4f.Provider.Reka` | 21B | gpt-4-vision |
| [dbrx-instruct](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | `g4f.Provider.DeepInfra` | 132B / 36B active| gpt-3.5-turbo |
| [mixtral-8x22b](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1) | `g4f.Provider.DeepInfra` | 176B / 44b active | gpt-3.5-turbo |

### GPT-3.5

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [chat3.aiyunos.top](https://chat3.aiyunos.top/) | `g4f.Provider.AItianhuSpace` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat10.aichatos.xyz](https://chat10.aichatos.xyz) | `g4f.Provider.Aichatos` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatforai.store](https://chatforai.store) | `g4f.Provider.ChatForAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt4online.org](https://chatgpt4online.org) | `g4f.Provider.Chatgpt4Online` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt-free.cc](https://www.chatgpt-free.cc) | `g4f.Provider.ChatgptNext` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgptx.de](https://chatgptx.de) | `g4f.Provider.ChatgptX` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [f1.cnote.top](https://f1.cnote.top) | `g4f.Provider.Cnote` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [duckduckgo.com](https://duckduckgo.com/duckchat) | `g4f.Provider.DuckDuckGo` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [feedough.com](https://www.feedough.com) | `g4f.Provider.Feedough` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [flowgpt.com](https://flowgpt.com/chat) | `g4f.Provider.FlowGpt` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [freegptsnav.aifree.site](https://freegptsnav.aifree.site) | `g4f.Provider.FreeGpt` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gpttalk.ru](https://gpttalk.ru) | `g4f.Provider.GptTalkRu` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [koala.sh](https://koala.sh) | `g4f.Provider.Koala` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [app.myshell.ai](https://app.myshell.ai/chat) | `g4f.Provider.MyShell` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [perplexity.ai](https://www.perplexity.ai) | `g4f.Provider.PerplexityAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [poe.com](https://poe.com) | `g4f.Provider.Poe` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [talkai.info](https://talkai.info) | `g4f.Provider.TalkAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.vercel.ai](https://chat.vercel.ai) | `g4f.Provider.Vercel` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [aitianhu.com](https://www.aitianhu.com) | `g4f.Provider.AItianhu` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgpt.bestim.org](https://chatgpt.bestim.org) | `g4f.Provider.Bestim` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatbase.co](https://www.chatbase.co) | `g4f.Provider.ChatBase` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptdemo.info](https://chatgptdemo.info/chat) | `g4f.Provider.ChatgptDemo` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.chatgptdemo.ai](https://chat.chatgptdemo.ai) | `g4f.Provider.ChatgptDemoAi` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptfree.ai](https://chatgptfree.ai) | `g4f.Provider.ChatgptFree` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptlogin.ai](https://chatgptlogin.ai) | `g4f.Provider.ChatgptLogin` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.3211000.xyz](https://chat.3211000.xyz) | `g4f.Provider.Chatxyz` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gpt6.ai](https://gpt6.ai) | `g4f.Provider.Gpt6` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gptchatly.com](https://gptchatly.com) | `g4f.Provider.GptChatly` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [ai18.gptforlove.com](https://ai18.gptforlove.com) | `g4f.Provider.GptForLove` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gptgo.ai](https://gptgo.ai) | `g4f.Provider.GptGo` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gptgod.site](https://gptgod.site) | `g4f.Provider.GptGod` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [onlinegpt.org](https://onlinegpt.org) | `g4f.Provider.OnlineGpt` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |

### Other

| Website | Provider | Stream | Status | Auth |
| ------  | -------  | ------ | ------ | ---- |
| [openchat.team](https://openchat.team) | `g4f.Provider.Aura`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [blackbox.ai](https://www.blackbox.ai) | `g4f.Provider.Blackbox`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [cohereforai-c4ai-command-r-plus.hf.space](https://cohereforai-c4ai-command-r-plus.hf.space) | `g4f.Provider.Cohere`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [deepinfra.com](https://deepinfra.com) | `g4f.Provider.DeepInfra`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [free.chatgpt.org.uk](https://free.chatgpt.org.uk) | `g4f.Provider.FreeChatgpt`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gemini.google.com](https://gemini.google.com) | `g4f.Provider.Gemini`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [ai.google.dev](https://ai.google.dev) | `g4f.Provider.GeminiPro`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [gemini-chatbot-sigma.vercel.app](https://gemini-chatbot-sigma.vercel.app) | `g4f.Provider.GeminiProChat`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [developers.sber.ru](https://developers.sber.ru/gigachat) | `g4f.Provider.GigaChat`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [console.groq.com](https://console.groq.com/playground) | `g4f.Provider.Groq`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [huggingface.co](https://huggingface.co/chat) | `g4f.Provider.HuggingChat`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [huggingface.co](https://huggingface.co/chat) | `g4f.Provider.HuggingFace`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [llama2.ai](https://www.llama2.ai) | `g4f.Provider.Llama`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [meta.ai](https://www.meta.ai) | `g4f.Provider.MetaAI`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [openrouter.ai](https://openrouter.ai) | `g4f.Provider.OpenRouter`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [labs.perplexity.ai](https://labs.perplexity.ai) | `g4f.Provider.PerplexityLabs`| ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [pi.ai](https://pi.ai/talk) | `g4f.Provider.Pi`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [replicate.com](https://replicate.com) | `g4f.Provider.Replicate`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [theb.ai](https://theb.ai) | `g4f.Provider.ThebApi`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [whiterabbitneo.com](https://www.whiterabbitneo.com) | `g4f.Provider.WhiteRabbitNeo`| ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [bard.google.com](https://bard.google.com) | `g4f.Provider.Bard`| ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |

### Models

| Model | Base Provider | Provider | Website |
| ----- | ------------- | -------- | ------- |
| gpt-3.5-turbo | OpenAI | 8+ Providers | [openai.com](https://openai.com/) |
| gpt-4 | OpenAI | 2+ Providers | [openai.com](https://openai.com/) |
| gpt-4-turbo | OpenAI | g4f.Provider.Bing | [openai.com](https://openai.com/) |
| Llama-2-7b-chat-hf | Meta | 2+ Providers | [llama.meta.com](https://llama.meta.com/) |
| Llama-2-13b-chat-hf | Meta | 2+ Providers | [llama.meta.com](https://llama.meta.com/) |
| Llama-2-70b-chat-hf | Meta | 3+ Providers | [llama.meta.com](https://llama.meta.com/) |
| Meta-Llama-3-8b-instruct | Meta | 1+ Providers | [llama.meta.com](https://llama.meta.com/) |
| Meta-Llama-3-70b-instruct | Meta | 2+ Providers | [llama.meta.com](https://llama.meta.com/) |
| CodeLlama-34b-Instruct-hf | Meta | g4f.Provider.HuggingChat | [llama.meta.com](https://llama.meta.com/) |
| CodeLlama-70b-Instruct-hf | Meta | 2+ Providers | [llama.meta.com](https://llama.meta.com/) |
| Mixtral-8x7B-Instruct-v0.1 | Huggingface | 4+ Providers | [huggingface.co](https://huggingface.co/) |
| Mistral-7B-Instruct-v0.1 | Huggingface | 3+ Providers | [huggingface.co](https://huggingface.co/) |
| Mistral-7B-Instruct-v0.2 | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| zephyr-orpo-141b-A35b-v0.1 | Huggingface | 2+ Providers | [huggingface.co](https://huggingface.co/) |
| dolphin-2.6-mixtral-8x7b | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| gemini | Google | g4f.Provider.Gemini | [gemini.google.com](https://gemini.google.com/) |
| gemini-pro | Google | 2+ Providers | [gemini.google.com](https://gemini.google.com/) |
| claude-v2 | Anthropic | 1+ Providers | [anthropic.com](https://www.anthropic.com/) |
| claude-3-opus | Anthropic | g4f.Provider.You | [anthropic.com](https://www.anthropic.com/) |
| claude-3-sonnet | Anthropic | g4f.Provider.You | [anthropic.com](https://www.anthropic.com/) |
| lzlv_70b_fp16_hf | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| airoboros-70b | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| openchat_3.5 | Huggingface | 2+ Providers | [huggingface.co](https://huggingface.co/) |
| pi | Inflection | g4f.Provider.Pi | [inflection.ai](https://inflection.ai/) |

### Image and Vision Models

| Label | Provider | Image Model | Vision Model | Website |
| ----- | -------- | ----------- | ------------ | ------- |
| Microsoft Copilot in Bing | `g4f.Provider.Bing` | dall-e-3 | gpt-4-vision | [bing.com](https://bing.com/chat) |
| DeepInfra | `g4f.Provider.DeepInfra` | stability-ai/sdxl | llava-1.5-7b-hf | [deepinfra.com](https://deepinfra.com) |
| Gemini | `g4f.Provider.Gemini` | ✔️ | ✔️ | [gemini.google.com](https://gemini.google.com) |
| Gemini API | `g4f.Provider.GeminiPro` | ❌ | gemini-1.5-pro | [ai.google.dev](https://ai.google.dev) |
| Meta AI | `g4f.Provider.MetaAI` | ✔️ | ❌ | [meta.ai](https://www.meta.ai) |
| OpenAI ChatGPT | `g4f.Provider.OpenaiChat` | dall-e-3 | gpt-4-vision | [chatgpt.com](https://chatgpt.com) |
| Reka | `g4f.Provider.Reka` | ❌ | ✔️ | [chat.reka.ai](https://chat.reka.ai/) |
| Replicate | `g4f.Provider.Replicate` | stability-ai/sdxl| llava-v1.6-34b | [replicate.com](https://replicate.com) |
| You.com | `g4f.Provider.You` | dall-e-3| ✔️ | [you.com](https://you.com) |


## 🔗 Powered by gpt4free

<table>
  <thead align="center">
    <tr border: none;>
      <td>
        <b>🎁 Projects</b>
      </td>
      <td>
        <b>⭐ Stars</b>
      </td>
      <td>
        <b>📚 Forks</b>
      </td>
      <td>
        <b>🛎 Issues</b>
      </td>
      <td>
        <b>📬 Pull requests</b>
      </td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <a href="https://github.com/xtekky/gpt4free">
          <b>gpt4free</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/gpt4free/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/xtekky/gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/gpt4free/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/xtekky/gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/gpt4free/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/xtekky/gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/gpt4free/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xtekky/gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <td>
      <a href="https://github.com/xiangsx/gpt4free-ts">
        <b>gpt4free-ts</b>
      </a>
    </td>
    <td>
      <a href="https://github.com/xiangsx/gpt4free-ts/stargazers">
        <img alt="Stars" src="https://img.shields.io/github/stars/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41" />
      </a>
    </td>
    <td>
      <a href="https://github.com/xiangsx/gpt4free-ts/network/members">
        <img alt="Forks" src="https://img.shields.io/github/forks/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41" />
      </a>
    </td>
    <td>
      <a href="https://github.com/xiangsx/gpt4free-ts/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41" />
      </a>
    </td>
    <td>
      <a href="https://github.com/xiangsx/gpt4free-ts/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41" />
      </a>
    </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/zukixa/cool-ai-stuff/">
          <b>Free AI API's & Potential Providers List</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/zukixa/cool-ai-stuff/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/zukixa/cool-ai-stuff/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/zukixa/cool-ai-stuff/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/zukixa/cool-ai-stuff/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
    <tr>
      <td>
        <a href="https://github.com/xtekky/chatgpt-clone">
          <b>ChatGPT-Clone</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/chatgpt-clone/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/chatgpt-clone/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/chatgpt-clone/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/xtekky/chatgpt-clone/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free">
          <b>Ai agent</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Josh-XT/AGiXT/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Josh-XT/AGiXT/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Josh-XT/AGiXT/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Josh-XT/AGiXT/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free">
          <b>ChatGpt Discord Bot</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/mishalhossin/Coding-Chatbot-Gpt4Free/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    <tr>
    <tr>
      <td>
        <a href="https://github.com/Zero6992/chatGPT-discord-bot">
          <b>chatGPT-discord-bot</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Zero6992/chatGPT-discord-bot/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/Zero6992/chatGPT-discord-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Zero6992/chatGPT-discord-bot/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/Zero6992/chatGPT-discord-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Zero6992/chatGPT-discord-bot/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/Zero6992/chatGPT-discord-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Zero6992/chatGPT-discord-bot/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Zero6992/chatGPT-discord-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    <tr>
      <td>
        <a href="https://github.com/SamirXR/Nyx-Bot">
          <b>Nyx-Bot (Discord)</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/SamirXR/Nyx-Bot/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/SamirXR/Nyx-Bot/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/SamirXR/Nyx-Bot/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/SamirXR/Nyx-Bot/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/MIDORIBIN/langchain-gpt4free">
          <b>LangChain gpt4free</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/MIDORIBIN/langchain-gpt4free/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/MIDORIBIN/langchain-gpt4free/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/MIDORIBIN/langchain-gpt4free/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/MIDORIBIN/langchain-gpt4free/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free">
          <b>ChatGpt Telegram Bot</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot">
          <b>ChatGpt Line Bot</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/Lin-jun-xiang/action-translate-readme">
          <b>Action Translate Readme</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/action-translate-readme/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/action-translate-readme/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/action-translate-readme/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/action-translate-readme/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/Lin-jun-xiang/docGPT-streamlit">
          <b>Langchain Document GPT</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
    <tr>
      <td>
        <a href="https://github.com/Simatwa/python-tgpt">
          <b>python-tgpt</b>
        </a>
      </td>
      <td>
        <a href="https://github.com/Simatwa/python-tgpt/stargazers">
          <img alt="Stars" src="https://img.shields.io/github/stars/Simatwa/python-tgpt?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Simatwa/python-tgpt/network/members">
          <img alt="Forks" src="https://img.shields.io/github/forks/Simatwa/python-tgpt?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Simatwa/python-tgpt/issues">
          <img alt="Issues" src="https://img.shields.io/github/issues/Simatwa/python-tgpt?style=flat-square&labelColor=343b41" />
        </a>
      </td>
      <td>
        <a href="https://github.com/Simatwa/python-tgpt/pulls">
          <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Simatwa/python-tgpt?style=flat-square&labelColor=343b41" />
        </a>
      </td>
    </tr>
  </tbody>
</table>

## 🤝 Contribute

We welcome contributions from the community. Whether you're adding new providers or features, or simply fixing typos and making small improvements, your input is valued. Creating a pull request is all it takes – our co-pilot will handle the code review process. Once all changes have been addressed, we'll merge the pull request into the main branch and release the updates at a later time.

###### Guide: How do i create a new Provider?

 - Read: [/docs/guides/create_provider](https://github.com/xtekky/gpt4free/blob/main/docs/guides/create_provider.md)

###### Guide: How can AI help me with writing code?

 - Read: [/docs/guides/help_me](https://github.com/xtekky/gpt4free/blob/main/docs/guides/help_me.md)

## 🙌 Contributors

A list of all contributors is available [here](https://github.com/xtekky/gpt4free/graphs/contributors)

<a href="https://github.com/xtekky" target="_blank"><img src="https://avatars.githubusercontent.com/u/98614666?v=4&s=45" width="45" title="xtekky"></a>
<a href="https://github.com/hlohaus" target="_blank"><img src="https://avatars.githubusercontent.com/u/983577?v=4&s=45" width="45" title="hlohaus"></a>
<a href="https://github.com/bagusindrayana" target="_blank"><img src="https://avatars.githubusercontent.com/u/36830534?v=4&s=45" width="45" title="bagusindrayana"></a>
<a href="https://github.com/sudouser777" target="_blank"><img src="https://avatars.githubusercontent.com/u/22415463?v=4&s=45" width="45" title="sudouser777"></a>
<a href="https://github.com/thatlukinhasguy1" target="_blank"><img src="https://avatars.githubusercontent.com/u/139662282?v=4&s=45" width="45" title="thatlukinhasguy1"></a>
<a href="https://github.com/Commenter123321" target="_blank"><img src="https://avatars.githubusercontent.com/u/36051603?v=4&s=45" width="45" title="Commenter123321"></a>
<a href="https://github.com/DanielShemesh" target="_blank"><img src="https://avatars.githubusercontent.com/u/20585236?v=4&s=45" width="45" title="DanielShemesh"></a>
<a href="https://github.com/Luneye" target="_blank"><img src="https://avatars.githubusercontent.com/u/73485421?v=4&s=45" width="45" title="Luneye"></a>
<a href="https://github.com/ezerinz" target="_blank"><img src="https://avatars.githubusercontent.com/u/100193740?v=4&s=45" width="45" title="ezerinz"></a>
<a href="https://github.com/enganese" target="_blank"><img src="https://avatars.githubusercontent.com/u/69082498?v=4&s=45" width="45" title="enganese"></a>
<a href="https://github.com/Lin-jun-xiang" target="_blank"><img src="https://avatars.githubusercontent.com/u/63782903?v=4&s=45" width="45" title="Lin-jun-xiang"></a>
<a href="https://github.com/nullstreak" target="_blank"><img src="https://avatars.githubusercontent.com/u/139914347?v=4&s=45" width="45" title="nullstreak"></a>
<a href="https://github.com/valerii-chirkov" target="_blank"><img src="https://avatars.githubusercontent.com/u/81074936?v=4&s=45" width="45" title="valerii-chirkov"></a>
<a href="https://github.com/MIDORIBIN" target="_blank"><img src="https://avatars.githubusercontent.com/u/25425217?v=4&s=45" width="45" title="MIDORIBIN"></a>
<a href="https://github.com/repollo" target="_blank"><img src="https://avatars.githubusercontent.com/u/2671466?v=4&s=45" width="45" title="repollo"></a>
<a href="https://github.com/hpsj" target="_blank"><img src="https://avatars.githubusercontent.com/u/54535414?v=4&s=45" width="45" title="hpsj"></a>
<a href="https://github.com/taiyi747" target="_blank"><img src="https://avatars.githubusercontent.com/u/63543716?v=4&s=45" width="45" title="taiyi747"></a>
<a href="https://github.com/9fo" target="_blank"><img src="https://avatars.githubusercontent.com/u/71867245?v=4&s=45" width="45" title="9fo"></a>
<a href="https://github.com/HexyeDEV" target="_blank"><img src="https://avatars.githubusercontent.com/u/65314629?v=4&s=45" width="45" title="HexyeDEV"></a>
<a href="https://github.com/WdR-Tech" target="_blank"><img src="https://avatars.githubusercontent.com/u/143020293?v=4&s=45" width="45" title="WdR-Tech"></a>
<a href="https://github.com/ostix360" target="_blank"><img src="https://avatars.githubusercontent.com/u/55257054?v=4&s=45" width="45" title="ostix360"></a>
<a href="https://github.com/devAdityaa" target="_blank"><img src="https://avatars.githubusercontent.com/u/77636021?v=4&s=45" width="45" title="devAdityaa"></a>
<a href="https://github.com/editor-syntax" target="_blank"><img src="https://avatars.githubusercontent.com/u/109844019?v=4&s=45" width="45" title="editor-syntax"></a>
<a href="https://github.com/zeng-rr" target="_blank"><img src="https://avatars.githubusercontent.com/u/47846202?v=4&s=45" width="45" title="zeng-rr"></a>
<a href="https://github.com/naa7" target="_blank"><img src="https://avatars.githubusercontent.com/u/44613678?v=4&s=45" width="45" title="naa7"></a>
<a href="https://github.com/ramonvc" target="_blank"><img src="https://avatars.githubusercontent.com/u/13617054?v=4&s=45" width="45" title="ramonvc"></a>
<a href="https://github.com/eltociear" target="_blank"><img src="https://avatars.githubusercontent.com/u/22633385?v=4&s=45" width="45" title="eltociear"></a>
<a href="https://github.com/kggn" target="_blank"><img src="https://avatars.githubusercontent.com/u/95663228?v=4&s=45" width="45" title="kggn"></a>
<a href="https://github.com/xiangsx" target="_blank"><img src="https://avatars.githubusercontent.com/u/29322721?v=4&s=45" width="45" title="xiangsx"></a>
<a href="https://github.com/ggindinson" target="_blank"><img src="https://avatars.githubusercontent.com/u/97807772?v=4&s=45" width="45" title="ggindinson"></a>
<span></span>
<img src="https://avatars.githubusercontent.com/u/71154407?s=45&v=4" width="45" title="ading2210">
<img src="https://avatars.githubusercontent.com/u/12299238?s=45&v=4" width="45" title="xqdoo00o">
<img src="https://avatars.githubusercontent.com/u/97126670?s=45&v=4" width="45" title="nathanrchn">
<img src="https://avatars.githubusercontent.com/u/81407603?v=4&s=45" width="45" title="dsdanielpark">
<img src="https://avatars.githubusercontent.com/u/55200481?v=4&s=45" width="45" title="missuo">

- The [`Vercel.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/Vercel.py) file contains code from [vercel-llm-api](https://github.com/ading2210/vercel-llm-api) by [@ading2210](https://github.com/ading2210)
- The [`har_file.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/openai/har_file.py) has input from [xqdoo00o/ChatGPT-to-API](https://github.com/xqdoo00o/ChatGPT-to-API)
- The [`PerplexityLabs.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/openai/har_file.py) has input from [nathanrchn/perplexityai](https://github.com/nathanrchn/perplexityai)
- The [`Gemini.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/needs_auth/Gemini.py) has input from [dsdanielpark/Gemini-API](https://github.com/dsdanielpark/Gemini-API)
- The [`MetaAI.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/MetaAI.py) file contains code from [meta-ai-api](https://github.com/Strvm/meta-ai-api) by [@Strvm](https://github.com/Strvm)
- The [`proofofwork.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/openai/proofofwork.py) has input from [missuo/FreeGPT35](https://github.com/missuo/FreeGPT35)

*Having input implies that the AI's code generation utilized it as one of many sources.*

## ©️ Copyright

This program is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)

```
xtekky/gpt4free: Copyright (C) 2023 xtekky

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

## ⭐ Star History

<a href="https://github.com/xtekky/gpt4free/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=xtekky/gpt4free&type=Date">
</a>

## 📄 License

<table>
  <tr>
     <td>
       <p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/GPLv3_Logo.svg/1200px-GPLv3_Logo.svg.png" width="80%"></img>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/License-GNU_GPL_v3.0-red.svg"/> <br> 
This project is licensed under <a href="https://github.com/xtekky/gpt4free/blob/main/LICENSE">GNU_GPL_v3.0</a>.
    </td>
  </tr>
</table>

<p align="right">(<a href="#top">🔼 Back to top</a>)</p>
