![248433934-7886223b-c1d1-4260-82aa-da5741f303bb](https://github.com/xtekky/gpt4free/assets/98614666/ea012c87-76e0-496a-8ac4-e2de090cc6c9)
Written by [@xtekky](https://github.com/hlohaus) & maintained by [@hlohaus](https://github.com/hlohaus)

<div id="top"></div>

> By using this repository or any code related to it, you agree to the [legal notice](LEGAL_NOTICE.md). The author is not responsible for any copies, forks, re-uploads made by other users, or anything else related to GPT4Free. This is the author's only account and repository. To prevent impersonation or irresponsible actions, please comply with the GNU GPL license this Repository uses.

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
- Join our Telegram Channel: [t.me/g4f_channel](https://telegram.me/g4f_channel)
- Join our Discord Group: [discord.gg/XfybzPXPH5](https://discord.gg/XfybzPXPH5)
- Explore the g4f Documentation (unfinished): [g4f.mintlify.app](https://g4f.mintlify.app) | Contribute to the docs via: [github.com/xtekky/gpt4free-docs](https://github.com/xtekky/gpt4free-docs)

# Feedback
You can always leave some feedback here: https://forms.gle/FeWV9RLEedfdkmFN6

# To do
As per the survey, here is a list of improvements to come
- [ ] golang implementation
- [ ] Improve Documentation (on g4f.mintlify.app) & Do video tutorials
- [ ] Improve the provider status list & updates
- [ ] Tutorials on how to reverse sites to write your own wrapper (PoC only ofc)
- [ ] Improve the Bing wrapper. (might write a new wrapper in golang as it is very fast)
- [ ] Write a standard provider performance test to improve the stability
- [ ] update the repository to include the new openai library syntax (ex: `Openai()` class)
- [ ] Potential support and development of local models
- [ ] improve compatibility and error handling


## 📚 Table of Contents

- [🆕 What's New](#-whats-new)
- [📚 Table of Contents](#-table-of-contents)
- [🛠️ Getting Started](#-getting-started)
    + [Docker container](#docker-container)
      - [Quick start](#quick-start)
    + [Use python package](#use-python-package)
      - [Prerequisites](#prerequisites)
      - [Install using pypi](#install-using-pypi)
    + [Docker for Developers](#docker-for-developers)
- [💡 Usage](#-usage)
  * [The Web UI](#the-web-ui)
  * [The `g4f` Package](#the-g4f-package)
    + [ChatCompletion](#chatcompletion)
      - [Completion](#completion)
      - [Providers](#providers)
      - [Using Browser](#using-browser)
      - [Async Support](#async-support)
      - [Proxy and Timeout Support](#proxy-and-timeout-support)
  * [Interference openai-proxy API](#interference-openai-proxy-api-use-with-openai-python-package-)
    + [Run interference API from PyPi package](#run-interference-api-from-pypi-package)
    + [Run interference API from repo](#run-interference-api-from-repo)
- [🚀 Providers and Models](#-providers-and-models)
  * [GPT-4](#gpt-4)
  * [GPT-3.5](#gpt-35)
  * [Other](#other)
  * [Models](#models)
- [🔗 Related GPT4Free Projects](#-related-gpt4free-projects)
- [🤝 Contribute](#-contribute)
    + [Create Provider with AI Tool](#create-provider-with-ai-tool)
    + [Create Provider](#create-provider)
- [🙌 Contributors](#-contributors)
- [©️ Copyright](#-copyright)
- [⭐ Star History](#-star-history)
- [📄 License](#-license)

## 🛠️ Getting Started

#### Docker container

##### Quick start:

1. [Download and install Docker](https://docs.docker.com/get-docker/)
2. Pull latest image and run the container:

```sh
docker pull hlohaus789/g4f
docker run -p 8080:8080 -p 1337:1337 -p 7900:7900 --shm-size="2g" hlohaus789/g4f:latest
```
3. Open the included client on: [http://localhost:8080/chat/](http://localhost:8080/chat/)
or set the api base in your client to: [http://localhost:1337/v1](http://localhost:1337/v1)
4. (Optional) If you need to log in to a provider, you can view the desktop from the container here: http://localhost:7900/?autoconnect=1&resize=scale&password=secret.

#### Use python package

##### Prerequisites:

1. [Download and install Python](https://www.python.org/downloads/) (Version 3.10+ is recommended).
2. [Install Google Chrome](https://www.google.com/chrome/) for providers with webdriver

##### Install using pypi:

Install all supported tools / all used packages:
```
pip install -U g4f[all]
```
Install required packages for the OpenaiChat provider:
```
pip install -U g4f[openai]
```
Install required packages for the interference api:
```
pip install -U g4f[api]
```
Install required packages for the web interface:
```
pip install -U g4f[gui]
```
Install required packages for uploading / generating images:
```
pip install -U g4f[image]
```
Install required packages for providers with webdriver:
```
pip install -U g4f[webdriver]
```
Install required packages for proxy support:
```
pip install -U aiohttp_socks
```

##### or:

1. Clone the GitHub repository:

```
git clone https://github.com/xtekky/gpt4free.git
```

2. Navigate to the project directory:

```
cd gpt4free
```

3. (Recommended) Create a Python virtual environment:
You can follow the [Python official documentation](https://docs.python.org/3/tutorial/venv.html) for virtual environments.


```
python3 -m venv venv
```

4. Activate the virtual environment:
   - On Windows:
   ```
   .\venv\Scripts\activate
   ```
   - On macOS and Linux:
   ```
   source venv/bin/activate
   ```
5. Install minimum requirements:

```
pip install -r requirements-min.txt
```

6. Or install all used Python packages from `requirements.txt`:

```
pip install -r requirements.txt
```

7. Create a `test.py` file in the root folder and start using the repo, further Instructions are below

```py
import g4f
...
```

#### Docker for Developers

If you have Docker installed, you can easily set up and run the project without manually installing dependencies.

1. First, ensure you have both Docker and Docker Compose installed.

   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

2. Clone the GitHub repo:

```bash
git clone https://github.com/xtekky/gpt4free.git
```

3. Navigate to the project directory:

```bash
cd gpt4free
```

4. Build the Docker image:

```bash
docker pull selenium/node-chrome
docker-compose build
```

5. Start the service using Docker Compose:

```bash
docker-compose up
```

Your server will now be running at `http://localhost:1337`. You can interact with the API or run your tests as you would normally.

To stop the Docker containers, simply run:

```bash
docker-compose down
```

> [!Note]
> When using Docker, any changes you make to your local files will be reflected in the Docker container thanks to the volume mapping in the `docker-compose.yml` file. If you add or remove dependencies, however, you'll need to rebuild the Docker image using `docker-compose build`.

## 💡 Usage

### New Client with Image Generation

#### not jet released

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
Result:

[![Image with cat](/docs/cat.jpeg)](/docs/client.md)

[to the client API](/docs/client.md)

### The Web UI

To start the web interface, type the following codes in the command line.

```python
from g4f.gui import run_gui
run_gui()
```

### The `g4f` Package

#### ChatCompletion

```python
import g4f

g4f.debug.logging = True  # Enable debug logging
g4f.debug.version_check = False  # Disable automatic version checking
print(g4f.Provider.Bing.params)  # Print supported args for Bing

# Using automatic a provider for the given model
## Streamed completion
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for message in response:
    print(message, flush=True, end='')

## Normal response
response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    messages=[{"role": "user", "content": "Hello"}],
)  # Alternative model setting

print(response)
```

##### Completion

```python
import g4f

allowed_models = [
    'code-davinci-002',
    'text-ada-001',
    'text-babbage-001',
    'text-curie-001',
    'text-davinci-002',
    'text-davinci-003'
]

response = g4f.Completion.create(
    model='text-davinci-003',
    prompt='say this is a test'
)

print(response)
```

##### Providers

```python
import g4f

# Print all available providers
print([
    provider.__name__
    for provider in g4f.Provider.__providers__
    if provider.working
])

# Execute with a specific provider
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    provider=g4f.Provider.Aichat,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for message in response:
    print(message)
```

##### Cookies / Access Token

For generating images with Bing and for the OpenAi Chat  you need cookies or a token from your browser session. From Bing you need the "_U" cookie and from OpenAI you need the "access_token". You can pass the cookies / the  access token in the create function or you use the `set_cookies` setter:

```python
from g4f import set_cookies

set_cookies(".bing.com", {
  "_U": "cookie value"
})
set_cookies("chat.openai.com", {
  "access_token": "token value"
})

from g4f.gui import run_gui
run_gui()
```

Alternatively, g4f reads the cookies with “browser_cookie3” from your browser
or it starts a browser instance with selenium "webdriver" for logging in.
If you use the pip package, you have to install “browser_cookie3” or "webdriver" by yourself.

```bash
pip install browser_cookie3
pip install g4f[webdriver]
```

##### Image Upload & Generation

Image upload and generation are supported by three main providers:

- **Bing & Other GPT-4 Providers:** Utilizes Microsoft's Image Creator.
- **Google Gemini:** Available for free accounts with IP addresses outside Europe.
- **OpenaiChat with GPT-4:** Accessible for users with a Plus subscription.

```python
import g4f

# Setting up the request for image creation
response = g4f.ChatCompletion.create(
    model=g4f.models.default, # Using the default model
    provider=g4f.Provider.Gemini, # Specifying the provider as Gemini
    messages=[{"role": "user", "content": "Create an image like this"}],
    image=open("images/g4f.png", "rb"), # Image input can be a data URI, bytes, PIL Image, or IO object
    image_name="g4f.png" # Optional: specifying the filename
)

# Displaying the response
print(response)

from g4f.image import ImageResponse

# Get image links from response
for chunk in g4f.ChatCompletion.create(
    model=g4f.models.default, # Using the default model
    provider=g4f.Provider.OpenaiChat, # Specifying the provider as OpenaiChat
    messages=[{"role": "user", "content": "Create images with dogs"}],
    access_token="...", # Need a access token from a plus user
    stream=True,
    ignore_stream=True
):
    if isinstance(chunk, ImageResponse):
        print(chunk.images) # Print generated image links
        print(chunk.alt) # Print used prompt for image generation
```

##### Using Browser

Some providers using a browser to bypass the bot protection. They using the selenium webdriver to control the browser. The browser settings and the login data are saved in a custom directory. If the headless mode is enabled, the browser windows are loaded invisibly. For performance reasons, it is recommended to reuse the browser instances and close them yourself at the end:

```python
import g4f
from undetected_chromedriver import Chrome, ChromeOptions
from g4f.Provider import (
    Bard,
    Poe,
    AItianhuSpace,
    MyShell,
    PerplexityAi,
)

options = ChromeOptions()
options.add_argument("--incognito");
webdriver = Chrome(options=options, headless=True)
for idx in range(10):
    response = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.MyShell,
        messages=[{"role": "user", "content": "Suggest me a name."}],
        webdriver=webdriver
    )
    print(f"{idx}:", response)
webdriver.quit()
```

##### Async Support

To enhance speed and overall performance, execute providers asynchronously. The total execution time will be determined by the duration of the slowest provider's execution.

```python
import g4f
import asyncio

_providers = [
    g4f.Provider.Aichat,
    g4f.Provider.ChatBase,
    g4f.Provider.Bing,
    g4f.Provider.GptGo,
    g4f.Provider.You,
    g4f.Provider.Yqcloud,
]

async def run_provider(provider: g4f.Provider.BaseProvider):
    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.default,
            messages=[{"role": "user", "content": "Hello"}],
            provider=provider,
        )
        print(f"{provider.__name__}:", response)
    except Exception as e:
        print(f"{provider.__name__}:", e)
        
async def run_all():
    calls = [
        run_provider(provider) for provider in _providers
    ]
    await asyncio.gather(*calls)

asyncio.run(run_all())
```

##### Proxy and Timeout Support

All providers support specifying a proxy and increasing timeout in the create functions.

```python
import g4f

response = g4f.ChatCompletion.create(
    model=g4f.models.default,
    messages=[{"role": "user", "content": "Hello"}],
    proxy="http://host:port",
    # or socks5://user:pass@host:port
    timeout=120,  # in secs
)

print(f"Result:", response)
```

You can also set a proxy globally via an environment variable:

```sh
export G4F_PROXY="http://host:port"
```

### Interference openai-proxy API (Use with openai python package)

#### Run interference API from PyPi package

```python
from g4f.api import run_api

run_api()
```

#### Run interference API from repo

If you want to use the embedding function, you need to get a Hugging Face token. You can get one at [Hugging Face Tokens](https://huggingface.co/settings/tokens). Make sure your role is set to write. If you have your token, just use it instead of the OpenAI api-key.

Run server:

```sh
g4f api
```

or

```sh
python -m g4f.api.run
```

```python
from openai import OpenAI

client = OpenAI(
    # Set your Hugging Face token as the API key if you use embeddings
    api_key="YOUR_HUGGING_FACE_TOKEN",

    # Set the API base URL if needed, e.g., for a local development environment
    base_url="http://localhost:1337/v1"
)


def main():
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "write a poem about a tree"}],
        stream=True,
    )

    if isinstance(chat_completion, dict):
        # Not streaming
        print(chat_completion.choices[0].message.content)
    else:
        # Streaming
        for token in chat_completion:
            content = token.choices[0].delta.content
            if content is not None:
                print(content, end="", flush=True)


if __name__ == "__main__":
    main()
```

##  API usage (POST)
#### Chat completions
Send the POST request to /v1/chat/completions with body containing the `model` method. This example uses python with requests library:
```python
import requests
url = "http://localhost:1337/v1/chat/completions"
body = {
    "model": "gpt-3.5-turbo-16k",
    "stream": False,
    "messages": [
        {"role": "assistant", "content": "What can you do?"}
    ]
}
json_response = requests.post(url, json=body).json().get('choices', [])

for choice in json_response:
    print(choice.get('message', {}).get('content', ''))
```


## 🚀 Providers and Models

### GPT-4

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bing.com](https://bing.com/chat) | `g4f.Provider.Bing` | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [free.chatgpt.org.uk](https://free.chatgpt.org.uk) | `g4f.Provider.FreeChatgpt` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [liaobots.site](https://liaobots.site) | `g4f.Provider.Liaobots` | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.openai.com](https://chat.openai.com) | `g4f.Provider.OpenaiChat` | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [raycast.com](https://raycast.com) | `g4f.Provider.Raycast` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [beta.theb.ai](https://beta.theb.ai) | `g4f.Provider.Theb` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.geekgpt.org](https://chat.geekgpt.org) | `g4f.Provider.GeekGpt` | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |

### GPT-3.5

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [chat3.aiyunos.top](https://chat3.aiyunos.top/) | `g4f.Provider.AItianhuSpace` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [aichatonline.org](https://aichatonline.org) | `g4f.Provider.AiChatOnline` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [openchat.team](https://openchat.team) | `g4f.Provider.Aura` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatbase.co](https://www.chatbase.co) | `g4f.Provider.ChatBase` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatforai.store](https://chatforai.store) | `g4f.Provider.ChatForAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt.ai](https://chatgpt.ai) | `g4f.Provider.ChatgptAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.chatgptdemo.net](https://chat.chatgptdemo.net) | `g4f.Provider.ChatgptDemo` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt-free.cc](https://www.chatgpt-free.cc) | `g4f.Provider.ChatgptNext` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.3211000.xyz](https://chat.3211000.xyz) | `g4f.Provider.Chatxyz` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gptalk.net](https://gptalk.net) | `g4f.Provider.GPTalk` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [geminiprochat.com](https://geminiprochat.com) | `g4f.Provider.GeminiProChat` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gpt6.ai](https://gpt6.ai) | `g4f.Provider.Gpt6` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gptchatly.com](https://gptchatly.com) | `g4f.Provider.GptChatly` | ✔️ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [ai18.gptforlove.com](https://ai18.gptforlove.com) | `g4f.Provider.GptForLove` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gptgo.ai](https://gptgo.ai) | `g4f.Provider.GptGo` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gpttalk.ru](https://gpttalk.ru) | `g4f.Provider.GptTalkRu` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [koala.sh](https://koala.sh) | `g4f.Provider.Koala` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [app.myshell.ai](https://app.myshell.ai/chat) | `g4f.Provider.MyShell` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [onlinegpt.org](https://onlinegpt.org) | `g4f.Provider.OnlineGpt` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [perplexity.ai](https://www.perplexity.ai) | `g4f.Provider.PerplexityAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [poe.com](https://poe.com) | `g4f.Provider.Poe` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [talkai.info](https://talkai.info) | `g4f.Provider.TalkAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [you.com](https://you.com) | `g4f.Provider.You` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [aitianhu.com](https://www.aitianhu.com) | `g4f.Provider.AItianhu` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [e.aiask.me](https://e.aiask.me) | `g4f.Provider.AiAsk` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgpt.bestim.org](https://chatgpt.bestim.org) | `g4f.Provider.Bestim` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatanywhere.cn](https://chatanywhere.cn) | `g4f.Provider.ChatAnywhere` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgpt4online.org](https://chatgpt4online.org) | `g4f.Provider.Chatgpt4Online` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.chatgptdemo.ai](https://chat.chatgptdemo.ai) | `g4f.Provider.ChatgptDemoAi` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptfree.ai](https://chatgptfree.ai) | `g4f.Provider.ChatgptFree` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptlogin.ai](https://chatgptlogin.ai) | `g4f.Provider.ChatgptLogin` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptx.de](https://chatgptx.de) | `g4f.Provider.ChatgptX` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat-shared2.zhile.io](https://chat-shared2.zhile.io) | `g4f.Provider.FakeGpt` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [freegpts1.aifree.site](https://freegpts1.aifree.site/) | `g4f.Provider.FreeGpt` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gptgod.site](https://gptgod.site) | `g4f.Provider.GptGod` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [hashnode.com](https://hashnode.com) | `g4f.Provider.Hashnode` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [sdk.vercel.ai](https://sdk.vercel.ai) | `g4f.Provider.Vercel` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.ylokh.xyz](https://chat.ylokh.xyz) | `g4f.Provider.Ylokh` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |

### Other

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bard.google.com](https://bard.google.com) | `g4f.Provider.Bard` | ❌ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [deepinfra.com](https://deepinfra.com) | `g4f.Provider.DeepInfra` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gemini.google.com](https://gemini.google.com) | `g4f.Provider.Gemini` | ❌ | ❌ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [huggingface.co](https://huggingface.co/chat) | `g4f.Provider.HuggingChat` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [llama2.ai](https://www.llama2.ai) | `g4f.Provider.Llama2` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [labs.perplexity.ai](https://labs.perplexity.ai) | `g4f.Provider.PerplexityLabs` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [phind.com](https://www.phind.com) | `g4f.Provider.Phind` | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [pi.ai](https://pi.ai/talk) | `g4f.Provider.Pi` | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [theb.ai](https://theb.ai) | `g4f.Provider.ThebApi` | ❌ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [open-assistant.io](https://open-assistant.io/chat) | `g4f.Provider.OpenAssistant` | ❌ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |

### Models

| Model | Base Provider | Provider | Website |
| ----- | ------------- | -------- | ------- |
| gpt-3.5-turbo | OpenAI | 5+ Providers | [openai.com](https://openai.com/) |
| gpt-4 | OpenAI | 2+ Providers | [openai.com](https://openai.com/) |
| gpt-4-turbo | OpenAI | g4f.Provider.Bing | [openai.com](https://openai.com/) |
| Llama-2-7b-chat-hf | Huggingface | 2+ Providers | [huggingface.co](https://huggingface.co/) |
| Llama-2-13b-chat-hf | Huggingface | 2+ Providers | [huggingface.co](https://huggingface.co/) |
| Llama-2-70b-chat-hf | Huggingface | 4+ Providers | [huggingface.co](https://huggingface.co/) |
| CodeLlama-34b-Instruct-hf | Huggingface | 3+ Providers | [huggingface.co](https://huggingface.co/) |
| Mixtral-8x7B-Instruct-v0.1 | Huggingface | 3+ Providers | [huggingface.co](https://huggingface.co/) |
| Mistral-7B-Instruct-v0.1 | Huggingface | 3+ Providers | [huggingface.co](https://huggingface.co/) |
| dolphin-2.6-mixtral-8x7b | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| lzlv_70b_fp16_hf | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| airoboros-70b | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| airoboros-l2-70b-gpt4-1.4.1 | Huggingface | g4f.Provider.DeepInfra | [huggingface.co](https://huggingface.co/) |
| openchat_3.5 | Huggingface | 2+ Providers | [huggingface.co](https://huggingface.co/) |
| gemini | Google | g4f.Provider.Gemini | [gemini.google.com](https://gemini.google.com/) |
| gemini-pro | Google | 2+ Providers | [gemini.google.com](https://gemini.google.com/) |
| claude-v2 | Anthropic | 2+ Providers | [anthropic.com](https://www.anthropic.com/) |
| pi | Inflection | g4f.Provider.Pi | [inflection.ai](https://inflection.ai/) |

## 🔗 Related GPT4Free Projects

<table>
  <thead align="center">
    <tr border: none;>
      <td><b>🎁 Projects</b></td>
      <td><b>⭐ Stars</b></td>
      <td><b>📚 Forks</b></td>
      <td><b>🛎 Issues</b></td>
      <td><b>📬 Pull requests</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/xtekky/gpt4free"><b>gpt4free</b></a></td>
      <td><a href="https://github.com/xtekky/gpt4free/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/xtekky/gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/gpt4free/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/xtekky/gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/gpt4free/issues"><img alt="Issues" src="https://img.shields.io/github/issues/xtekky/gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/gpt4free/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xtekky/gpt4free?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
      <td><a href="https://github.com/xiangsx/gpt4free-ts"><b>gpt4free-ts</b></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/issues"><img alt="Issues" src="https://img.shields.io/github/issues/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
     <tr>
      <td><a href="https://github.com/zukixa/cool-ai-stuff/"><b>Free AI API's & Potential Providers List</b></a></td>
      <td><a href="https://github.com/zukixa/cool-ai-stuff/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/zukixa/cool-ai-stuff/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/zukixa/cool-ai-stuff/issues"><img alt="Issues" src="https://img.shields.io/github/issues/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/zukixa/cool-ai-stuff/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/zukixa/cool-ai-stuff?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
    <tr>
      <td><a href="https://github.com/xtekky/chatgpt-clone"><b>ChatGPT-Clone</b></a></td>
      <td><a href="https://github.com/xtekky/chatgpt-clone/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/chatgpt-clone/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/chatgpt-clone/issues"><img alt="Issues" src="https://img.shields.io/github/issues/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xtekky/chatgpt-clone/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xtekky/chatgpt-clone?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free"><b>ChatGpt Discord Bot</b></a></td>
      <td><a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/mishalhossin/Discord-Chatbot-Gpt4Free/issues"><img alt="Issues" src="https://img.shields.io/github/issues/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/mishalhossin/Coding-Chatbot-Gpt4Free/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/mishalhossin/Discord-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
<tr>
  <td><a href="https://github.com/SamirXR/Nyx-Bot"><b>Nyx-Bot (Discord)</b></a></td>
  <td><a href="https://github.com/SamirXR/Nyx-Bot/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41"/></a></td>
  <td><a href="https://github.com/SamirXR/Nyx-Bot/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41"/></a></td>
  <td><a href="https://github.com/SamirXR/Nyx-Bot/issues"><img alt="Issues" src="https://img.shields.io/github/issues/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41"/></a></td>
  <td><a href="https://github.com/SamirXR/Nyx-Bot/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/SamirXR/Nyx-Bot?style=flat-square&labelColor=343b41"/></a></td>
</tr>
    </tr>
    <tr>
      <td><a href="https://github.com/MIDORIBIN/langchain-gpt4free"><b>LangChain gpt4free</b></a></td>
      <td><a href="https://github.com/MIDORIBIN/langchain-gpt4free/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/MIDORIBIN/langchain-gpt4free/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/MIDORIBIN/langchain-gpt4free/issues"><img alt="Issues" src="https://img.shields.io/github/issues/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/MIDORIBIN/langchain-gpt4free/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/MIDORIBIN/langchain-gpt4free?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free"><b>ChatGpt Telegram Bot</b></a></td>
      <td><a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/issues"><img alt="Issues" src="https://img.shields.io/github/issues/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/HexyeDEV/Telegram-Chatbot-Gpt4Free/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/HexyeDEV/Telegram-Chatbot-Gpt4Free?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
        <tr>
      <td><a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot"><b>ChatGpt Line Bot</b></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/issues"><img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/chatgpt-line-bot/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/chatgpt-line-bot?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/Lin-jun-xiang/action-translate-readme"><b>Action Translate Readme</b></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/action-translate-readme/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/action-translate-readme/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/action-translate-readme/issues"><img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/action-translate-readme/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/action-translate-readme?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/Lin-jun-xiang/docGPT-streamlit"><b>Langchain Document GPT</b></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/issues"><img alt="Issues" src="https://img.shields.io/github/issues/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Lin-jun-xiang/docGPT-streamlit/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Lin-jun-xiang/docGPT-streamlit?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
    <tr>
      <td><a href="https://github.com/Simatwa/python-tgpt"><b>python-tgpt</b></a></td>
      <td><a href="https://github.com/Simatwa/python-tgpt/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Simatwa/python-tgpt?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Simatwa/python-tgpt/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/Simatwa/python-tgpt?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Simatwa/python-tgpt/issues"><img alt="Issues" src="https://img.shields.io/github/issues/Simatwa/python-tgpt?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/Simatwa/python-tgpt/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/Simatwa/python-tgpt?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
  </tbody>
</table>

## 🤝 Contribute

#### Create Provider with AI Tool

Call in your terminal the `create_provider.py` script:
```bash
python etc/tool/create_provider.py
```
1. Enter your name for the new provider.
2. Copy and paste the `cURL` command from your browser developer tools.
3. Let the AI ​​create the provider for you.
4. Customize the provider according to your needs.

#### Create Provider

1. Check out the current [list of potential providers](https://github.com/zukixa/cool-ai-stuff#ai-chat-websites), or find your own provider source!
2. Create a new file in [g4f/Provider](./g4f/Provider) with the name of the Provider
3. Implement a class that extends [BaseProvider](./g4f/Provider/base_provider.py).

```py
from __future__ import annotations

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class HogeService(AsyncGeneratorProvider):
    url                   = "https://chat-gpt.com"
    working               = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        yield ""
```

4. Here, you can adjust the settings, for example, if the website does support streaming, set `supports_stream` to `True`...
5. Write code to request the provider in `create_async_generator` and `yield` the response, _even if_ it's a one-time response, do not hesitate to look at other providers for inspiration
6. Add the Provider Name in [`g4f/Provider/__init__.py`](./g4f/Provider/__init__.py)

```py
from .HogeService import HogeService

__all__ = [
  HogeService,
]
```

7. You are done !, test the provider by calling it:

```py
import g4f

response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.PROVIDERNAME,
                                    messages=[{"role": "user", "content": "test"}], stream=g4f.Provider.PROVIDERNAME.supports_stream)

for message in response:
    print(message, flush=True, end='')
```

## 🙌 Contributors

A list of the contributors is available [here](https://github.com/xtekky/gpt4free/graphs/contributors)   
The [`Vercel.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/Vercel.py) file contains code from [vercel-llm-api](https://github.com/ading2210/vercel-llm-api) by [@ading2210](https://github.com/ading2210), which is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)   
Top 1 Contributor: [@hlohaus](https://github.com/hlohaus)

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
This project is licensed under <a href="./LICENSE">GNU_GPL_v3.0</a>.
    </td>
  </tr>
</table>

<p align="right">(<a href="#top">🔼 Back to top</a>)</p>
