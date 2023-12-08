![248433934-7886223b-c1d1-4260-82aa-da5741f303bb](https://github.com/xtekky/gpt4free/assets/98614666/ea012c87-76e0-496a-8ac4-e2de090cc6c9)

<a href='https://ko-fi.com/xtekky' target='_blank'><img height='35' style='border:0px;height:46px;' src='https://az743702.vo.msecnd.net/cdn/kofi3.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' />
<div id="top"></div>

> By using this repository or any code related to it, you agree to the [legal notice](LEGAL_NOTICE.md). The author is not responsible for any copies, forks, re-uploads made by other users, or anything else related to GPT4Free. This is the author's only account and repository. To prevent impersonation or irresponsible actions, please comply with the GNU GPL license this Repository uses.

> [!Note]
<sup><strong>Lastet version:</strong></sup> [![PyPI version](https://img.shields.io/pypi/v/g4f?color=blue)](https://pypi.org/project/g4f) [![Docker version](https://img.shields.io/docker/v/hlohaus789/g4f?label=docker&color=blue)](https://hub.docker.com/r/hlohaus789/g4f)
```sh
pip install -U g4f
```
```sh
docker pull hlohaus789/g4f
```

## 🆕 What's New
- <a href="./README-DE.md"><img src="https://img.shields.io/badge/öffnen in-🇩🇪 deutsch-bleu.svg" alt="Öffnen en DE"></a>
- Join our Telegram Channel: [t.me/g4f_channel](https://telegram.me/g4f_channel)
- Join our Discord Group: [discord.gg/XfybzPXPH5](https://discord.gg/XfybzPXPH5)
- Explore the g4f Documentation (unfinished): [g4f.mintlify.app](https://g4f.mintlify.app) | Contribute to the docs via: [github.com/xtekky/gpt4free-docs](https://github.com/xtekky/gpt4free-docs)

## 📚 Table of Contents

- [🆕 What's New](#-what-s-new)
- [📚 Table of Contents](#-table-of-contents)
- [🛠️ Getting Started](#-getting-started)
    + [Docker container](#docker-container)
      - [Quick start](#quick-start)
    + [Use python package](#use-python-package)
      - [Prerequisites](#prerequisites)
      - [Install using pypi](#install-using-pypi)
    + [Docker for Developers](#docker-for-developers)
- [💡 Usage](#-usage)
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
2. Pull lastet image and run the container:

```sh
docker pull hlohaus789/g4f
docker run -p 8080:80 -p 1337:1337 -p 7900:7900 --shm-size="2g" hlohaus789/g4f:latest
```
3. Open the included client on: [http://localhost:8080/chat/](http://localhost:8080/chat/)
or set the api base in your client to: [http://localhost:1337/v1](http://localhost:1337/v1)
4. (Optional) If you need to log in to a provider, you can view the desktop from the container here: http://localhost:7900/?autoconnect=1&resize=scale&password=secret.

#### Use python package

##### Prerequisites:

1. [Download and install Python](https://www.python.org/downloads/) (Version 3.10+ is recommended).
2. [Install Google Chrome](https://www.google.com/chrome/) for providers with webdriver

##### Install using pypi:

```
pip install -U g4f
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
5. Install the required Python packages from `requirements.txt`:

```
pip install -r requirements.txt
```

6. Create a `test.py` file in the root folder and start using the repo, further Instructions are below

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

### The `g4f` Package

#### ChatCompletion

```python
import g4f

g4f.debug.logging = True  # Enable debug logging
g4f.debug.check_version = False  # Disable automatic version checking
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

##### Using Browser

Some providers using a a browser to bypass the bot protection. They using the selenium webdriver to control the browser. The browser settings and the login data are saved in a custom directory. If the headless mode is enabled, the browser windows are loaded invisibly. For performance reasons, it is recommended to reuse the browser instances and close them yourself at the end:

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
import openai

# Set your Hugging Face token as the API key if you use embeddings
# If you don't use embeddings, leave it empty
openai.api_key = "YOUR_HUGGING_FACE_TOKEN"  # Replace with your actual token

# Set the API base URL if needed, e.g., for a local development environment
openai.api_base = "http://localhost:1337/v1"

def main():
    chat_completion = openai.ChatCompletion.create(
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
            content = token["choices"][0]["delta"].get("content")
            if content is not None:
                print(content, end="", flush=True)

if __name__ == "__main__":
    main()
```

## 🚀 Providers and Models

### GPT-4

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bing.com](https://bing.com/chat) | `g4f.Provider.Bing` | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.geekgpt.org](https://chat.geekgpt.org) | `g4f.Provider.GeekGpt` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [gptchatly.com](https://gptchatly.com) | `g4f.Provider.GptChatly` | ✔️ | ✔️ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [liaobots.site](https://liaobots.site) | `g4f.Provider.Liaobots` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [www.phind.com](https://www.phind.com) | `g4f.Provider.Phind` | ❌ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [raycast.com](https://raycast.com) | `g4f.Provider.Raycast` | ✔️ | ✔️ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |

### GPT-3.5

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [www.aitianhu.com](https://www.aitianhu.com) | `g4f.Provider.AItianhu` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat3.aiyunos.top](https://chat3.aiyunos.top/) | `g4f.Provider.AItianhuSpace` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [e.aiask.me](https://e.aiask.me) | `g4f.Provider.AiAsk` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat-gpt.org](https://chat-gpt.org/chat) | `g4f.Provider.Aichat` | ✔️ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [www.chatbase.co](https://www.chatbase.co) | `g4f.Provider.ChatBase` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatforai.store](https://chatforai.store) | `g4f.Provider.ChatForAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chatgpt.ai](https://chatgpt.ai) | `g4f.Provider.ChatgptAi` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatgptx.de](https://chatgptx.de) | `g4f.Provider.ChatgptX` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat-shared2.zhile.io](https://chat-shared2.zhile.io) | `g4f.Provider.FakeGpt` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [freegpts1.aifree.site](https://freegpts1.aifree.site/) | `g4f.Provider.FreeGpt` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gptalk.net](https://gptalk.net) | `g4f.Provider.GPTalk` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [ai18.gptforlove.com](https://ai18.gptforlove.com) | `g4f.Provider.GptForLove` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gptgo.ai](https://gptgo.ai) | `g4f.Provider.GptGo` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [hashnode.com](https://hashnode.com) | `g4f.Provider.Hashnode` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [app.myshell.ai](https://app.myshell.ai/chat) | `g4f.Provider.MyShell` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [noowai.com](https://noowai.com) | `g4f.Provider.NoowAi` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.openai.com](https://chat.openai.com) | `g4f.Provider.OpenaiChat` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [theb.ai](https://theb.ai) | `g4f.Provider.Theb` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [sdk.vercel.ai](https://sdk.vercel.ai) | `g4f.Provider.Vercel` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [you.com](https://you.com) | `g4f.Provider.You` | ✔️ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat9.yqcloud.top](https://chat9.yqcloud.top/) | `g4f.Provider.Yqcloud` | ✔️ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [chat.acytoo.com](https://chat.acytoo.com) | `g4f.Provider.Acytoo` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [aibn.cc](https://aibn.cc) | `g4f.Provider.Aibn` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [ai.ls](https://ai.ls) | `g4f.Provider.Ails` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgpt4online.org](https://chatgpt4online.org) | `g4f.Provider.Chatgpt4Online` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.chatgptdemo.net](https://chat.chatgptdemo.net) | `g4f.Provider.ChatgptDemo` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptduo.com](https://chatgptduo.com) | `g4f.Provider.ChatgptDuo` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptfree.ai](https://chatgptfree.ai) | `g4f.Provider.ChatgptFree` | ✔️ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chatgptlogin.ai](https://chatgptlogin.ai) | `g4f.Provider.ChatgptLogin` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [cromicle.top](https://cromicle.top) | `g4f.Provider.Cromicle` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [gptgod.site](https://gptgod.site) | `g4f.Provider.GptGod` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [opchatgpts.net](https://opchatgpts.net) | `g4f.Provider.Opchatgpts` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.ylokh.xyz](https://chat.ylokh.xyz) | `g4f.Provider.Ylokh` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |

### Other

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bard.google.com](https://bard.google.com) | `g4f.Provider.Bard` | ❌ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [deepinfra.com](https://deepinfra.com) | `g4f.Provider.DeepInfra` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [huggingface.co](https://huggingface.co/chat) | `g4f.Provider.HuggingChat` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [www.llama2.ai](https://www.llama2.ai) | `g4f.Provider.Llama2` | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [open-assistant.io](https://open-assistant.io/chat) | `g4f.Provider.OpenAssistant` | ❌ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |

### Models

| Model                                   | Base Provider | Provider            | Website                                     |
| --------------------------------------- | ------------- | ------------------- | ------------------------------------------- |
| palm                                    | Google        | g4f.Provider.Bard   | [bard.google.com](https://bard.google.com/) |
| h2ogpt-gm-oasst1-en-2048-falcon-7b-v3   | Hugging Face  | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| h2ogpt-gm-oasst1-en-2048-falcon-40b-v1  | Hugging Face  | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| h2ogpt-gm-oasst1-en-2048-open-llama-13b | Hugging Face  | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| claude-instant-v1                       | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| claude-v1                               | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| claude-v2                               | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| command-light-nightly                   | Cohere        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| command-nightly                         | Cohere        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| gpt-neox-20b                            | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| oasst-sft-1-pythia-12b                  | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| oasst-sft-4-pythia-12b-epoch-3.5        | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| santacoder                              | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| bloom                                   | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| flan-t5-xxl                             | Hugging Face  | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| code-davinci-002                        | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| gpt-3.5-turbo-16k                       | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| gpt-3.5-turbo-16k-0613                  | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| gpt-4-0613                              | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| text-ada-001                            | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| text-babbage-001                        | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| text-curie-001                          | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| text-davinci-002                        | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| text-davinci-003                        | OpenAI        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| llama13b-v2-chat                        | Replicate     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| llama7b-v2-chat                         | Replicate     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |

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
