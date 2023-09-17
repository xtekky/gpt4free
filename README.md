![248433934-7886223b-c1d1-4260-82aa-da5741f303bb](https://github.com/xtekky/gpt4free/assets/98614666/ea012c87-76e0-496a-8ac4-e2de090cc6c9)

By using this repository or any code related to it, you agree to the [legal notice](./LEGAL_NOTICE.md). The author is not responsible for any copies, forks, reuploads made by other users, or anything else related to gpt4free. This is the author's only account and repository. To prevent impersonation or irresponsible actions, please comply with the GNU GPL license this Repository uses.

### New

- pypi package:

```
pip install -U g4f
```

## Table of Contents:

- [Table of Contents:](#table-of-contents)
- [Getting Started](#getting-started)
  - [Prerequisites:](#prerequisites)
  - [Setting up the project:](#setting-up-the-project)
    - [Install using pypi](#install-using-pypi)
    - [or](#or)
    - [Setting up with Docker:](#setting-up-with-docker)
- [Usage](#usage)
  - [The `g4f` Package](#the-g4f-package)
  - [interference openai-proxy api (use with openai python package)](#interference-openai-proxy-api-use-with-openai-python-package)
- [Models](#models)
  - [gpt-3.5 / gpt-4](#gpt-35--gpt-4)
  - [Other Models](#other-models)
- [Related gpt4free projects](#related-gpt4free-projects)
- [Contribute](#contribute)
- [ChatGPT clone](#chatgpt-clone)
- [Copyright:](#copyright)
- [Copyright Notice:](#copyright-notice)
- [Star History](#star-history)

## Getting Started

#### Prerequisites:

1. [Download and install Python](https://www.python.org/downloads/) (Version 3.x is recommended).

#### Setting up the project:

##### Install using pypi

```
pip install -U g4f
```

##### or

1. Clone the GitHub repository:

```
git clone https://github.com/xtekky/gpt4free.git
```

2. Navigate to the project directory:

```
cd gpt4free
```

3. (Recommended) Create a virtual environment to manage Python packages for your project:

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

##### Setting up with Docker:

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
docker compose build
```

5. Start the service using Docker Compose:

```bash
docker compose up
```

You server will now be running at `http://localhost:1337`. You can interact with the API or run your tests as you would normally.

To stop the Docker containers, simply run:

```bash
docker compose down
```

**Note:** When using Docker, any changes you make to your local files will be reflected in the Docker container thanks to the volume mapping in the `docker-compose.yml` file. If you add or remove dependencies, however, you'll need to rebuild the Docker image using `docker compose build`.

## Usage

### The `g4f` Package

```py
import g4f


print(g4f.Provider.Ails.params)  # supported args

# Automatic selection of provider

# streamed completion
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}],
    stream=True,
)

for message in response:
    print(message, flush=True, end='')

# normal response
response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    messages=[{"role": "user", "content": "hi"}],
)  # alterative model setting

print(response)


# Set with provider
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    provider=g4f.Provider.DeepAi,
    messages=[{"role": "user", "content": "Hello world"}],
    stream=True,
)

for message in response:
    print(message)
```


##### Providers:
```py
from g4f.Provider import (
    AItianhu,
    Acytoo,
    Aichat,
    Ails,
    Aivvm,
    Bard,
    Bing,
    ChatBase,
    ChatgptAi,
    ChatgptLogin,
    CodeLinkAva,
    DeepAi,
    H2o,
    HuggingChat,
    Opchatgpts,
    OpenAssistant,
    OpenaiChat,
    Raycast,
    Theb,
    Vercel,
    Vitalentum,
    Wewordle,
    Ylokh,
    You,
    Yqcloud,
)
# Usage:
response = g4f.ChatCompletion.create(..., provider=ProviderName)
```

##### Cookies Required:

Cookies are essential for the proper functioning of some service providers.
It is imperative to maintain an active session, typically achieved by logging into your account.

When running the g4f package locally, the package automatically retrieves cookies from your web browser using the `get_cookies` function. However, if you're not running it locally, you'll need to provide the cookies manually by passing them as parameters using the `cookies` parameter.

```py
import g4f
from g4f.Provider import (
    Bard,
    Bing,
    HuggingChat,
    OpenAssistant,
    OpenaiChat,
)
# Usage:
response = g4f.ChatCompletion.create(
    model=g4f.models.default,
    messages=[{"role": "user", "content": "Hello"}],
    provider=Bard,
    #cookies=g4f.get_cookies(".google.com"),
    cookies={"cookie_name": "value", "cookie_name2": "value2"},
    auth=True
)
```

##### Async Support:

To enhance speed and overall performance, execute providers asynchronously. The total execution time will be determined by the duration of the slowest provider's execution.

```py
import g4f, asyncio

async def run_async():
  _providers = [
      g4f.Provider.AItianhu,
      g4f.Provider.Acytoo,
      g4f.Provider.Aichat,
      g4f.Provider.Ails,
      g4f.Provider.Aivvm,
      g4f.Provider.ChatBase,
      g4f.Provider.ChatgptAi,
      g4f.Provider.ChatgptLogin,
      g4f.Provider.CodeLinkAva,
      g4f.Provider.DeepAi,
      g4f.Provider.Opchatgpts,
      g4f.Provider.Vercel,
      g4f.Provider.Vitalentum,
      g4f.Provider.Wewordle,
      g4f.Provider.Ylokh,
      g4f.Provider.You,
      g4f.Provider.Yqcloud,
  ]
  responses = [
      provider.create_async(
          model=g4f.models.default,
          messages=[{"role": "user", "content": "Hello"}],
      )
      for provider in _providers
  ]
  responses = await asyncio.gather(*responses)
  for idx, provider in enumerate(_providers):
      print(f"{provider.__name__}:", responses[idx])

asyncio.run(run_async())
```

### interference openai-proxy api (use with openai python package)

get requirements:

```sh
pip install -r interference/requirements.txt
```

run server:

```sh
python3 -m interference.app
```

```py
import openai

openai.api_key = ""
openai.api_base = "http://localhost:1337"


def main():
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "write a poem about a tree"}],
        stream=True,
    )

    if isinstance(chat_completion, dict):
        # not stream
        print(chat_completion.choices[0].message.content)
    else:
        # stream
        for token in chat_completion:
            content = token["choices"][0]["delta"].get("content")
            if content != None:
                print(content, end="", flush=True)


if __name__ == "__main__":
    main()
```

## Models

### gpt-3.5 / gpt-4

| Website| Provider| gpt-3.5 | gpt-4 | Streaming | Asynchron | Status | Auth |
| ------ | ------- | ------- | ----- | --------- | --------- | ------ | ---- |
| [www.aitianhu.com](https://www.aitianhu.com) | g4f.provider.AItianhu | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.acytoo.com](https://chat.acytoo.com) | g4f.provider.Acytoo | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat-gpt.org](https://chat-gpt.org/chat) | g4f.provider.Aichat | ✔️ | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [ai.ls](https://ai.ls) | g4f.provider.Ails | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.aivvm.com](https://chat.aivvm.com) | g4f.provider.Aivvm | ✔️ | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [bard.google.com](https://bard.google.com) | g4f.provider.Bard | ❌ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [bing.com](https://bing.com/chat) | g4f.provider.Bing | ❌ | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [www.chatbase.co](https://www.chatbase.co) | g4f.provider.ChatBase | ✔️ | ✔️ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chatgpt.ai](https://chatgpt.ai/) | g4f.provider.ChatgptAi | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [opchatgpts.net](https://opchatgpts.net) | g4f.provider.ChatgptLogin | ✔️ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [ava-ai-ef611.web.app](https://ava-ai-ef611.web.app) | g4f.provider.CodeLinkAva | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [deepai.org](https://deepai.org) | g4f.provider.DeepAi | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [gpt-gm.h2o.ai](https://gpt-gm.h2o.ai) | g4f.provider.H2o | ❌ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [huggingface.co](https://huggingface.co/chat/) | g4f.provider.HuggingChat | ❌ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [opchatgpts.net](https://opchatgpts.net) | g4f.provider.Opchatgpts | ✔️ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [open-assistant.io](https://open-assistant.io/chat) | g4f.provider.OpenAssistant | ❌ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [chat.openai.com](https://chat.openai.com) | g4f.provider.OpenaiChat | ✔️ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [raycast.com](https://raycast.com) | g4f.provider.Raycast | ✔️ | ✔️ | ✔️ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [theb.ai](https://theb.ai) | g4f.provider.Theb | ✔️ | ❌ | ✔️ | ❌ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [sdk.vercel.ai](https://sdk.vercel.ai) | g4f.provider.Vercel | ✔️ | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [app.vitalentum.io](https://app.vitalentum.io) | g4f.provider.Vitalentum | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [wewordle.org](https://wewordle.org) | g4f.provider.Wewordle | ✔️ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.ylokh.xyz](https://chat.ylokh.xyz) | g4f.provider.Ylokh | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [you.com](https://you.com) | g4f.provider.You | ✔️ | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat9.yqcloud.top](https://chat9.yqcloud.top/) | g4f.provider.Yqcloud | ✔️ | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [aiservice.vercel.app](https://aiservice.vercel.app/) | g4f.provider.AiService | ✔️ | ❌ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.dfehub.com](https://chat.dfehub.com/) | g4f.provider.DfeHub | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [free.easychat.work](https://free.easychat.work) | g4f.provider.EasyChat | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [next.eqing.tech](https://next.eqing.tech/) | g4f.provider.Equing | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat9.fastgpt.me](https://chat9.fastgpt.me/) | g4f.provider.FastGpt | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [forefront.com](https://forefront.com) | g4f.provider.Forefront | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.getgpt.world](https://chat.getgpt.world/) | g4f.provider.GetGpt | ✔️ | ❌ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [liaobots.com](https://liaobots.com) | g4f.provider.Liaobots | ✔️ | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [supertest.lockchat.app](http://supertest.lockchat.app) | g4f.provider.Lockchat | ✔️ | ✔️ | ✔️ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [p5.v50.ltd](https://p5.v50.ltd) | g4f.provider.V50 | ✔️ | ❌ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
| [chat.wuguokai.xyz](https://chat.wuguokai.xyz) | g4f.provider.Wuguokai | ✔️ | ❌ | ❌ | ❌ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |


### Other Models

| Model                                   | Base Provider | Provider            | Website                                     |
| --------------------------------------- | ------------- | ------------------- | ------------------------------------------- |
| palm                                    | Google        | g4f.Provider.Bard   | [bard.google.com](https://bard.google.com/) |
| h2ogpt-gm-oasst1-en-2048-falcon-7b-v3   | Huggingface   | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| h2ogpt-gm-oasst1-en-2048-falcon-40b-v1  | Huggingface   | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| h2ogpt-gm-oasst1-en-2048-open-llama-13b | Huggingface   | g4f.Provider.H2o    | [www.h2o.ai](https://www.h2o.ai/)           |
| claude-instant-v1                       | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| claude-v1                               | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| claude-v2                               | Anthropic     | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| command-light-nightly                   | Cohere        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| command-nightly                         | Cohere        | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| gpt-neox-20b                            | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| oasst-sft-1-pythia-12b                  | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| oasst-sft-4-pythia-12b-epoch-3.5        | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| santacoder                              | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| bloom                                   | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
| flan-t5-xxl                             | Huggingface   | g4f.Provider.Vercel | [sdk.vercel.ai](https://sdk.vercel.ai/)     |
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

## Related gpt4free projects

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
    <tr>
      <td><a href="https://github.com/xiangsx/gpt4free-ts"><b>gpt4free-ts</b></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/issues"><img alt="Issues" src="https://img.shields.io/github/issues/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
      <td><a href="https://github.com/xiangsx/gpt4free-ts/pulls"><img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/xiangsx/gpt4free-ts?style=flat-square&labelColor=343b41"/></a></td>
    </tr>
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

## Contribute

to add another provider, its very simple:

1. create a new file in [g4f/provider](./g4f/provider) with the name of the Provider
2. Implement a class that extends [BaseProvider](./g4f/provider/base_provider.py).

```py
from .base_provider import BaseProvider
from ..typing import CreateResult, Any


class HogeService(BaseProvider):
    url = "http://hoge.com"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        pass
```

3. Here, you can adjust the settings, for example if the website does support streaming, set `working` to `True`...
4. Write code to request the provider in `create_completion` and `yield` the response, _even if_ its a one-time response, do not hesitate to look at other providers for inspiration
5. Add the Provider Name in [g4f/provider/**init**.py](./g4f/provider/__init__.py)

```py
from .base_provider import BaseProvider
from .HogeService import HogeService

__all__ = [
  HogeService,
]
```

6. You are done !, test the provider by calling it:

```py
import g4f

response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.PROVIDERNAME,
                                    messages=[{"role": "user", "content": "test"}], stream=g4f.Provider.PROVIDERNAME.supports_stream)

for message in response:
    print(message, flush=True, end='')
```

## ChatGPT clone

> We are currently implementing new features and trying to scale it, but please be patient as it may be unstable  
> https://chat.g4f.ai/chat
> This site was developed by me and includes **gpt-4/3.5**, **internet access** and **gpt-jailbreak's** like DAN  
> Run locally here: https://github.com/xtekky/chatgpt-clone

## Copyright:

This program is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)

## Copyright Notice:

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

## Star History

<a href="https://github.com/xtekky/gpt4free/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=xtekky/gpt4free&type=Date">
</a>
