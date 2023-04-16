# Free LLM APIs

This repository provides reverse-engineered language models from various sources. Some of these models are already available in the repo, while others are currently being worked on.

> **Important:** If you come across any website offering free language models, please create an issue or submit a pull request with the details. We will reverse engineer it and add it to this repository.

## Best Chatgpt site
> https://chat.chatbot.sex/chat
> This site was developed by me and includes **gpt-4**, **internet access** and **gpt-jailbreak's** like DAN

## To-Do List

- [x] implement poe.com create bot feature | AVAILABLE NOW
- [x] renaming the 'poe' module to 'quora'
- [x] add you.com api


## Table of Contents

- [Current Sites (No Authentication / Easy Account Creation)](#current-sites)
- [Sites with Authentication (Will Reverse Engineer but Need Account Access)](#sites-with-authentication)
- [Usage Examples](#usage-examples)
  - [`quora (poe)`](#example-poe)
  - [`phind`](#example-phind)
  - [`t3nsor`](#example-t3nsor)
  - [`ora`](#example-ora)
  - [`writesonic`](#example-writesonic)
  - [`you`](#example-you)

## Current Sites <a name="current-sites"></a>

| Website                    | Model(s)             |
| -------------------------- | -------------------- |
| [ora.sh](https://ora.sh)   | GPT-3.5 / 4              |
| [poe.com](https://poe.com) | GPT-4/3.5            |
| [writesonic.com](https://writesonic.com)|GPT-3.5 / Internet|
| [t3nsor.com](https://t3nsor.com)|GPT-3.5|
| [you.com](https://you.com)|GPT-3.5 / Internet / good search|
| [phind.com](https://phind.com)|GPT-4 / Internet / good search|

## Sites with Authentication <a name="sites-with-authentication"></a>

These sites will be reverse engineered but need account access:

* [chat.openai.com/chat](https://chat.openai.com/chat)
* [bard.google.com](https://bard.google.com)
* [bing.com/chat](https://bing.com/chat)

## Usage Examples <a name="usage-examples"></a>

### Example: `quora (poe)` (use like openai pypi package) - GPT-4 <a name="example-poe"></a>

```python
# quora model names: (use left key as argument)
models = {
    'sage'   : 'capybara',
    'gpt-4'  : 'beaver',
    'claude-v1.2'         : 'a2_2',
    'claude-instant-v1.0' : 'a2',
    'gpt-3.5-turbo'       : 'chinchilla'
}
```

#### !! new: bot creation

```python
# import quora (poe) package
import quora

# create account
# make shure to set enable_bot_creation to True
token = quora.Account.create(logging = True, enable_bot_creation=True)

model = quora.Model.create(
    token = token,
    model = 'gpt-3.5-turbo', # or claude-instant-v1.0
    system_prompt = 'you are ChatGPT a large language model ...' 
)

print(model.name) # gptx....

# streaming response
for response in quora.StreamingCompletion.create(
    custom_model = model.name,
    prompt       ='hello world',
    token        = token):
    
    print(response.completion.choices[0].text)
```

#### Normal Response:
```python

response = quora.Completion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token)

print(response.completion.choices[0].text)    
```     

### Example: `phind` (use like openai pypi package) <a name="example-phind"></a>

```python
# HELP WANTED: tls_client does not accept stream and timeout gets hit with long responses

import phind

prompt = 'hello world'

result = phind.Completion.create(
    model  = 'gpt-4',
    prompt = prompt,
    results     = phind.Search.create(prompt, actualSearch = False), # create search (set actualSearch to False to disable internet)
    creative    = False,
    detailed    = False,
    codeContext = '') # up to 3000 chars of code

print(result.completion.choices[0].text)
```

### Example: `t3nsor` (use like openai pypi package) <a name="example-t3nsor"></a>

```python
# Import t3nsor
import t3nsor

# t3nsor.Completion.create
# t3nsor.StreamCompletion.create

[...]

```

#### Example Chatbot
```python
messages = []

while True:
    user = input('you: ')

    t3nsor_cmpl = t3nsor.Completion.create(
        prompt   = user,
        messages = messages
    )

    print('gpt:', t3nsor_cmpl.completion.choices[0].text)
    
    messages.extend([
        {'role': 'user', 'content': user }, 
        {'role': 'assistant', 'content': t3nsor_cmpl.completion.choices[0].text}
    ])
```

#### Streaming Response:

```python
for response in t3nsor.StreamCompletion.create(
    prompt   = 'write python code to reverse a string',
    messages = []):

    print(response.completion.choices[0].text)
```

### Example: `ora` (use like openai pypi package) <a name="example-ora"></a>

### load model (new)

more gpt4 models in `/testing/ora_gpt4.py`

```python
# normal gpt-4: b8b12eaa-5d47-44d3-92a6-4d706f2bcacf
model = ora.CompletionModel.load(chatbot_id, 'gpt-4') # or gpt-3.5
```

#### create model / chatbot: 
```python
# inport ora
import ora

# create model
model = ora.CompletionModel.create(
    system_prompt = 'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible',
    description   = 'ChatGPT Openai Language Model',
    name          = 'gpt-3.5')

# init conversation (will give you a conversationId)
init = ora.Completion.create(
    model  = model,
    prompt = 'hello world')

print(init.completion.choices[0].text)

while True:
    # pass in conversationId to continue conversation
    
    prompt = input('>>> ')
    response = ora.Completion.create(
        model  = model,
        prompt = prompt,
        includeHistory = True, # remember history
        conversationId = init.id)
    
    print(response.completion.choices[0].text)
```

### Example: `writesonic` (use like openai pypi package) <a name="example-writesonic"></a>

```python
# import writesonic
import writesonic

# create account (3-4s)
account = writesonic.Account.create(logging = True)

# with loging: 
    # 2023-04-06 21:50:25 INFO __main__ -> register success : '{"id":"51aa0809-3053-44f7-922a...' (2s)
    # 2023-04-06 21:50:25 INFO __main__ -> id : '51aa0809-3053-44f7-922a-2b85d8d07edf'
    # 2023-04-06 21:50:25 INFO __main__ -> token : 'eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...'
    # 2023-04-06 21:50:28 INFO __main__ -> got key : '194158c4-d249-4be0-82c6-5049e869533c' (2s)

# simple completion
response = writesonic.Completion.create(
    api_key = account.key,
    prompt  = 'hello world'
)

print(response.completion.choices[0].text) # Hello! How may I assist you today?

# conversation

response = writesonic.Completion.create(
    api_key = account.key,
    prompt  = 'what is my name ?',
    enable_memory = True,
    history_data  = [
        {
            'is_sent': True,
            'message': 'my name is Tekky'
        },
        {
            'is_sent': False,
            'message': 'hello Tekky'
        }
    ]
)

print(response.completion.choices[0].text) # Your name is Tekky.

# enable internet

response = writesonic.Completion.create(
    api_key = account.key,
    prompt  = 'who won the quatar world cup ?',
    enable_google_results = True
)

print(response.completion.choices[0].text) # Argentina won the 2022 FIFA World Cup tournament held in Qatar ...
```

### Example: `you` (use like openai pypi package) <a name="example-you"></a>

```python
import you

# simple request with links and details
response = you.Completion.create(
    prompt       = "hello world",
    detailed     = True,
    includelinks = True,)

print(response)

# {
#     "response": "...",
#     "links": [...],
#     "extra": {...},
#         "slots": {...}
#     }
# }

#chatbot

chat = []

while True:
    prompt = input("You: ")
    
    response = you.Completion.create(
        prompt  = prompt,
        chat    = chat)
    
    print("Bot:", response["response"])
    
    chat.append({"question": prompt, "answer": response["response"]})
```

## Dependencies

The repository is written in Python and requires the following packages:

* websocket-client
* requests
* tls-client

You can install these packages using the provided `requirements.txt` file.

## Repository structure:
    .
    ├── ora/
    ├── quora/ (/poe)
    ├── t3nsor/
    ├── testing/
    ├── writesonic/
    ├── you/
    ├── README.md  <-- this file.
    └── requirements.txt


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xtekky/openai-gpt4&type=Timeline)](https://star-history.com/#xtekky/openai-gpt4&Timeline)


## Copyright: 
This program is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)     

Most code, with the exception of `quora/api.py` (by [ading2210](https://github.com/ading2210)), has been written by me, [xtekky](https://github.com/xtekky).

### Copyright Notice:
```
xtekky/openai-gpt4: multiple reverse engineered language-model api's to decentralise the ai industry.  
Copyright (C) 2023 xtekky

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

