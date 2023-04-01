# Free LLM APIs

This repository provides reverse-engineered language models from various sources. Some of these models are already available in the repo, while others are currently being worked on.

> **Important:** If you come across any website offering free language models, please create an issue or submit a pull request with the details. We will reverse engineer it and add it to this repository.

## Table of Contents

- [Current Sites (No Authentication / Easy Account Creation)](#current-sites)
- [Sites with Authentication (Will Reverse Engineer but Need Account Access)](#sites-with-authentication)
- [Usage Examples](#usage-examples)
  - [`poe`](#example-poe)
  - [`t3nsor`](#example-t3nsor)
  - [`ora`](#example-ora)

## Current Sites <a name="current-sites"></a>

| Website                    | Model(s)             |
| -------------------------- | -------------------- |
| [ora.sh](https://ora.sh)   | GPT-3.5              |
| [nat.dev](https://nat.dev) | GPT-4/3.5 (paid now, looking for bypass)|
| [poe.com](https://poe.com) | GPT-4/3.5            |
| [writesonic.com](https://writesonic.com)|GPT-3.5 / Internet|
|[t3nsor.com](https://t3nsor.com)|GPT-3.5|

## Sites with Authentication <a name="sites-with-authentication"></a>

These sites will be reverse engineered but need account access:

* [chat.openai.com/chat](https://chat.openai.com/chat)
* [bard.google.com](https://bard.google.com)
* [bing.com/chat](https://bing.com/chat)

## Usage Examples <a name="usage-examples"></a>

### Example: `poe` (use like openai pypi package) - GPT-4 <a name="example-poe"></a>

```python
# Import poe
import poe

# poe.Account.create
# poe.Completion.create
# poe.StreamCompletion.create

[...]

```

#### Create Token (3-6s)
```python
token = poe.Account.create(logging = True)
print('token', token)
```

#### Streaming Response
```python

for response in poe.StreamingCompletion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token):
    
    print(response.completion.choices[0].text, end="", flush=True)
```

#### Normal Response:
```python

response = poe.Completion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token)

print(response.completion.choices[0].text)    
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

```python
# import ora
import ora

[...]

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
        conversationId = init.id)
    
    print(response.completion.choices[0].text)
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
    ├── poe/
    ├── t3nsor/
    ├── README.md  <-- this file.
    └── requirements.txt
