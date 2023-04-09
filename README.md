# Free LLM APIs

This repository provides reverse-engineered language models from various sources. Some of these models are already available in the repo, while others are currently being worked on.

> **Important:** If you come across any website offering free language models, please create an issue or submit a pull request with the details. We will reverse engineer it and add it to this repository.

## To-Do List

- [ ] implement poe.com create bot feature (4)
- [ ] poe.com chat history management (3)
- [ ] renaming the 'poe' module to 'quora' (2)
- [x] add you.com api (1)


## Table of Contents

- [Current Sites (No Authentication / Easy Account Creation)](#current-sites)
- [Sites with Authentication (Will Reverse Engineer but Need Account Access)](#sites-with-authentication)
- [Usage Examples](#usage-examples)
  - [`poe`](#example-poe)
  - [`t3nsor`](#example-t3nsor)
  - [`ora`](#example-ora)
  - [`writesonic`](#example-writesonic)
  - [`you`](#example-you)

## Current Sites <a name="current-sites"></a>

| Website                    | Model(s)             |
| -------------------------- | -------------------- |
| [ora.sh](https://ora.sh)   | GPT-3.5              |
| [nat.dev](https://nat.dev) | GPT-4/3.5 (paid now, looking for bypass)|
| [poe.com](https://poe.com) | GPT-4/3.5            |
| [writesonic.com](https://writesonic.com)|GPT-3.5 / Internet|
| [t3nsor.com](https://t3nsor.com)|GPT-3.5|
| [you.com](https://you.com)|GPT-3.5 / Internet / good search|

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
    ├── poe/
    ├── t3nsor/
    ├── testing/
    ├── writesonic/
    ├── you/
    ├── README.md  <-- this file.
    └── requirements.txt