# gpt4free package

### What is it?

gpt4free is a python package that provides some language model api's

### Main Features

- It's free to use
- Easy access

### Installation:

```bash
pip install gpt4free
```

#### Usage:

```python
import gpt4free
from gpt4free import Provider, quora, forefront

# usage You
response = gpt4free.Completion.create(Provider.You, prompt='Write a poem on Lionel Messi')
print(response)

# usage Poe
token = quora.Account.create(logging=False)
response = gpt4free.Completion.create(Provider.Poe, prompt='Write a poem on Lionel Messi', token=token, model='ChatGPT')
print(response)

# usage forefront
token = forefront.Account.create(logging=False)
response = gpt4free.Completion.create(
    Provider.ForeFront, prompt='Write a poem on Lionel Messi', model='gpt-4', token=token
)
print(response)
print(f'END')

# usage theb
response = gpt4free.Completion.create(Provider.Theb, prompt='Write a poem on Lionel Messi')
print(response)


```

### Invocation Arguments

`gpt4free.Completion.create()` method has two required arguments

1. Provider: This is an enum representing different provider
2. prompt: This is the user input

#### Keyword Arguments

Some of the keyword arguments are optional, while others are required.

- You:
    - `safe_search`: boolean - default value is `False`
    - `include_links`: boolean - default value is `False`
    - `detailed`: boolean - default value is `False`
- Quora:
    - `token`: str - this needs to be provided by the user
    - `model`: str - default value is `gpt-4`.
      
  (Available models: `['Sage', 'GPT-4', 'Claude+', 'Claude-instant', 'ChatGPT', 'Dragonfly', 'NeevaAI']`)
- ForeFront:
  - `token`: str - this need to be provided by the user

- Theb:
  (no keyword arguments required)

#### Token generation of quora
```python
from gpt4free import quora

token = quora.Account.create(logging=False)
```

### Token generation of ForeFront
```python
from gpt4free import forefront

token = forefront.Account.create(logging=False)
```

## Copyright:

This program is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)

### Copyright Notice: <a name="copyright"></a>

```
xtekky/gpt4free: multiple reverse engineered language-model api's to decentralise the ai industry.  
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
