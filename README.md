# Free LLM APIs

This repository provides reverse-engineered language models from various sources. Some of these models are already available in the repo, while others are currently being worked on.

> **Important:** If you come across any website offering free language models, please create an issue or submit a pull request with the details. We will reverse engineer it and add it to this repository.

## Chatgpt clone
> https://chat.chatbot.sex/chat
> This site was developed by me and includes **gpt-4/3.5**, **internet access** and **gpt-jailbreak's** like DAN
> You can find an opensource version of it to run locally here: https://github.com/xtekky/chatgpt-clone


## Table of Contents

- [Current Sites (No Authentication / Easy Account Creation)](#current-sites)
- [Sites with Authentication (Will Reverse Engineer but Need Account Access)](#sites-with-authentication)
- [Usage Examples]
  - [`quora (poe)`](./quora/README.md)
  - [`phind`](./phind/README.md)
  - [`t3nsor`](./t3nsor/README.md)
  - [`ora`](./ora/README.md)
  - [`writesonic`](./writesonic/README.md)
  - [`you`](./you/README.md)
  - [`sqlchat`](./sqlchat/README.md)

## Current Sites <a name="current-sites"></a>

| Website                    | Model(s)             |
| -------------------------- | -------------------- |
| [ora.sh](https://ora.sh)   | GPT-3.5 / 4              |
| [poe.com](https://poe.com) | GPT-4/3.5            |
| [writesonic.com](https://writesonic.com)|GPT-3.5 / Internet|
| [t3nsor.com](https://t3nsor.com)|GPT-3.5|
| [you.com](https://you.com)|GPT-3.5 / Internet / good search|
| [phind.com](https://phind.com)|GPT-4 / Internet / good search|
| [sqlchat.ai](https://sqlchat.ai)|GPT-3.5|

## Sites with Authentication <a name="sites-with-authentication"></a>

These sites will be reverse engineered but need account access:

| Website                                             | Model(s)       |
| --------------------------------------------------- | -------------- |
| [chat.openai.com/chat](https://chat.openai.com/chat)| GPT-3.5        |
| [bard.google.com](https://bard.google.com)          | custom / search|
| [bing.com/chat](https://bing.com/chat)              | gpt-4/3.5      |

## Best sites
#### gpt-4
- [`/ora`](./ora/README.md) 
- here is proof / test: [`ora_gpt4_proof.py`](./testing/ora_gpt4_proof.py)
- why ?, no streaming compared to poe.com but u can send more than 1 message

#### gpt-3.5
- [`/sqlchat`](./sqlchat/README.md)
- why ? (streaming + you can give conversation history)

#### search
- [`/phind`](./phind/README.md)
- why ? its not sure if they use gpt, but rather claude but they have an amazing search and good reasoning model


## Dependencies

* websocket-client
* requests
* tls-client
* pypasser
* names
* colorama
* curl_cffi

install with:
```sh
pip3 install -r requirements.txt
```

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

