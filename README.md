# GPT4free - use ChatGPT, for free!!

<img width="1383" alt="image" src="https://user-images.githubusercontent.com/98614666/233799515-1a7cb6a3-b17f-42c4-956d-8d2a0664466f.png">

Have you ever come across some amazing projects that you couldn't use **just because you didn't have an OpenAI API key?** 

**We've got you covered!** This repository offers **reverse-engineered** third-party APIs for `GPT-4/3.5`, sourced from various websites. You can simply **download** this repository, and use the available modules, which are designed to be used **just like OpenAI's official package**. **Unleash ChatGPT's potential for your projects, now!** You are welcome ; ).

By the way, thank you so much for `5k` stars and all the support!!


## Table of Contents

- [To do list](#todo)
- [Current Sites](#current-sites)
- [Best Sites for gpt4](#best-sites)
- [How to install](#install)
- [Legal Notice](#legal-notice)
- [Copyright](#copyright)


- [Usage Examples](./README.md)
  - [`quora (poe)`](./quora/README.md)
  - [`phind`](./phind/README.md)
  - [`t3nsor`](./t3nsor/README.md)
  - [`writesonic`](./writesonic/README.md)
  - [`you`](./you/README.md)
  - [`sqlchat`](./sqlchat/README.md)
  
- [replit Example (feel free to fork this repl)](#replit)


## Todo <a name="todo"></a>

- [ ] Add a GUI for the repo 
- [ ] Make a general package named `openai_rev`, instead of different folders
- [ ] Live api status to know which are down and which can be used
- [ ] Integrate more API's in `./unfinished` as well as other ones in the lists
- [ ] Make an API to use as proxy for other projects
- [ ] Make a pypi package

## Current Sites <a name="current-sites"></a>

| Website                                              | Model(s)                        |
| ---------------------------------------------------- | ------------------------------- |
| [poe.com](https://poe.com)                           | GPT-4/3.5                       |
| [writesonic.com](https://writesonic.com)             | GPT-3.5 / Internet              |
| [t3nsor.com](https://t3nsor.com)                     | GPT-3.5                         |
| [you.com](https://you.com)                           | GPT-3.5 / Internet / good search|
| [phind.com](https://phind.com)                       | GPT-4 / Internet / good search  |
| [sqlchat.ai](https://sqlchat.ai)                     | GPT-3.5                         |
| [chat.openai.com/chat](https://chat.openai.com/chat) | GPT-3.5                         |
| [bard.google.com](https://bard.google.com)           | custom / search                 |
| [bing.com/chat](https://bing.com/chat)               | GPT-4/3.5                       |

## Best sites  <a name="best-sites"></a>

#### gpt-4
- [`/phind`](./phind/README.md) 
- pro: only stable gpt-4 with streaming ( no limit )
- contra: weird backend prompting 
- why not `ora` anymore ? gpt-4 requires login + limited

#### gpt-3.5
- looking for a stable api at the moment

## Install  <a name="install"></a>
download or clone this GitHub repo  
install requirements with:
```sh
pip3 install -r requirements.txt
```

## To start gpt4free GUI
move `streamlit_app.py` from `./gui` to the base folder   
then run:   
`streamlit run streamlit_app.py` or `python3 -m streamlit run streamlit_app.py`

## Docker
Build
```
docker build -t gpt4free:latest -f Docker/Dockerfile .
```
Run
```
docker run -p 8501:8501 gpt4free:latest
```

## ChatGPT clone
> currently implementing new features and trying to scale it, please be patient it may be unstable     
> https://chat.chatbot.sex/chat
> This site was developed by me and includes **gpt-4/3.5**, **internet access** and **gpt-jailbreak's** like DAN   
> run locally here: https://github.com/xtekky/chatgpt-clone

## Legal Notice <a name="legal-notice"></a>

This repository uses third-party APIs and AI models and is *not* associated with or endorsed by the API providers or the original developers of the models. This project is intended **for educational purposes only**.

Please note the following:

1. **Disclaimer**: The APIs, services, and trademarks mentioned in this repository belong to their respective owners. This project is *not* claiming any right over them.

2. **Responsibility**: The author of this repository is *not* responsible for any consequences arising from the use or misuse of this repository or the content provided by the third-party APIs and any damage or losses caused by users' actions.

3. **Educational Purposes Only**: This repository and its content are provided strictly for educational purposes. By using the information and code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws and regulations.

## Copyright: 
This program is licensed under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt)     

Most code, with the exception of `quora/api.py` (by [ading2210](https://github.com/ading2210)), has been written by me, [xtekky](https://github.com/xtekky).

### Copyright Notice: <a name="copyright"></a>
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

## replit
You can fork this repl to host your own ChatGPT-clone WebUI. https://replit.com/@gpt4free/gpt4free-webui
