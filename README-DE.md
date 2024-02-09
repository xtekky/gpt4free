<a href="./README.md">
    <img src="https://img.shields.io/badge/open in-🇬🇧 english-blue.svg" alt="Open in EN">
</a>

![248433934-7886223b-c1d1-4260-82aa-da5741f303bb](https://github.com/xtekky/gpt4free/assets/98614666/ea012c87-76e0-496a-8ac4-e2de090cc6c9)

<a href='https://ko-fi.com/xtekky' target='_blank'><img height='35' style='border:0px;height:46px;' src='https://az743702.vo.msecnd.net/cdn/kofi3.png?v=0' border='0' alt='Kauf mir einen Kaffee auf ko-fi.com' />

<div id="top"></div>

> Durch die Nutzung dieses Repositories oder jeglichen damit verbundenen Code stimmen Sie dem [Rechtshinweis](LEGAL_NOTICE.md) zu. Der Autor ist nicht verantwortlich für Kopien, Forks, erneute Uploads durch andere Benutzer oder sonstige mit GPT4Free verbundene Aktivitäten. Dies ist das einzige Konto und Repository des Autors. Um Identitätsdiebstahl oder unverantwortliche Handlungen zu verhindern, halten Sie sich bitte an die GNU GPL-Lizenz, die dieses Repository verwendet.

```sh
pip install -U g4f
```


## 🆕 Was gibt es Neues

- Tritt unserem Telegram-Kanal bei: [t.me/g4f_channel](https://telegram.me/g4f_channel)
- Tritt unserer Discord-Gruppe bei: [discord.gg/XfybzPXPH5](https://discord.gg/XfybzPXPH5)
- Erkunde die g4f-Dokumentation (unvollständig): [g4f.mintlify.app](https://g4f.mintlify.app) | Trage zur Dokumentation bei: [github.com/xtekky/gpt4free-docs](https://github.com/xtekky/gpt4free-docs)


## 📚 Inhaltsverzeichnis

- [🆕 Was ist neu](#-was-ist-neu)
- [📚 Inhaltsverzeichnis](#-inhaltsverzeichnis)
- [🛠️ Erste Schritte](#️-erste-schritte)
    - [Voraussetzungen:](#voraussetzungen)
    - [Projekt einrichten:](#projekt-einrichten)
      - [Installation über PyPi](#installation-über-pypi)
      - [oder](#oder)
      - [Einrichten mit Docker:](#einrichten-mit-docker)
- [💡 Verwendung](#-verwendung)
  - [Das `g4f` Paket](#das-g4f-paket)
    - [ChatCompletion](#chatcompletion)
      - [Vervollständigung](#vervollständigung)
      - [Anbieter](#anbieter)
      - [Cookies erforderlich](#cookies-erforderlich)
      - [Async-Unterstützung](#async-unterstützung)
      - [Proxy- und Timeout-Unterstützung](#proxy-und-timeout-unterstützung)
  - [Interference openai-proxy API (Verwendung mit openai Python-Paket)](#interference-openai-proxy-api-verwendung-mit-openai-python-paket)
    - [API von PyPi-Paket ausführen](#api-von-pypi-paket-ausführen)
    - [API von Repository ausführen](#api-von-repository-ausführen)
- [🚀 Anbieter und Modelle](#-anbieter-und-modelle)
  - [GPT-4](#gpt-4)
  - [GPT-3.5](#gpt-35)
  - [Andere](#andere)
  - [Modelle](#modelle)
- [🔗 Verwandte GPT4Free-Projekte](#-verwandte-gpt4free-projekte)
- [🤝 Mitwirken](#-mitwirken)
    - [Anbieter mit KI-Tool erstellen](#anbieter-mit-ki-tool-erstellen)
    - [Anbieter erstellen](#anbieter-erstellen)
- [🙌 Mitwirkende](#-mitwirkende)
- [©️ Urheberrecht](#️-urheberrecht)
- [⭐ Sternenhistorie](#-sternenhistorie)
- [📄 Lizenz](#-lizenz)


## 🛠️ Erste Schritte

#### Voraussetzungen:

1. [Python herunterladen und installieren](https://www.python.org/downloads/) (Version 3.10+ wird empfohlen).

#### Projekt einrichten:

##### Installation über pypi

```
pip install -U g4f
```

##### oder

1. Klonen Sie das GitHub-Repository:

```
git clone https://github.com/xtekky/gpt4free.git
```

2. Navigieren Sie zum Projektverzeichnis:

```
cd gpt4free
```

3. (Empfohlen) Erstellen Sie eine Python-Virtual-Umgebung:
Sie können der [Python-Offiziellen Dokumentation](https://docs.python.org/3/tutorial/venv.html) für virtuelle Umgebungen folgen.

```
python3 -m venv venv
```

4. Aktivieren Sie die virtuelle Umgebung:
   - Unter Windows:
   ```
   .\venv\Scripts\activate
   ```
   - Unter macOS und Linux:
   ```
   source venv/bin/activate
   ```
5. Installieren Sie die erforderlichen Python-Pakete aus `requirements.txt`:

```
pip install -r requirements.txt
```

6. Erstellen Sie eine Datei `test.py` im Stammverzeichnis und beginnen Sie mit der Verwendung des Repositories. Weitere Anweisungen finden Sie unten

```py
import g4f

...
```

##### Einrichten mit Docker:

Wenn Docker installiert ist, können Sie das Projekt ohne manuelle Installation von Abhängigkeiten einfach einrichten und ausführen.

1. Stellen Sie zunächst sicher, dass sowohl Docker als auch Docker Compose installiert sind.

   - [Docker installieren](https://docs.docker.com/get-docker/)
   - [Docker Compose installieren](https://docs.docker.com/compose/install/)

2. Klonen Sie das GitHub-Repo:

```bash
git clone https://github.com/xtekky/gpt4free.git
```

3. Navigieren Sie zum Projektverzeichnis:

```bash
cd gpt4free
```

4. Erstellen Sie das Docker-Image:

```bash
docker-compose build
```

5. Starten Sie den Dienst mit Docker Compose:

```bash
docker-compose up
```

Ihr Server wird jetzt unter `http://localhost:1337` ausgeführt. Sie können mit der API interagieren oder Ihre Tests wie gewohnt ausführen.

Um die Docker-Container zu stoppen, führen Sie einfach aus:

```bash
docker-compose down
```

> [!Note]
> Wenn Sie Docker verwenden, werden alle Änderungen, die Sie an Ihren lokalen Dateien vornehmen, im Docker-Container durch die Volumenabbildung in der `docker-compose.yml`-Datei widergespiegelt. Wenn Sie jedoch Abhängigkeiten hinzufügen oder entfernen, müssen Sie das Docker-Image mit `docker-compose build` neu erstellen.


## 💡 Verwendung

### Das `g4f` Paket

#### ChatCompletion

```python
import g4f

g4f.debug.logging = True  # Aktiviere das Protokollieren
g4f.debug.check_version = False  # Deaktiviere die automatische Versionsüberprüfung
print(g4f.debug.get_version())  # Überprüfe die Version
print(g4f.Provider.Bing.params)  # Unterstützte Argumente

# Automatische Auswahl des Anbieters

# Gestreamte Vervollständigung
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hallo"}],
    stream=True,
)

for message in response:
    print(message, flush=True, end='')

# Normale Antwort
response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_4,
    messages=[{"role": "user", "content": "Hallo"}],
)  # Alternative Modellkonfiguration

print(response)
```

##### Completion

```python
import g4f

erlaubte_modelle = [
    'code-davinci-002',
    'text-ada-001',
    'text-babbage-001',
    'text-curie-001',
    'text-davinci-002',
    'text-davinci-003'
]

response = g4f.Completion.create(
    model='text-davinci-003',
    prompt='sage, dass dies ein Test ist'
)

print(response)
```

##### Anbieter

```python
import g4f

from g4f.Provider import (
    AItianhu,
    Aichat,
    Bard,
    Bing,
    ChatBase,
    ChatgptAi,
    OpenaiChat,
    Vercel,
    You,
    Yqcloud,
)

# Festlegen des Anbieters
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    provider=g4f.Provider.Aichat,
    messages=[{"role": "user", "content": "Hallo"}],
    stream=True,
)

for message in response:
    print(message)
```

##### Verwendung des Browsers

Einige Anbieter verwenden einen Browser, um den Bot-Schutz zu umgehen.
Sie verwenden den Selenium-Webtreiber, um den Browser zu steuern.
Die Browsereinstellungen und die Anmeldedaten werden in einem benutzerdefinierten Verzeichnis gespeichert.
Wenn der Headless-Modus aktiviert ist, werden die Browserfenster unsichtbar geladen.
Aus Leistungsgründen wird empfohlen, die Browserinstanzen wiederzuverwenden
und sie am Ende selbst zu schließen:

```python
import g4f
from undetected_chromedriver import Chrome, ChromeOptions
from g4f.Provider import (
    Bard,
    Poe,
    AItianhuSpace,
    MyShell,
    Phind,
    PerplexityAi,
)

options = ChromeOptions()
options.add_argument("--incognito")
browser = Chrome(options=options, headless=True)
for idx in range(10):
    response = g4f.ChatCompletion.create(
        model=g4f.models.default,
        provider=g4f.Provider.Phind,
        messages=[{"role": "user", "content": "Schlage mir einen Namen vor."}],
        browser=browser
    )
    print(f"{idx}:", response)
browser.quit()
```

##### Erforderliche Cookies

Cookies sind für die ordnungsgemäße Funktion einiger Dienstanbieter unerlässlich. Es ist unerlässlich, eine aktive Sitzung aufrechtzuerhalten, die in der Regel durch das Anmelden in Ihrem Konto erreicht wird.

Wenn Sie das g4f-Paket lokal ausführen, ruft das Paket automatisch Cookies aus Ihrem Webbrowser ab, indem es die `get_cookies`-Funktion verwendet. Wenn Sie es jedoch nicht lokal ausführen, müssen Sie die Cookies manuell bereitstellen, indem Sie sie als Parameter unter Verwendung des `cookies`-Parameters übergeben.

```python
import g4f

from g4f.Provider import (
    Bing,
    HuggingChat,
    OpenAssistant,
)

# Verwendung
response = g4f.ChatCompletion.create(
    model=g4f.models.default,
    messages=[{"role": "user", "content": "Hallo"}],
    provider=Bing,
    #cookies=g4f.get_cookies(".google.com"),
    cookies={"cookie_name": "value", "cookie_name2": "value2"},
    auth=True
)
```

##### Unterstützung für asynchrone Ausführung

Um die Geschwindigkeit und Gesamtleistung zu verbessern, führen Sie Anbieter asynchron aus. Die Gesamtausführungszeit wird durch die Dauer der langsamsten Anbieterausführung bestimmt.

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
            messages=[{"role": "user", "content": "Hallo"}],
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

##### Unterstützung für Proxy und Timeout

Alle Anbieter unterstützen das Angeben eines Proxy und das Erhöhen des Timeouts in den Erstellungsfunktionen.

```python
import g4f

response = g4f.ChatCompletion.create(
    model=g4f.models.default,
    messages=[{"role": "user", "content": "Hallo"}],
    proxy="http://host:port",
    # oder socks5://user:pass@host:port
    timeout=120,  # in Sekunden
)

print(f"Ergebnis:", response)
```

### Interference openai-proxy API (Verwendung mit dem openai Python-Paket)

#### Führen Sie die Interference API aus dem PyPi-Paket aus

```python
from g4f.api import run_api

run_api()
```

#### Führen Sie die Interference API aus dem Repository aus

Wenn Sie die Einbettungsfunktion verwenden möchten, benötigen Sie einen Hugging Face-Token. Sie können einen unter [Hugging Face Tokens](https://huggingface.co/settings/tokens) erhalten. Stellen Sie sicher, dass Ihre Rolle auf Schreiben eingestellt ist. Wenn Sie Ihren Token haben, verwenden Sie ihn einfach anstelle des OpenAI-API-Schlüssels.

Server ausführen:

```sh
g4f api
```

oder

```sh
python -m g4f.api
```

```python
import openai

# Setzen Sie Ihren Hugging Face-Token als API-Schlüssel, wenn Sie Einbettungen verwenden
# Wenn Sie keine Einbettungen verwenden, lassen Sie es leer
openai.api_key = "IHR_HUGGING_FACE_TOKEN"  # Ersetzen Sie dies durch Ihren tatsächlichen Token

# Setzen Sie die API-Basis-URL, falls erforderlich, z.B. für eine lokale Entwicklungsumgebung
openai.api_base = "http://localhost:1337/v1"

def main():
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "schreibe ein Gedicht über einen Baum"}],
        stream=True,
    )

    if isinstance(chat_completion, dict):
        # Nicht gestreamt
        print(chat_completion.choices[0].message.content)
    else:
        # Gestreamt
        for token in chat_completion:
            content = token["choices"][0]["delta"].get("content")
            if content is not None:
                print(content, end="", flush=True)

if __name__ == "__main__":
    main()
```

## 🚀 Anbieter und Modelle

### GPT-4

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bing.com](https://bing.com/chat) | `g4f.Provider.Bing` | ❌ | ✔️ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [chat.geekgpt.org](https://chat.geekgpt.org) | `g4f.Provider.GeekGpt` | ✔️ | ✔️ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
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
| [chat-shared2.zhile.io](https://chat-shared2.zhile.io) | `g4f.Provider.FakeGpt` | ✔️ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ❌ |
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

### Andere

| Website | Provider | GPT-3.5 | GPT-4 | Stream | Status | Auth |
| ------  | -------  | ------- | ----- | ------ | ------ | ---- |
| [bard.google.com](https://bard.google.com) | `g4f.Provider.Bard` | ❌ | ❌ | ❌ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ✔️ |
| [deepinfra.com](https://deepinfra.com) | `g4f.Provider.DeepInfra` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ❌ |
| [huggingface.co](https://huggingface.co/chat) | `g4f.Provider.HuggingChat` | ❌ | ❌ | ✔️ | ![Active](https://img.shields.io/badge/Active-brightgreen) | ✔️ |
| [www.llama2.ai](https://www.llama2.ai) | `g4f.Provider.Llama2` | ❌ | ❌ | ✔️ | ![Unknown](https://img.shields.io/badge/Unknown-grey) | ❌ |
| [open-assistant.io](https://open-assistant.io/chat) | `g4f.Provider.OpenAssistant` | ❌ | ❌ | ✔️ | ![Inactive](https://img.shields.io/badge/Inactive-red) | ✔️ |

### Modelle

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


## 🔗 Verwandte GPT4Free-Projekte

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



## 🤝 Mitwirken

#### Erstellen Sie einen Anbieter mit AI-Tool

Rufen Sie im Terminal das Skript `create_provider.py` auf:
```bash
python etc/tool/create_provider.py
```
1. Geben Sie Ihren Namen für den neuen Anbieter ein.
2. Kopieren Sie den `cURL`-Befehl aus den Entwicklertools Ihres Browsers und fügen Sie ihn ein.
3. Lassen Sie die KI den Anbieter für Sie erstellen.
4. Passen Sie den Anbieter nach Ihren Bedürfnissen an.

#### Anbieter erstellen

1. Überprüfen Sie die aktuelle [Liste potenzieller Anbieter](https://github.com/zukixa/cool-ai-stuff#ai-chat-websites) oder finden Sie Ihre eigene Anbieterquelle!
2. Erstellen Sie eine neue Datei in [g4f/Provider](./g4f/Provider) mit dem Namen des Anbieters.
3. Implementieren Sie eine Klasse, die von [BaseProvider](./g4f/Provider/base_provider.py) erbt.

```py
from __future__ import annotations

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class HogeService(AsyncGeneratorProvider):
    url                   = "https://chat-gpt.com"
    supports_gpt_35_turbo = True
    working               = True

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

4. Hier können Sie die Einstellungen anpassen, zum Beispiel, wenn die Website Streaming unterstützt, setzen Sie `supports_stream` auf `True`...
5. Schreiben Sie Code, um den Anbieter in `create_async_generator` anzufordern und die Antwort mit `yield` zurückzugeben, selbst wenn es sich um eine einmalige Antwort handelt. Zögern Sie nicht, sich bei anderen Anbietern inspirieren zu lassen.
6. Fügen Sie den Namen des Anbieters in [`g4f/Provider/__init__.py`](./g4f/Provider/__init__.py) hinzu.

```py
from .HogeService import HogeService

__all__ = [
  HogeService,
]
```

7. Sie sind fertig! Testen Sie den Anbieter, indem Sie ihn aufrufen:

```py
import g4f

response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.PROVIDERNAME,
                                    messages=[{"role": "user", "content": "test"}], stream=g4f.Provider.PROVIDERNAME.supports_stream)

for message in response:
    print(message, flush=True, end='')
```


## 🙌 Mitwirkende

Eine Liste der Mitwirkenden ist [hier](https://github.com/xtekky/gpt4free/graphs/contributors) verfügbar.
Die Datei [`Vercel.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/Vercel.py) enthält Code von [vercel-llm-api](https://github.com/ading2210/vercel-llm-api) von [@ading2210](https://github.com/ading2210), der unter der [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.txt) lizenziert ist.
Top 1 Mitwirkender: [@hlohaus](https://github.com/hlohaus)

## ©️ Urheberrecht

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

## ⭐ Sternenverlauf

<a href="https://github.com/xtekky/gpt4free/stargazers">
        <img width="500" alt="Star History Chart" src="https://api.star-history.com/svg?repos=xtekky/gpt4free&type=Date">
</a>


## 📄 Lizenz

<table>
  <tr>
    <td>
      <p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/GPLv3_Logo.svg/1200px-GPLv3_Logo.svg.png" width="80%"></img>
    </td>
    <td>
      <img src="https://img.shields.io/badge/Lizenz-GNU_GPL_v3.0-rot.svg"/> <br>
      Dieses Projekt steht unter der <a href="./LICENSE">GNU_GPL_v3.0-Lizenz</a>.
    </td>
  </tr>
</table>

<p align="right">(<a href="#top">🔼 Zurück nach oben</a>)</p>
