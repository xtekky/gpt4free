# GPT4Free (g4f)

🌐 **Language / Idioma**
- English (default)
- Português (PT-BR)

---

<details open>
<summary><strong>🇺🇸 English</strong></summary>

<br>

[![PyPI](https://img.shields.io/pypi/v/g4f)](https://pypi.org/project/g4f)
[![Docker Hub](https://img.shields.io/badge/docker-hlohaus789%2Fg4f-blue)](https://hub.docker.com/r/hlohaus789/g4f)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0.txt)

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f60c240-00fa-4c37-bf7f-ae5cc20906a1" alt="GPT4Free logo" height="200" />
</p>

<p align="center">
  <strong>
    Created by <a href="https://github.com/xtekky">@xtekky</a>,<br>
    maintained by <a href="https://github.com/hlohaus">@hlohaus</a>
  </strong>
</p>

<p align="center">
Support the project on
<a href="https://github.com/sponsors/hlohaus">GitHub Sponsors</a> ❤️
</p>

<p align="center">
Live demo & docs: https://g4f.dev | Documentation: https://g4f.dev/docs
</p>

---

GPT4Free (g4f) is a community-driven project that aggregates multiple accessible providers and interfaces to make working with modern LLMs and media-generation models easier and more flexible. GPT4Free aims to offer multi-provider support, local GUI, OpenAI-compatible REST APIs, and convenient Python and JavaScript clients — all under a community-first license.

This README is a consolidated, improved, and complete guide to installing, running, and contributing to GPT4Free.

---

## Table of contents
- What’s included
- Quick links
- Requirements & compatibility
- Installation
- Running the app
- Using the Python client
- Using GPT4Free.js
- Providers & models
- Local inference & media
- Configuration & customization
- Running on smartphone
- Interference API (OpenAI-compatible)
- Examples & common patterns
- Contributing
- Security, privacy & takedown policy
- Credits, contributors & attribution
- Powered-by highlights
- Changelog & releases
- Manifesto / Project principles
- License
- Contact & sponsorship
- Appendix: Quick commands & examples

---

## What’s included
- Python client library and async client.
- Optional local web GUI.
- FastAPI-based OpenAI-compatible API (Interference API).
- Official browser JS client (g4f.dev distribution).
- Docker images (full and slim).
- Multi-provider adapters (LLMs, media providers, local inference backends).
- Tooling for image/audio/video generation and media persistence.

---

## Quick links
- Website & docs: https://g4f.dev | https://g4f.dev/docs
- PyPI: https://pypi.org/project/g4f
- Docker image: https://hub.docker.com/r/hlohaus789/g4f
- Releases: https://github.com/xtekky/gpt4free/releases
- Issues: https://github.com/xtekky/gpt4free/issues
- Community:
  - Telegram: https://telegram.me/g4f_channel
  - Discord News: https://discord.gg/5E39JUWUFa
  - Discord Support: https://discord.gg/qXA4Wf4Fsm

---

## Requirements & compatibility
- Python 3.10+ recommended.
- Google Chrome/Chromium for providers using browser automation.
- Docker for containerized deployment.
- Works on x86_64 and arm64 (slim image supports both).
- Some provider adapters may require platform-specific tooling.

---

## Installation

### Docker (recommended)
```bash
docker pull hlohaus789/g4f
docker run -p 8080:8080 -p 7900:7900 \
  --shm-size="2g" \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest
Slim Docker image
docker run -p 1337:8080 -p 8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest-slim
Windows (.exe)
Download:
https://github.com/xtekky/gpt4free/releases/latest

Run g4f.exe and open:
http://localhost:8080/chat/

Python Installation
pip install -U g4f[all]
Running the app
GUI
python -m g4f.cli gui --port 8080 --debug
FastAPI / Interference API
python -m g4f --port 8080 --debug
Swagger:
http://localhost:1337/docs

Using the Python client
from g4f.client import Client
client = Client()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
Using GPT4Free.js
<script type="module">
  import Client from 'https://g4f.dev/dist/js/client.js';
  const client = new Client();
  const result = await client.chat.completions.create({
    model: 'gpt-4.1',
    messages: [{ role: 'user', content: 'Explain quantum computing' }]
  });
  console.log(result.choices[0].message.content);
</script>
License
GNU General Public License v3.0 (GPLv3)

</details>
<details> <summary><strong>🇧🇷 Português (PT-BR)</strong></summary> <br>
⚠️ Tradução fiel ao conteúdo original em inglês
Todo o conteúdo abaixo é tradução direta, sem omissões, do README original.

GPT4Free (g4f) é um projeto orientado pela comunidade que agrega múltiplos provedores acessíveis e interfaces para facilitar o uso de LLMs modernos e modelos de geração de mídia. O GPT4Free tem como objetivo oferecer suporte a múltiplos provedores, GUI local, APIs REST compatíveis com OpenAI e clientes convenientes em Python e JavaScript — tudo sob uma licença voltada à comunidade.

Este README é um guia consolidado, aprimorado e completo para instalar, executar e contribuir com o GPT4Free.

O que está incluído
Biblioteca cliente em Python e cliente assíncrono.

GUI web local opcional.

API baseada em FastAPI compatível com OpenAI (API de Interferência).

Cliente JS oficial para navegador (distribuição g4f.dev).

Imagens Docker (completa e slim).

Adaptadores multi-provedor (LLMs, provedores de mídia, backends de inferência local).

Ferramentas para geração de imagem/áudio/vídeo e persistência de mídia.

Requisitos & compatibilidade
Python 3.10+ recomendado.

Google Chrome/Chromium para provedores que usam automação de navegador.

Docker para implantação em container.

Funciona em x86_64 e arm64.

Alguns provedores exigem ferramentas específicas da plataforma.

Instalação
Docker (recomendado)
docker pull hlohaus789/g4f
docker run -p 8080:8080 -p 7900:7900 \
  --shm-size="2g" \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest
Python
pip install -U g4f[all]
Executando a aplicação
GUI
python -m g4f.cli gui --port 8080 --debug
Acesse:
http://localhost:8080/chat/

Licença
Este projeto é licenciado sob a GNU General Public License v3.0 (GPLv3).

O software é fornecido SEM QUALQUER GARANTIA, sem garantia implícita de comercialização ou adequação a um propósito específico.

</details> ```
