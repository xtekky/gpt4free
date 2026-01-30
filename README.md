# GPT4Free (g4f)

🌐 **Languages / Idiomas**  
- [English](#english)  
- [Português (PT-BR)](#português-pt-br)

---

## English

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

GPT4Free (g4f) is a community-driven project that aggregates multiple accessible providers and interfaces to make working with modern LLMs and media-generation models easier and more flexible.

It offers:
- Multi-provider support  
- Local web GUI  
- OpenAI-compatible REST APIs  
- Python & JavaScript clients  

All under a community-first license.

This README is a complete guide to installing, running, and contributing to GPT4Free.

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
- Configuration
- Interference API
- Contributing
- License

---

## What’s included
- Python client & async client
- Optional local web GUI
- FastAPI OpenAI-compatible API (Interference API)
- Official browser JS client
- Docker images (full & slim)
- Multi-provider adapters
- Media generation tooling

---

## Quick links
- Website & docs: https://g4f.dev  
- PyPI: https://pypi.org/project/g4f  
- Docker: https://hub.docker.com/r/hlohaus789/g4f  
- Releases: https://github.com/xtekky/gpt4free/releases  
- Issues: https://github.com/xtekky/gpt4free/issues  

---

## Requirements & compatibility
- Python 3.10+
- Chrome / Chromium
- Docker (recommended)
- x86_64 and arm64 supported

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
