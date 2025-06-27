<a href="https://trendshift.io/repositories/1692" target="_blank"><img src="https://trendshift.io/api/badge/repositories/1692" alt="xtekky%2Fgpt4free | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

---

<p align="center">
  <span style="background: linear-gradient(45deg, #12c2e9, #c471ed, #f64f59); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    <strong>Written by <a href="https://github.com/xtekky">@xtekky</a></strong>
  </span>
</p>

<div id="top"></div>

> [!IMPORTANT]
> By using this repository or any code related to it, you agree to the [legal notice](LEGAL_NOTICE.md). The author is **not responsible for the usage of this repository nor endorses it**, nor is the author responsible for any copies, forks, re-uploads made by other users, or anything else related to GPT4Free. This is the author's only account and repository. To prevent impersonation or irresponsible actions, please comply with the GNU GPL license this Repository uses.

> [!WARNING]
> _"gpt4free"_ serves as a **PoC** (proof of concept), demonstrating the development of an API package with multi-provider requests, with features like timeouts, load balance and flow control.

> [!NOTE]
> <sup><strong>Latest version:</strong></sup><br> [![PyPI version](https://img.shields.io/pypi/v/g4f?color=blue)](https://pypi.org/project/g4f) [![Docker version](https://img.shields.io/docker/v/hlohaus789/g4f?label=docker&color=blue)](https://hub.docker.com/r/hlohaus789/g4f)  
> <sup><strong>Stats:</strong></sup><br> [![Downloads](https://static.pepy.tech/badge/g4f)](https://pepy.tech/project/g4f) [![Downloads](https://static.pepy.tech/badge/g4f/month)](https://pepy.tech/project/g4f)

```sh
pip install -U g4f[all]
```

```sh
docker pull hlohaus789/g4f
```

## 🆕 What's New

- **Explore the latest features and updates**  
  Find comprehensive details on our [Releases Page](https://github.com/xtekky/gpt4free/releases).  

- **Stay updated with our Telegram Channel** 📨  
  Join us at [telegram.me/g4f_channel](https://telegram.me/g4f_channel).
  
- **Subscribe to our Discord News Channel** 💬🆕️  
  Stay informed about updates via our [News Channel: discord.gg/5E39JUWUFa](https://discord.gg/5E39JUWUFa).
  
- **Get support in our Discord Community** 🤝💻  
  Reach out for help in our [Support Group: discord.gg/qXA4Wf4Fsm](https://discord.gg/qXA4Wf4Fsm).

- **Read our Documentation** 📖  
  Find detailed guidance and resources at [g4f.dev/docs](https://github.com/gpt4free/g4f.dev).

## 🔻 Site Takedown

Is your site on this repository and you want to take it down? Send an email to takedown@g4f.ai with proof it is yours and it will be removed as fast as possible. To prevent reproduction please secure your API. 😉

## 🚀 **Experience Live G4F**

Want to see G4F in action? Explore a live demo now!

[**Click here to explore the live G4F demo!**](https://github.com/user-attachments/assets/83519200-2f27-48c6-9fc3-bff0fcd96f41)

Curious to see what G4F can do? Dive into a live demonstration and visit the [official g4f.dev homepage](https://g4f.dev/) for more.

---

## 📚 Table of Contents
   - [🆕 What's New](#-whats-new)
   - [📚 Table of Contents](#-table-of-contents)
   - [⚡ Getting Started](#-getting-started)
      - [🛠 Installation](#-installation)
         - [🐳 Using Docker](#-using-docker)
         - [🪟 Windows Guide (.exe)](#-windows-guide-exe)
         - [🐍 Python Installation](#-python-installation)
  - [💡 Usage](#-usage)
     - [📝 Text Generation](#-text-generation)
     - [🎨 Image Generation](#-image-generation)
     - [🌐 Web Interface](#-web-interface)
     - [🖥️ Local Inference](https://github.com/gpt4free/g4f.dev/blob/main/docs/local.md)
     - [🤖 Inference API](#-inference-api)
     - [🛠️ Configuration](https://github.com/gpt4free/g4f.dev/blob/main/docs/configuration.md)
     - [📱 Run on Smartphone](#-run-on-smartphone)
     - [📘 Full Documentation for Python API](#-full-documentation-for-python-api)
  - [🚀 Providers and Models](https://github.com/gpt4free/g4f.dev/blob/main/docs%2Fproviders-and-models.md)
  - [🔗 Powered by gpt4free](#-powered-by-gpt4free)
  - [🤝 Contribute](#-contribute)
     - [How do i create a new Provider?](#guide-how-do-i-create-a-new-provider)
     - [How can AI help me with writing code?](#guide-how-can-ai-help-me-with-writing-code)
   - [🙌 Contributors](#-contributors)
   - [©️ Copyright](#-copyright)
  - [⭐ Star History](#-star-history)
  - [📄 License](#-license)

---

## ⚡️ Getting Started

## 🛠 Installation

### 🐳 Using Docker
1. **Install Docker:** [Download and install Docker](https://docs.docker.com/get-docker/).
2. **Set Up Directories:** Before running the container, make sure the necessary data directories exist or can be created. For example, you can create and set ownership on these directories by running: 
```bash
mkdir -p ${PWD}/har_and_cookies ${PWD}/generated_media
sudo chown -R 1200:1201 ${PWD}/har_and_cookies ${PWD}/generated_media
```
3. **Run the Docker Container:** Use the following commands to pull the latest image and start the container (Only x64):
```bash
docker pull hlohaus789/g4f
docker run -p 8080:8080 -p 7900:7900 \
  --shm-size="2g" \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest
```

4. **Running the Slim Docker Image:** And use the following commands to run the Slim Docker image. This command also updates the `g4f` package at startup and installs any additional dependencies: (x64 and arm64)
```bash
mkdir -p ${PWD}/har_and_cookies ${PWD}/generated_media
chown -R 1000:1000 ${PWD}/har_and_cookies ${PWD}/generated_media
docker run \
  -p 1337:8080 -p 8080:8080 \
  -v ${PWD}/har_and_cookies:/app/har_and_cookies \
  -v ${PWD}/generated_media:/app/generated_media \
  hlohaus789/g4f:latest-slim
```
 
5. **Access the Client Interface:**
   - **To use the included client, navigate to:** [http://localhost:8080/chat/](http://localhost:8080/chat/)
   - **Or set the API base for your client to:** [http://localhost:8080/v1](http://localhost:8080/v1)

6. **(Optional) Provider Login:**
   If required, you can access the container's desktop here: http://localhost:7900/?autoconnect=1&resize=scale&password=secret for provider login purposes.

---

### 🪟 Windows Guide (.exe)
To ensure the seamless operation of our application, please follow the instructions below. These steps are designed to guide you through the installation process on Windows operating systems.

**Installation Steps:**
1. **Download the Application**: Visit our [releases page](https://github.com/xtekky/gpt4free/releases/tag/0.4.2.0) and download the most recent version of the application, named `g4f.exe.zip`.
2. **File Placement**: After downloading, locate the `.zip` file in your Downloads folder. Unpack it to a directory of your choice on your system, then execute the `g4f.exe` file to run the app.
3. **Open GUI**: The app starts a web server with the GUI. Open your favorite browser and navigate to [http://localhost:8080/chat/](http://localhost:8080/chat/) to access the application interface.
4. **Firewall Configuration (Hotfix)**: Upon installation, it may be necessary to adjust your Windows Firewall settings to allow the application to operate correctly. To do this, access your Windows Firewall settings and allow the application.

By following these steps, you should be able to successfully install and run the application on your Windows system. If you encounter any issues during the installation process, please refer to our Issue Tracker or try to get contact over Discord for assistance.

---

### 🐍 Python Installation

#### Prerequisites:
1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/).
2. Install Google Chrome for certain providers.

#### Install with PyPI:
```bash
pip install -U g4f[all]
```

> How do I install only parts or do disable parts? **Use partial requirements:** [/docs/requirements](https://github.com/gpt4free/g4f.dev/blob/main/docs/requirements.md)

#### Install from Source:
```bash
git clone https://github.com/xtekky/gpt4free.git
cd gpt4free
pip install -r requirements.txt
```

> How do I load the project using git and installing the project requirements? **Read this tutorial and follow it step by step:** [/docs/git](https://github.com/gpt4free/g4f.dev/blob/main/docs/git.md)

---

## 💡 Usage

### 📝 Text Generation
```python
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    web_search=False
)
print(response.choices[0].message.content)
```
```
Hello! How can I assist you today?
```

### 🎨  Image Generation
```python
from g4f.client import Client

client = Client()
response = client.images.generate(
    model="flux",
    prompt="a white siamese cat",
    response_format="url"
)

print(f"Generated image URL: {response.data[0].url}")
```
[![Image with cat](https://g4f.dev/docs/images/cat.jpeg)](https://github.com/gpt4free/g4f.dev/blob/main/docs/client.md)

### 🌐 Web Interface
**Run the GUI using Python:**
```python
from g4f.gui import run_gui

run_gui()
```
**Run via CLI (To start the Flask Server):**
```bash
python -m g4f.cli gui --port 8080 --debug
```
**Or, start the FastAPI Server:**
```bash
python -m g4f --port 8080 --debug
```

> **Learn More About the GUI:** For detailed instructions on how to set up, configure, and use the GPT4Free GUI, refer to the [GUI Documentation](https://github.com/gpt4free/g4f.dev/blob/main/docs/gui.md) . This guide includes step-by-step details on provider selection, managing conversations, using advanced features like speech recognition, and more.

---

### 🤖 Inference API

The **Inference API** enables seamless integration with OpenAI's services through G4F, allowing you to deploy efficient AI solutions.

- **Documentation**: [Inference API Docs](https://github.com/gpt4free/g4f.dev/blob/main/docs/inference-api.md)
- **Endpoint**: `http://localhost:1337/v1`
- **Swagger UI**: Explore the OpenAPI documentation via Swagger UI at `http://localhost:1337/docs`
- **Provider Selection**: [How to Specify a Provider?](https://github.com/gpt4free/g4f.dev/blob/main/docs/selecting_a_provider.md)

This API is designed for straightforward implementation and enhanced compatibility with other OpenAI integrations.

---

### 📱 Run on Smartphone
Run the Web UI on your smartphone for easy access on the go. Check out the dedicated guide to learn how to set up and use the GUI on your mobile device: [Run on Smartphone Guide](https://github.com/gpt4free/g4f.dev/blob/main/docs/guides/phone.md)

---

#### **📘 Full Documentation for Python API**
   - **Client API from G4F:** [/docs/client](https://github.com/gpt4free/g4f.dev/blob/main/docs/client.md)
   - **AsyncClient API from G4F:** [/docs/async_client](https://github.com/gpt4free/g4f.dev/blob/main/docs/async_client.md)
   - **Requests API from G4F:** [/docs/requests](https://github.com/gpt4free/g4f.dev/blob/main/docs/requests.md)
   - **File API from G4F:** [/docs/file](https://github.com/gpt4free/g4f.dev/blob/main/docs/file.md)
   - **PydanticAI and LangChain Integration for G4F:** [/docs/pydantic_ai](https://github.com/gpt4free/g4f.dev/blob/main/docs/pydantic_ai.md)
   - **Legacy API with python modules:** [/docs/legacy](https://github.com/gpt4free/g4f.dev/blob/main/docs/legacy.md)
   - **G4F - Media Documentation (Image, Audio and Video)** [/docs/media](https://github.com/gpt4free/g4f.dev/blob/main/docs/media.md) *(New)*

---

### Powered by Pollinations AI

**🌟 Pollinations AI**

<img src="https://image.pollinations.ai/prompt/Create+a+logo+for+Pollinations+AI+featuring+an+abstract+flower+blooming+digital+petals+glowing+center+futuristic+font+Pollinations+AI?width=512&height=256&nologo=true" height="128">

A creative AI content platform that generates images, audios, and other media using advanced generative models. Pollinations AI empowers users and developers to turn text into visuals and multimedia experiences.

> [pollinations/pollinations on GitHub](https://github.com/pollinations/pollinations)

---

### Powered by GPT4Free

**💸 MoneyPrinter**

<img src="https://image.pollinations.ai/prompt/Create+a+logo+for+MoneyPrinter+glowing+center+futuristic+font?width=512&height=256&nologo=true" height="128">

MoneyPrinter V2 cranks up the automation for making money online. It’s a complete overhaul of the original MoneyPrinter, rebuilt from the ground up for more features and a plug-and-play, modular design. MPV2 takes the grind and guesswork out of online income: just set it up, let it run, and watch your earnings stack.

> [FujiwaraChoki/MoneyPrinterV2 on GitHub](https://github.com/FujiwaraChoki/MoneyPrinterV2)

> [Full list of GPT4Free powered sites and tools](https://github.com/gpt4free/g4f.dev/blob/main/docs/powered-by.md)

## 🤝 Contribute
We welcome contributions from the community. Whether you're adding new providers or features, or simply fixing typos and making small improvements, your input is valued. Creating a pull request is all it takes – our co-pilot will handle the code review process. Once all changes have been addressed, we'll merge the pull request into the main branch and release the updates at a later time.

###### Guide: How do i create a new Provider?
   - **Read:** [Create Provider Guide](https://github.com/gpt4free/g4f.dev/blob/main/docs/guides/create_provider.md)

###### Guide: How can AI help me with writing code?
   - **Read:** [AI Assistance Guide](https://github.com/gpt4free/g4f.dev/blob/main/docs/guides/help_me.md)



## Contributors
A list of all contributors is available [here](https://github.com/xtekky/gpt4free/graphs/contributors)

<a href="https://github.com/xtekky" target="_blank"><img src="https://avatars.githubusercontent.com/u/98614666?v=4&s=45" width="45" title="xtekky"></a>
<a href="https://github.com/hlohaus" target="_blank"><img src="https://avatars.githubusercontent.com/u/983577?v=4&s=45" width="45" title="hlohaus"></a>
<a href="https://github.com/kqlio67" target="_blank"><img src="https://avatars.githubusercontent.com/u/166700875?v=4&s=45" width="45" title="kqlio67"></a>
<a href="https://github.com/bagusindrayana" target="_blank"><img src="https://avatars.githubusercontent.com/u/36830534?v=4&s=45" width="45" title="bagusindrayana"></a>
<a href="https://github.com/sudouser777" target="_blank"><img src="https://avatars.githubusercontent.com/u/22415463?v=4&s=45" width="45" title="sudouser777"></a>
<a href="https://github.com/thatlukinhasguy1" target="_blank"><img src="https://avatars.githubusercontent.com/u/139662282?v=4&s=45" width="45" title="thatlukinhasguy1"></a>
<a href="https://github.com/Commenter123321" target="_blank"><img src="https://avatars.githubusercontent.com/u/36051603?v=4&s=45" width="45" title="Commenter123321"></a>
<a href="https://github.com/DanielShemesh" target="_blank"><img src="https://avatars.githubusercontent.com/u/20585236?v=4&s=45" width="45" title="DanielShemesh"></a>
<a href="https://github.com/Luneye" target="_blank"><img src="https://avatars.githubusercontent.com/u/73485421?v=4&s=45" width="45" title="Luneye"></a>
<a href="https://github.com/foxfire52" target="_blank"><img src="https://avatars.githubusercontent.com/u/185073927?v=4&s=45" width="45" title="foxfire52"></a>
<a href="https://github.com/ezerinz" target="_blank"><img src="https://avatars.githubusercontent.com/u/100193740?v=4&s=45" width="45" title="ezerinz"></a>
<a href="https://github.com/enganese" target="_blank"><img src="https://avatars.githubusercontent.com/u/69082498?v=4&s=45" width="45" title="enganese"></a>
<a href="https://github.com/Lin-jun-xiang" target="_blank"><img src="https://avatars.githubusercontent.com/u/63782903?v=4&s=45" width="45" title="Lin-jun-xiang"></a>
<a href="https://github.com/nullstreak" target="_blank"><img src="https://avatars.githubusercontent.com/u/139914347?v=4&s=45" width="45" title="nullstreak"></a>
<a href="https://github.com/valerii-chirkov" target="_blank"><img src="https://avatars.githubusercontent.com/u/81074936?v=4&s=45" width="45" title="valerii-chirkov"></a>
<a href="https://github.com/MIDORIBIN" target="_blank"><img src="https://avatars.githubusercontent.com/u/25425217?v=4&s=45" width="45" title="MIDORIBIN"></a>
<a href="https://github.com/repollo" target="_blank"><img src="https://avatars.githubusercontent.com/u/2671466?v=4&s=45" width="45" title="repollo"></a>
<a href="https://github.com/hpsj" target="_blank"><img src="https://avatars.githubusercontent.com/u/54535414?v=4&s=45" width="45" title="hpsj"></a>
<a href="https://github.com/taiyi747" target="_blank"><img src="https://avatars.githubusercontent.com/u/63543716?v=4&s=45" width="45" title="taiyi747"></a>
<a href="https://github.com/zukixa" target="_blank"><img src="https://avatars.githubusercontent.com/u/56563509?v=4&s=45" width="45" title="zukixa"></a>
<a href="https://github.com/ostix360" target="_blank"><img src="https://avatars.githubusercontent.com/u/55257054?v=4&s=45" width="45" title="ostix360"></a>
<a href="https://github.com/WdR-Tech" target="_blank"><img src="https://avatars.githubusercontent.com/u/143020293?v=4&s=45" width="45" title="WdR-Tech"></a>
<a href="https://github.com/HexyeDEV" target="_blank"><img src="https://avatars.githubusercontent.com/u/65314629?v=4&s=45" width="45" title="HexyeDEV"></a>
<a href="https://github.com/9fo" target="_blank"><img src="https://avatars.githubusercontent.com/u/71867245?v=4&s=45" width="45" title="9fo"></a>
<a href="https://github.com/devAdityaa" target="_blank"><img src="https://avatars.githubusercontent.com/u/77636021?v=4&s=45" width="45" title="devAdityaa"></a>
<a href="https://github.com/24rr" target="_blank"><img src="https://avatars.githubusercontent.com/u/109844019?v=4&s=45" width="45" title="24rr"></a>
<a href="https://github.com/zeng-rr" target="_blank"><img src="https://avatars.githubusercontent.com/u/47846202?v=4&s=45" width="45" title="zeng-rr"></a>
<a href="https://github.com/rkihacker" target="_blank"><img src="https://avatars.githubusercontent.com/u/182319878?v=4&s=45" width="45" title="rkihacker"></a>
<a href="https://github.com/naa7" target="_blank"><img src="https://avatars.githubusercontent.com/u/44613678?v=4&s=45" width="45" title="naa7"></a>
<a href="https://github.com/ramon-victor" target="_blank"><img src="https://avatars.githubusercontent.com/u/13617054?v=4&s=45" width="45" title="ramon-victor"></a>
<a href="https://github.com/eltociear" target="_blank"><img src="https://avatars.githubusercontent.com/u/22633385?v=4&s=45" width="45" title="eltociear"></a>
<a href="https://github.com/kggn" target="_blank"><img src="https://avatars.githubusercontent.com/u/95663228?v=4&s=45" width="45" title="kggn"></a>
<a href="https://github.com/xiangsx" target="_blank"><img src="https://avatars.githubusercontent.com/u/29322721?v=4&s=45" width="45" title="xiangsx"></a>
<a href="https://github.com/ggindinson" target="_blank"><img src="https://avatars.githubusercontent.com/u/97807772?v=4&s=45" width="45" title="ggindinson"></a>
<a href="https://github.com/ahobsonsayers" target="_blank"><img src="https://avatars.githubusercontent.com/u/32173585?v=4&s=45" width="45" title="ahobsonsayers"></a>
<a href="https://github.com/mache102" target="_blank"><img src="https://avatars.githubusercontent.com/u/91365155?v=4&s=45" width="45" title="mache102"></a>
<a href="https://github.com/kogakisaki" target="_blank"><img src="https://avatars.githubusercontent.com/u/95165750?v=4&s=45" width="45" title="kogakisaki"></a>
<a href="https://github.com/Andrew-Tsegaye" target="_blank"><img src="https://avatars.githubusercontent.com/u/91322467?v=4&s=45" width="45" title="Andrew-Tsegaye"></a>
<a href="https://github.com/omidima" target="_blank"><img src="https://avatars.githubusercontent.com/u/47784584?v=4&s=45" width="45" title="omidima"></a>
<a href="https://github.com/nonk123" target="_blank"><img src="https://avatars.githubusercontent.com/u/43842467?v=4&s=45" width="45" title="nonk123"></a>
<a href="https://github.com/MaxKUlish1" target="_blank"><img src="https://avatars.githubusercontent.com/u/93388714?v=4&s=45" width="45" title="MaxKUlish1"></a>
<a href="https://github.com/AymaneHrouch" target="_blank"><img src="https://avatars.githubusercontent.com/u/36491424?v=4&s=45" width="45" title="AymaneHrouch"></a>
<a href="https://github.com/Eikosa" target="_blank"><img src="https://avatars.githubusercontent.com/u/20538090?v=4&s=45" width="45" title="Eikosa"></a>
<a href="https://github.com/localagi" target="_blank"><img src="https://avatars.githubusercontent.com/u/132956819?v=4&s=45" width="45" title="localagi"></a>
<a href="https://github.com/thebigbone" target="_blank"><img src="https://avatars.githubusercontent.com/u/95130644?v=4&s=45" width="45" title="thebigbone"></a>
<a href="https://github.com/kailust" target="_blank"><img src="https://avatars.githubusercontent.com/u/82623773?v=4&s=45" width="45" title="kailust"></a>
<a href="https://github.com/ading2210" target="_blank"><img src="https://avatars.githubusercontent.com/u/71154407?v=4&s=45" width="45" title="ading2210"></a>
<a href="https://github.com/Zero6992" target="_blank"><img src="https://avatars.githubusercontent.com/u/89479282?v=4&s=45" width="45" title="Zero6992"></a>
<a href="https://github.com/mishl-dev" target="_blank"><img src="https://avatars.githubusercontent.com/u/91066601?v=4&s=45" width="45" title="mishl-dev"></a>
<a href="https://github.com/ElonGaties" target="_blank"><img src="https://avatars.githubusercontent.com/u/59313695?v=4&s=45" width="45" title="ElonGaties"></a>
<a href="https://github.com/TotoB12" target="_blank"><img src="https://avatars.githubusercontent.com/u/91705868?v=4&s=45" width="45" title="TotoB12"></a>
<a href="https://github.com/malivinayak" target="_blank"><img src="https://avatars.githubusercontent.com/u/66154908?v=4&s=45" width="45" title="malivinayak"></a>
<a href="https://github.com/Zedai00" target="_blank"><img src="https://avatars.githubusercontent.com/u/33319711?v=4&s=45" width="45" title="Zedai00"></a>
<a href="https://github.com/catmeowjiao" target="_blank"><img src="https://avatars.githubusercontent.com/u/138079152?v=4&s=45" width="45" title="catmeowjiao"></a>
<a href="https://github.com/cifer-sudo" target="_blank"><img src="https://avatars.githubusercontent.com/u/60644739?v=4&s=45" width="45" title="cifer-sudo"></a>
<a href="https://github.com/eminemkun" target="_blank"><img src="https://avatars.githubusercontent.com/u/49590289?v=4&s=45" width="45" title="eminemkun"></a>
<a href="https://github.com/kafmws" target="_blank"><img src="https://avatars.githubusercontent.com/u/33590879?v=4&s=45" width="45" title="kafmws"></a>
<a href="https://github.com/najam-tariq" target="_blank"><img src="https://avatars.githubusercontent.com/u/103676132?v=4&s=45" width="45" title="najam-tariq"></a>
<a href="https://github.com/ochen1" target="_blank"><img src="https://avatars.githubusercontent.com/u/59662605?v=4&s=45" width="45" title="ochen1"></a>
<a href="https://github.com/r1di" target="_blank"><img src="https://avatars.githubusercontent.com/u/33724815?v=4&s=45" width="45" title="r1di"></a>
<a href="https://github.com/sagadav" target="_blank"><img src="https://avatars.githubusercontent.com/u/42406802?v=4&s=45" width="45" title="sagadav"></a>
<a href="https://github.com/snyk-bot" target="_blank"><img src="https://avatars.githubusercontent.com/u/19733683?v=4&s=45" width="45" title="snyk-bot"></a>
<a href="https://github.com/vatva691" target="_blank"><img src="https://avatars.githubusercontent.com/u/30290559?v=4&s=45" width="45" title="vatva691"></a>
<a href="https://github.com/Qustelm" target="_blank"><img src="https://avatars.githubusercontent.com/u/83110161?v=4&s=45" width="45" title="Qustelm"></a>
<a href="https://github.com/HyiKi" target="_blank"><img src="https://avatars.githubusercontent.com/u/55942998?v=4&s=45" width="45" title="HyiKi"></a>
<a href="https://github.com/0dminnimda" target="_blank"><img src="https://avatars.githubusercontent.com/u/52697657?v=4&s=45" width="45" title="0dminnimda"></a>
<a href="https://github.com/Akash98Sky" target="_blank"><img src="https://avatars.githubusercontent.com/u/37451227?v=4&s=45" width="45" title="Akash98Sky"></a>
<a href="https://github.com/adeyinkaezra123" target="_blank"><img src="https://avatars.githubusercontent.com/u/65364356?v=4&s=45" width="45" title="adeyinkaezra123"></a>
<a href="https://github.com/Giancarlo-Ma" target="_blank"><img src="https://avatars.githubusercontent.com/u/65126107?v=4&s=45" width="45" title="Giancarlo-Ma"></a>
<a href="https://github.com/gran4" target="_blank"><img src="https://avatars.githubusercontent.com/u/80655391?v=4&s=45" width="45" title="gran4"></a>
<a href="https://github.com/guspan-tanadi" target="_blank"><img src="https://avatars.githubusercontent.com/u/36249910?v=4&s=45" width="45" title="guspan-tanadi"></a>
<a href="https://github.com/oubrax" target="_blank"><img src="https://avatars.githubusercontent.com/u/72103863?v=4&s=45" width="45" title="oubrax"></a>
<a href="https://github.com/hansipie" target="_blank"><img src="https://avatars.githubusercontent.com/u/5460714?v=4&s=45" width="45" title="hansipie"></a>
<a href="https://github.com/GetTuh" target="_blank"><img src="https://avatars.githubusercontent.com/u/27581581?v=4&s=45" width="45" title="GetTuh"></a>
<a href="https://github.com/kushal34712" target="_blank"><img src="https://avatars.githubusercontent.com/u/98145879?v=4&s=45" width="45" title="kushal34712"></a>
<a href="https://github.com/Fubge" target="_blank"><img src="https://avatars.githubusercontent.com/u/115476150?v=4&s=45" width="45" title="Fubge"></a>
<a href="https://github.com/Niapoll" target="_blank"><img src="https://avatars.githubusercontent.com/u/64135936?v=4&s=45" width="45" title="Niapoll"></a>
<a href="https://github.com/OmiiiDev" target="_blank"><img src="https://avatars.githubusercontent.com/u/103533638?v=4&s=45" width="45" title="OmiiiDev"></a>
<a href="https://github.com/RasyiidWho" target="_blank"><img src="https://avatars.githubusercontent.com/u/19422415?v=4&s=45" width="45" title="RasyiidWho"></a>
<a href="https://github.com/RavenOwO" target="_blank"><img src="https://avatars.githubusercontent.com/u/118295106?v=4&s=45" width="45" title="RavenOwO"></a>
<a href="https://github.com/anonymousx97" target="_blank"><img src="https://avatars.githubusercontent.com/u/88324835?v=4&s=45" width="45" title="anonymousx97"></a>
<a href="https://github.com/krjordan" target="_blank"><img src="https://avatars.githubusercontent.com/u/10234150?v=4&s=45" width="45" title="krjordan"></a>
<a href="https://github.com/SilverMarcs" target="_blank"><img src="https://avatars.githubusercontent.com/u/77480421?v=4&s=45" width="45" title="SilverMarcs"></a>
<a href="https://github.com/Yusufibin" target="_blank"><img src="https://avatars.githubusercontent.com/u/71589435?v=4&s=45" width="45" title="Yusufibin"></a>
<a href="https://github.com/yuri-val" target="_blank"><img src="https://avatars.githubusercontent.com/u/15129796?v=4&s=45" width="45" title="yuri-val"></a>
<a href="https://github.com/yousefnegmeldin" target="_blank"><img src="https://avatars.githubusercontent.com/u/96620955?v=4&s=45" width="45" title="yousefnegmeldin"></a>
<a href="https://github.com/perklet" target="_blank"><img src="https://avatars.githubusercontent.com/u/1035487?v=4&s=45" width="45" title="perklet"></a>
<a href="https://github.com/varshney-yash" target="_blank"><img src="https://avatars.githubusercontent.com/u/107148830?v=4&s=45" width="45" title="varshney-yash"></a>
<a href="https://github.com/Yoxmo" target="_blank"><img src="https://avatars.githubusercontent.com/u/94254616?v=4&s=45" width="45" title="Yoxmo"></a>
<a href="https://github.com/yjg30737" target="_blank"><img src="https://avatars.githubusercontent.com/u/55078043?v=4&s=45" width="45" title="yjg30737"></a>
<a href="https://github.com/williamstein" target="_blank"><img src="https://avatars.githubusercontent.com/u/1276278?v=4&s=45" width="45" title="williamstein"></a>
<a href="https://github.com/ZachKLYeh" target="_blank"><img src="https://avatars.githubusercontent.com/u/105150034?v=4&s=45" width="45" title="ZachKLYeh"></a>
<a href="https://github.com/alvarosoaress" target="_blank"><img src="https://avatars.githubusercontent.com/u/13721147?v=4&s=45" width="45" title="alvarosoaress"></a>
<a href="https://github.com/bruvv" target="_blank"><img src="https://avatars.githubusercontent.com/u/3063928?v=4&s=45" width="45" title="bruvv"></a>
<a href="https://github.com/carlinhoshk" target="_blank"><img src="https://avatars.githubusercontent.com/u/40872405?v=4&s=45" width="45" title="carlinhoshk"></a>
<a href="https://github.com/cckuailong" target="_blank"><img src="https://avatars.githubusercontent.com/u/10824150?v=4&s=45" width="45" title="cckuailong"></a>
<a href="https://github.com/chinmay7016" target="_blank"><img src="https://avatars.githubusercontent.com/u/75988613?v=4&s=45" width="45" title="chinmay7016"></a>
<a href="https://github.com/diaodeng" target="_blank"><img src="https://avatars.githubusercontent.com/u/108243171?v=4&s=45" width="45" title="diaodeng"></a>
<a href="https://github.com/monosans" target="_blank"><img src="https://avatars.githubusercontent.com/u/76561516?v=4&s=45" width="45" title="monosans"></a>
<a href="https://github.com/Ayushpanditmoto" target="_blank"><img src="https://avatars.githubusercontent.com/u/31253617?v=4&s=45" width="45" title="Ayushpanditmoto"></a>
<span></span>
<img src="https://avatars.githubusercontent.com/u/71154407?s=45&v=4" width="45" title="ading2210">
<img src="https://avatars.githubusercontent.com/u/12299238?s=45&v=4" width="45" title="xqdoo00o">
<img src="https://avatars.githubusercontent.com/u/97126670?s=45&v=4" width="45" title="nathanrchn">
<img src="https://avatars.githubusercontent.com/u/81407603?v=4&s=45" width="45" title="dsdanielpark">
<img src="https://avatars.githubusercontent.com/u/55200481?v=4&s=45" width="45" title="missuo">

- The [`Vercel.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/Vercel.py) file contains code from [vercel-llm-api](https://github.com/ading2210/vercel-llm-api) by [@ading2210](https://github.com/ading2210)
- The [`har_file.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/openai/har_file.py) has input from [xqdoo00o/ChatGPT-to-API](https://github.com/xqdoo00o/ChatGPT-to-API)
- The [`PerplexityLabs.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/PerplexityLabs.py) has input from [nathanrchn/perplexityai](https://github.com/nathanrchn/perplexityai)
- The [`Gemini.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/needs_auth/Gemini.py) has input from [dsdanielpark/Gemini-API](https://github.com/dsdanielpark/Gemini-API)
- The [`MetaAI.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/MetaAI.py) file contains code from [meta-ai-api](https://github.com/Strvm/meta-ai-api) by [@Strvm](https://github.com/Strvm)
- The [`proofofwork.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/openai/proofofwork.py) has input from [missuo/FreeGPT35](https://github.com/missuo/FreeGPT35)
- The [`Gemini.py`](https://github.com/xtekky/gpt4free/blob/main/g4f/Provider/needs_auth/Gemini.py) has input from [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API)

_Having input implies that the AI's code generation utilized it as one of many sources._


## ©️ Copyright

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


## ⭐ Star History

<!--![Star History Chart](https://api.star-history.com/svg?repos=xtekky/gpt4free&type=Date)-->

<img src="https://github.com/user-attachments/assets/1624121d-4ee1-4553-913e-00dbd937e61f" width="800" alt="Star History Chart">

## 📄 License

<table>
  <tr>
     <td>
       <p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/GPLv3_Logo.svg/1200px-GPLv3_Logo.svg.png" width="120"></img>
    </td>
    <td> 
      <img src="https://img.shields.io/badge/License-GNU_GPL_v3.0-red.svg"/> <br> 
This project is licensed under <a href="https://github.com/xtekky/gpt4free/blob/main/LICENSE">GNU_GPL_v3.0</a>.
    </td>
  </tr>
</table>

---

<p align="right">(<a href="#top">🔼 Back to top</a>)</p>
