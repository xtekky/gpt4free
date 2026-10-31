# Sandbox desktop menu — install & open desktop tools

![](home.png)

![](sandbox.png)

Entries behind the sandbox desktop's **Applications** menu
(`docker/Dockerfile-sandbox-desktop`), grouped into twelve top-level menus —
**g4f**, **Browsers**, **Files & Editors**, **Coding Agents**, **Agent Skills**,
**Gateways**, **Local AI**, **Chat UIs**, **Writing**, **Media**,
**Sandboxes** and **Assistants** — instead of one flat list under Development.
Each **Install & Open …** item opens a terminal maximized to the sandbox
screen, without window borders or menubar:

```
xfce4-terminal --disable-server --maximize --hide-borders --hide-menubar --icon=g4f-sandbox-<tool> --command="/usr/local/bin/sandbox-cli <tool>"
```

`--disable-server` gives every entry its own terminal window. Without it,
xfce4-terminal hands the new window to the already running instance as a tab,
so all apps opened from the menu end up in the same window.

A multi-word `--title` must be quoted (`--title="Sandbox Settings"`): the
desktop-entry `Exec` line is split on whitespace, so an unquoted title makes
xfce4-terminal treat the second word as an option and refuse to start
(`Unknown option "Settings"`).

The **g4f Live Wallpaper** entry runs `bin/g4f-livewall`, a self-contained
truecolour ANSI animation (drifting nebula, stars, orbits and a shimmering g4f
emblem). The desktop entrypoint also opens it at startup; set
`G4F_WALLPAPER=0` to skip it.

`sandbox-cli` installs tools on **first click** — a vendor install script,
an npm package, a `pip install --user`, or Google's Chrome `.deb` — then runs
them in that same terminal (Chrome opens on the desktop instead). The
 CLI is preinstalled in the image. Node.js 24 and  are also
preinstalled (from the official  image) for the npm-based tools; the rest is installed on demand.

![](agents.png)

| Menu entry               | Tool id     | Install recipe              | Binary         |
|--------------------------|-------------|-----------------------------|----------------|
| Install & Open OpenCode  | `opencode`  | opencode.ai v2 install script | `opencode`     |
| Install & Open Cindy     | `cindy`     | GitHub release Debian package, unpacked by its user installer; adds the g4f provider on first run | `cindy` |
| Install & Open Oh My Pi  | `omp`       | omp.sh install script + `omp plugin install pi-freeflow` | `omp` |
| Install & Open Pi        | `pi`        | pi.dev install script + `pi install npm:pi-freeflow` | `pi` |
| Install & Run Freeflow API | `pi-freeflow` | Pi/OMP package, daemon on port 28180 | `node` |
| Install & Open Kilo      | `kilocode`  | npm `@kilocode/cli`         | `kilo`         |
| Install & Open Antigravity | `antigravity` | antigravity.google install script | `agy`    |
| Install & Open Gemini    | `gemini-cli`| npm `@google/gemini-cli`    | `gemini`       |
| Install & Open Qwen      | `qwencode`  | npm `@qwen-code/qwen-code`  | `qwen`         |
| Install & Open Claude    | `claude-code`| npm `@anthropic-ai/claude-code` | `claude`   |
| Install & Open Codex     | `codex`     | npm `@openai/codex`         | `codex`        |
| Install & Open Copilot   | `copilot`   | npm `@github/copilot`       | `copilot`      |
| Open g4f-go CLI          | `g4f-go`    | preinstalled in the image   | `g4f-go`       |
| Open Remote Desktop      | `remote-desktop` | g4f-go bundled module + pip `qrcode`, `python-xlib`, `pynput` | `g4f-go` |
| Sandbox Settings         | `settings`  | none (writes a settings file) | —            |
| Update or Change g4f Package | `package` | `g4f-go install` (pip) | —           |
| Reset Sandbox Settings   | `reset`     | none (removes the settings file) | —         |
| Public URLs              | `urls`      | none (prints the g4f API URLs) | —           |
| Host Names               | `hosts`     | none (prints the `/etc/hosts` entries) | —    |
| g4f Live Wallpaper       | `wallpaper` | none (runs `/usr/local/bin/g4f-livewall`) | —  |
| Install & Open Cline     | `cline`     | npm `cline`                 | `cline`        |
| Install & Open Amp       | `amp`       | npm `@ampcode/cli`          | `amp`          |
| Install & Open Goose     | `goose`     | block/goose install script  | `goose`        |
| Install & Open Aider     | `aider`     | pip `aider-chat` (user)     | `aider`        |
| Install & Open Cursor    | `cursor`    | cursor.com/install script   | `cursor-agent` |
| Install & Open Chrome    | `chrome`    | Google's Chrome `.deb` (apt)| `start-browser` |
| Install & Open File Manager | `files` | apt `thunar` (Debian)       | `thunar`       |
| Install & Open File Manager (Nemo) | `nemo` | apt `nemo` + gvfs/thumbnail helpers (Debian) | `nemo`         |
| Install & Run Lightpanda | `lightpanda`| `g4f-go browser install`    | `g4f-go`       |
| Install & Open OmniRoute | `omniroute` | npm `omniroute`             | `omniroute`    |
| Install & Open Ollama    | `ollama`    | ollama.com install script   | `ollama`       |
| Install & Open llama.cpp | `llamacpp`  | GitHub release tarball      | `llama-server` |
| Install & Open LM Studio | `lmstudio`  | lmstudio.ai install script  | `lms`          |
| Install & Open Jan       | `jan`       | GitHub release AppImage     | `jan`          |
| Install & Open LibreChat | `librechat` | git clone + `npm run reinstall` + MongoDB | `npm` |
| Install & Open LobeChat  | `lobechat`  | GitHub release AppImage     | `lobechat`     |
| Install & Open Msty Studio | `msty`    | vendor "latest" AppImage    | `msty`         |
| Install & Open SillyTavern | `sillytavern` | git clone + `npm install` | `node`       |
| Install & Open Open WebUI | `open-webui` | pip `open-webui` (user) | `open-webui` |
| Install & Open Open WebUI Computer | `cptr` | pip `cptr` (user)     | `cptr`         |
| Install & Open VS Code Web | `code-server` | code-server.dev install script | `code-server` |
| Install & Open Kate      | `kate`      | apt `kate` (Debian)         | `kate`         |
| Install & Open OpenClaw  | `openclaw`  | openclaw.ai install script  | `openclaw`     |
| Install & Open Crush     | `crush`     | npm `@charmland/crush`      | `crush`        |
| Install & Open Continue  | `continue-cli` | npm `@continuedev/cli`   | `cn`           |
| Install & Open Grok      | `grok`      | x.ai/cli install script     | `grok`         |
| Install & Open Droid     | `droid`     | app.factory.ai/cli script   | `droid`        |
| Install & Open OpenHands | `openhands` | install.openhands.dev script| `openhands`    |
| Install & Open gptme     | `gptme`     | pip `gptme`                 | `gptme`        |
| Install & Open Plandex   | `plandex`   | plandex.ai install script   | `plandex`      |
| Install & Open Hermes    | `hermes`    | hermes-agent.nousresearch.com script | `hermes` |
| Install & Open Oh My Hermes | `omh`    | npm `oh-my-hermes` + `omh setup --yes` | `omh` |
| Install & Open Kiro      | `kiro`      | cli.kiro.dev install script | `kiro-cli`     |
| Install & Open Kimi      | `kimi`      | code.kimi.com install script| `kimi`         |
| Install & Open Roomote   | `roomote`   | get.roomote.dev install script | `roomote`      |
| Install & Open Auggie    | `auggie`    | npm `@augmentcode/auggie`   | `auggie`       |
| Install & Open Qoder     | `qoder`     | qoder.com install script    | `qodercli`     |
| Install & Open iFlow     | `iflow`     | npm `@iflow-ai/iflow-cli`   | `iflow`        |
| Install & Open Mistral Vibe | `vibe`   | mistral.ai/vibe install script | `vibe`      |
| Install & Open SWE-agent | `sweagent`  | uv tool `--editable` clone of SWE-agent (Python 3.13) | `sweagent` |
| Install & Open DeepSeek Harness | `dsh` | npm `@deepseek-ai/dsh` (listed under Chat UIs, Assistants and Coding Agents) | `dsh` |
| Install & Open Paseo     | `paseo`     | npm `@getpaseo/cli`         | `paseo`        |
| Install & Open T3 Code   | `t3code`    | t3.codes install script     | `t3`           |
| Install & Open Fabric    | `fabric`    | fabric installer script     | `fabric`       |
| Install & Open LiteLLM   | `litellm`   | pip `litellm[proxy]`        | `litellm`      |
| Install & Open Langfuse CLI | `langfuse` | npm `@langfuse/cli`      | `langfuse`     |
| Install & Run Proxy Server | `proxy`   | pip `mitmproxy`             | `mitmproxy`    |
| Install & Run Tunnel Client | `tunnel` | GitHub release `cloudflared` | `cloudflared`  |
| Install & Open ShellGPT  | `sgpt`      | pip `shell-gpt`             | `sgpt`         |
| Install & Open LLM CLI   | `llm`       | pip `llm`                   | `llm`          |
| Install & Open Hugging Face Spaces | `huggingface` | hf.co standalone CLI | `hf` |
| Install & Open assistant-ui | `assistant-ui` | npx `assistant-ui@latest create` + `npm install` | `npm`      |
| Install & Open Polli     | `polli`     | npm `@pollinations/cli`     | `polli`        |
| Install & Open Polli Sandboxes | `polli-sandbox` | npm `@pollinations/cli` + apt `openssh-client` | `polli` |
| Find & Add AI Tools      | `discover`  | uses OpenCode (installed if needed) + the g4f API | `opencode` |
| Install Humanizer        | `humanizer` | npx `skills add blader/humanizer` | — |
| Install & Open AI Text Detector | `ai-detect` | pip `git+lynote-ai/ai-text-detector` | `ai-detect` |
| Install & Open AI Image Detector | `aidetect` | pip `git+lynote-ai/ai-image-detector` | `aidetect` |
| Install & Open OpenBot   | `openbot`   | GitHub release AppImage     | `openbot`      |
| Install & Open ThinkRail | `thinkrail` | JetBrains install script    | `thinkrail`    |
| Install & Open Orca      | `orca`      | GitHub latest AppImage (stable URL) | `orca`, `orca-ide` |
| Install & Open VoiceStudio | `voicestudio` | GitHub release AppImage (x64) | `voicestudio` |
| Install & Open HyperFrames | `hyperframes` | npm `hyperframes`         | `hyperframes`  |
| Install & Open DeepTutor | `deeptutor` | pip `deeptutor[full]`       | `deeptutor`    |
| Install & Open nanobot   | `nanobot`   | pip `nanobot-ai`            | `nanobot`      |
| Install & Open Agent Reach | `agent-reach` | uv tool install (GitHub release zip) | `agent-reach` |
| Install ECC              | `ecc`       | npx `ecc-universal@2.2.3`   | —              |
| Install Superpowers      | `superpowers` | `opencode plugin add` (git) | —            |
| Install Graphify         | `graphify`  | pip `graphifyy` + `graphify install --platform agents` | `graphify` |
| Install Claude-Mem       | `claude-mem` | npx `claude-mem@15.4.0 install --ide opencode` | — |
| Open g4f Docs in Glow   | `glow`      | apt `glow` + `chafa` (installed on first open) | `glow` |
| Open g4f Docs           | `docs`      | Google Chrome (installed on first open) | `google-chrome` |

Beyond the coding agents, the menu also carries **gateways** (`omniroute`,
`litellm`, `pi-freeflow`) that expose an OpenAI-compatible endpoint, together
with a debugging **proxy server** (`proxy`, mitmproxy), a **tunnel** entry
(`tunnel`, Cloudflare's cloudflared) that publishes a local port on a
public HTTPS URL, and the **Langfuse CLI** (`langfuse`) for inspecting LLM
traces and prompts, **local AI** runtimes
(`ollama`, `llamacpp`, `lmstudio`, `jan`) that serve models on
the machine itself, **chat UIs** (`librechat`, `lobechat`, `msty`,
`sillytavern`, `open-webui`, `assistant-ui`) that put a chat front end on top of
the g4f API, browser and desktop **editors** (`code-server`, VS Code Web served
by code-server, and the KDE **Kate** text editor), **assistants** such as
Open WebUI Computer (`cptr`), and **assistant/agent CLIs**
(`openclaw`, `fabric`, `sgpt`, `gptme`, `llm`) that live in the terminal,
the **Writing** tools that humanize or score generated copy, and an **Agent
Skills** menu that installs skill packs and cross-session memory into the
sandbox's coding agents.
Install scripts land their
binary in `~/.local/bin`, `~/.kimi-code/bin`, `~/.qoder/bin`,
`~/.lmstudio/bin` or `/usr/local/bin`, all on `sandbox-cli`'s `PATH`.

**Oh My Hermes** (`omh`) is the npm-packaged operating layer for Hermes Agent:
the entry installs Hermes first, then `omh`, runs `omh setup --yes` to register
its managed skills, and finally points the same Hermes home at the g4f API, so
the `g4f` model alias and **Sandbox Settings** model survive setup. Bare `omh`
opens its own TUI over `hermes --tui`; the workflow commands (`/dag-ulw`,
`ulw-plan`, `ulw-work`, ...) are routed by Hermes.

**Hugging Face Spaces** (`huggingface`) installs the standalone `hf` CLI into
its own virtualenv under `~/.hf-cli` and writes the wrapper to `~/.local/bin`.
A bare click lists trending Spaces and then leaves the shell open for
`hf spaces info <owner/name>`, `hf spaces ls <owner/name>` or
`hf download <owner/name> --repo-type space`; extra arguments run directly, for
example `sandbox-cli huggingface spaces ls --search 3d`. Login is only needed
for private or gated repositories.

**Humanizer** (`humanizer`) installs the `blader/humanizer` skill through the
Skills CLI into `~/.agents/skills/humanizer` and links it for every supported
agent, where it answers to `/humanizer`. It has no standalone command, so the
entry prints where it landed and keeps a shell open. **AI Text Detector**
(`ai-detect`) is the lightweight, offline `lynote-ai/ai-text-detector` CLI and
**AI Image Detector** (`aidetect`) is `lynote-ai/ai-image-detector`; both are
installed straight from their git repositories with `pip install --user` on
first click.
The image detector pulls the torch stack and is by far the largest download in
the menu; the text detector is pure Python.

**OpenBot**, **Orca** and **VoiceStudio** are Electron desktop apps published
as GitHub release AppImages. They share `install_appimage`: the latest tag is
resolved from the GitHub API, the AppImage is unpacked into
`~/.local/opt/<tool>` (the container has no FUSE), and a launcher in
`~/.local/bin` pins `APPDIR` and adds `--no-sandbox` only when unprivileged
user namespaces are unavailable. OpenBot and Orca ship x64 and arm64 assets;
VoiceStudio only ships x64. The entries open the apps detached on the desktop,
like Jan and LobeChat. Orca instead uses its stable
`/releases/latest/download/` AppImage URL (the API lookup is rate-limited) and
installs a launcher that turns on software rendering and disables the GPU,
because Orca's Chromium GPU process crashes on the sandbox's virtual display.
It is also available as `orca-ide`, the upstream name for the Linux CLI.
**ThinkRail** instead uses JetBrains' own installer,
which verifies a SHA-256 and writes `~/.local/bin/thinkrail`; the entry installs
`git` too, because ThinkRail needs it to open a project.

**HyperFrames** (`hyperframes`) is an npm CLI that renders HTML compositions to
MP4/WebM. A bare click prints its command list and leaves a shell open because
the CLI has no TUI of its own. **DeepTutor** (`deeptutor`) installs the
`deeptutor[full]` extra and starts its web app on
<http://127.0.0.1:3782>; the first run asks for a runtime home and a model
provider. **nanobot** (`nanobot`) installs `nanobot-ai` and starts its WebUI.
nanobot defaults to port 8765, which the Remote Desktop entry already uses, so
the menu passes `--port 8766`. **Agent Reach** (`agent-reach`) must not be
installed from PyPI (that name belongs to an unrelated project): it is a uv
tool installed from the project's GitHub release zip, and the first click runs
its idempotent `setup` once to install the default channels and agent skill.

The **Agent Skills** menu installs each tool into its native host instead of
copying files. **Superpowers** (`superpowers`) uses
`opencode plugin add 'superpowers@git+https://github.com/obra/superpowers.git'`
and is detected by the entry in `~/.config/opencode/opencode.json`. **ECC**
(`ecc`) runs `npx ecc-universal install --target opencode --profile core
--no-hooks` and lands under `~/.config/opencode` with an
`ecc-install-state.json` marker. **Graphify** (`graphify`) is the `graphifyy`
pip CLI; `graphify install --platform agents` copies its skill into
`~/.agents/skills/graphify`. **Claude-Mem** (`claude-mem`) installs the
OpenCode plugin through `npx claude-mem install --ide opencode`, which needs
`unzip` for Bun; the menu uses `--provider claude`, the documented local mode
that runs memory on the user's own Claude plan. It writes
`~/.config/opencode/plugins/claude-mem.js` and starts its worker on port
37700. Restart OpenCode after installing any of the four.

A few entries need a package the image does not ship, so `sandbox-cli` pulls
it in on first click: **Hermes** clones its own checkout and therefore needs
`git` (`ensure_git`), while **Kiro CLI** unpacks a `.zip` and **Claude-Mem**
installs Bun, so both need `unzip` (`ensure_unzip`). The two detectors reuse
`ensure_git`, because pip fetches them over `git+https` on first click. Hermes installs into
`~/.hermes/hermes-agent/.hermes/bin` and only adds `~/.local/bin` to the shell
rc, so `ensure_hermes` symlinks `hermes` into `~/.local/bin` itself. It runs
with `--non-interactive`, which skips the interactive `setup`/`gateway`
stages. The launcher always registers a `g4f` model alias (model `auto`) with
Hermes and, when a model has been chosen in **Sandbox Settings**, also points
the default `model.*` settings at g4f; otherwise run `hermes setup` in the
terminal to configure another provider by hand.

**SWE-agent** installs through **uv** instead of pip (`ensure_sweagent`),
because the image's embedded CPython is 3.15 and SWE-agent's pinned
dependencies ship no wheels for it. `ensure_uv` installs uv from astral.sh on
first click; `uv tool install --editable --python 3.13` then downloads a
matching CPython and links the binary into `~/.local/bin`. The install is
*editable* on purpose: SWE-agent asserts at import time that its `config/`,
`tools/` and `trajectories/` directories sit next to the package, but its wheel
does not carry them, so a plain `uv tool install` of the git checkout produces
a binary that crashes on start. `ensure_sweagent` therefore clones the repo
into `~/.swe-agent` (via `ensure_git`) and installs from that checkout.
SWE-agent additionally needs Docker at run time for its sandboxed
environments.

**Roomote** is the one entry that is not a CLI: `get.roomote.dev` is a host
installer that runs as root, installs Docker when it is missing, writes
`/opt/roomote` and starts a Compose stack from published GHCR images, leaving
the `roomote` management CLI in `/usr/local/bin`. `ensure_roomote` pipes it
through `as_root` with `--tls-mode internal`, so Caddy serves a self-signed
certificate and the install needs no public DNS or ACME round trip. The image
has no init system, so the systemd unit the installer writes is inert here and
the stack is started directly; the menu entry prints the first-admin `/setup`
link (`roomote setup-url`) and then follows the service logs.

**assistant-ui** is a chat front end for the g4f API: the app is created with
the upstream CLI (`npx assistant-ui@latest create`, see
[the installation docs](https://www.assistant-ui.com/docs/installation)), which
downloads the starter template — a Next.js app with the AI SDK chat route —
into `~/.assistant-ui`, and `npm install` pulls its dependencies. The launcher
then writes the g4f endpoint into the app: `.env.local` gets `G4F_API_BASE_URL`,
`G4F_API_KEY` and `G4F_MODEL`, `app/api/chat/route.ts` is replaced with a
route that talks to the g4f API through the OpenAI provider pointed at it, and
`next.config.ts` gets the sandbox host names in `allowedDevOrigins`. The
g4f API speaks the OpenAI protocol and serves `/v1/chat/completions`, so the
route uses the provider's chat model (its default model speaks the Responses
API instead, which g4f does not serve). The app therefore needs no provider
key, and the default model is the g4f **"auto"** model, which lets g4f pick the
provider and the model itself. A model chosen in **Sandbox Settings** replaces
it (`G4F_MODEL`, with the provider prefix g4f resolves itself). The route and
the config are written once: a marker comment keeps later launches from
overwriting edits made in the app. The dev server runs in the foreground on
`http://127.0.0.1:3000` (also `http://assistant-ui.g4f.local/`), and the
terminal stays open as its console.

The `allowedDevOrigins` entry is what makes the `assistant-ui.g4f.local` URL
usable: Next.js blocks dev-only resources (`_next` chunks, the HMR websocket)
whose `Origin` is not listed, and the browser sends the host name it was
reached through. Without it the page loads but its chunks and the HMR socket
come back `403 Unauthorized`. The list is derived from `G4F_DOMAIN_SUFFIX`, so
a renamed suffix keeps working.

**pi-freeflow** is an extension for Pi and Oh My Pi (OMP), not a standalone
CLI, and its 35 free models (OpenCode Zen, KiloCode Gateway, Cline, OVHcloud)
reach the menu three ways:

- **Oh My Pi** (`omp`) installs the prebuilt OMP binary and adds the extension
  with `omp plugin install pi-freeflow`.
- **Pi** (`pi`) installs Pi's managed CLI and adds the package with
  `pi install npm:pi-freeflow`; the installer puts `pi` under `~/.pi/agent/bin`,
  so the launcher links it into `~/.local/bin`.
- **Freeflow API** (`pi-freeflow`, under Gateways) runs the extension's own
  daemon in the foreground on `127.0.0.1:28180`, without an agent UI: an
  OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoint. The entry
  prints the base URL, the bearer token (`~/.pi/agent/token`, shared with the
  OMP/Pi sessions) and a ready-to-paste models command.

Both agent entries open the host's model picker at `/model → freeflow`; relays
are added with `/freeflow add`. The shared daemon writes
`~/.pi/agent/pi-freeflow.log` and normally retires about ten seconds after its
last client. The API entry copies the package out of `node_modules` (Node
refuses to type-strip files there) and stretches that lease-less grace window,
so it stays up until Ctrl-C. If OMP or Pi already owns the port, the entry
instead attaches as a client and renews that daemon's lease, so the endpoint
survives the agent session too. It is loopback-only: expose it with the Tunnel
entry (`sandbox-cli tunnel --url http://127.0.0.1:28180`) and keep the bearer
token in the client.

### Polli sandboxes (E2B)

**Polli Sandboxes** (under **Sandboxes**) wraps `polli sandbox` in one terminal:
list, create, connect (`ssh <id>.polli`), set a timeout, pause, show logs and
kill. It installs `@pollinations/cli` and, on the first click, Debian's
`openssh-client`, because `polli sandbox create` runs `ssh-keygen` and writes a
ProxyCommand that calls `polli sandbox proxy`. Sandboxes are billed to the
Pollinations account and pause themselves after a few minutes without an ssh
session; `timeout 0` keeps one running until it is paused or killed.

The same entry can install this menu **inside** a Polli sandbox
(`sandbox-cli polli-sandbox install-menu <id>`). It streams the CLI, the
`.desktop` entries, the category directories, the Xfce menu patch and the icons
over ssh, then runs `polli-sandbox-bootstrap.sh` there. A Polli VM is headless,
so without options the copy is useful from a terminal (`sandbox-cli list`,
`sandbox-cli <tool>`). Pass `--desktop`, or answer yes to the prompt, to also
install Xfce + TigerVNC, `g4f-go` and noVNC (`/usr/share/novnc`). The bootstrap
then:

- patches the Applications menu (including removing the stock Internet and
  Development menus);
- starts an Xvnc display bound to `127.0.0.1:5900`;
- starts `g4f-go api` on `127.0.0.1:8080` with `G4F_VNC_PORT=5900`,
  `G4F_NOVNC_DIR=/usr/share/novnc` and `G4F_DOMAIN_ROUTER=0` (the VM has no
  app hostnames in `/etc/hosts`);
- prints the two tunnels to open from your machine:

```bash
ssh -L 8080:127.0.0.1:8080 <sandbox-id>.polli   # http://127.0.0.1:8080/desktop/
ssh -L 5900:127.0.0.1:5900 <sandbox-id>.polli   # direct VNC client
```

Open `http://127.0.0.1:8080/desktop/` for the noVNC desktop (with the clipboard
bridge and `resize=off`), or point a VNC client at `127.0.0.1:5900` for the
raw session. The tunnels keep both ports off the sandbox's public interfaces,
and the open ssh connection also keeps the sandbox from pausing. The first
`g4f-go api` start downloads its CPython + g4f runtime; watch
`/tmp/g4f-sandbox-api.log` inside the VM if the page is not up yet. Set
`G4F_API_KEY` before running the bootstrap to put a password on the web
desktop (any username, password = key).

### Categories

Each `.desktop` file carries `Categories=X-G4F-Sandbox;X-G4F-<Cat>;` instead of
`Development;`, and one top-level menu per category includes it:


| Menu             | Category          | Entries                                        |
|------------------|-------------------|------------------------------------------------|
| **g4f**          | `X-G4F-Core`      | `remote-desktop`, `g4f-go`, `package`, `glow`, `docs`, `discover`, `settings`, `reset` |
| **Browsers**     | `X-G4F-Browser`   | `chrome`, `lightpanda`                         |
| **Files & Editors** | `X-G4F-Files`  | `files`, `nemo`, `code-server`, `kate`         |
| **Coding Agents**| `X-G4F-Agent`     | `opencode`, `openbot`, `thinkrail`, `orca`, `omp`, `pi`, `kilocode`, `antigravity`, `gemini-cli`, `qwencode`, `claude-code`, `codex`, `copilot`, `cline`, `amp`, `goose`, `aider`, `cursor`, `crush`, `continue-cli`, `grok`, `droid`, `openhands`, `plandex`, `hermes`, `omh`, `kiro`, `kimi`, `roomote`, `auggie`, `qoder`, `iflow`, `vibe`, `sweagent`, `dsh`, `paseo`, `t3code` |
| **Agent Skills** | `X-G4F-Skills`    | `ecc`, `superpowers`, `graphify`, `claude-mem` |
| **Gateways**     | `X-G4F-Gateway`   | `omniroute`, `litellm`, `langfuse`, `pi-freeflow`, `proxy`, `tunnel`      |
| **Local AI**     | `X-G4F-LocalAI`   | `ollama`, `llamacpp`, `lmstudio`, `jan` |
| **Chat UIs**     | `X-G4F-ChatUI`    | `librechat`, `lobechat`, `msty`, `sillytavern`, `open-webui`, `dsh`, `assistant-ui` |
| **Writing**      | `X-G4F-Writing`   | `humanizer`, `ai-detect`, `aidetect` |
| **Media**        | `X-G4F-Media`     | `polli`, `voicestudio`, `hyperframes`          |
| **Sandboxes**    | `X-G4F-Sandboxes` | `polli-sandbox`                                |
| **Assistants**   | `X-G4F-Assistant` | `cptr`, `dsh`, `openclaw`, `fabric`, `sgpt`, `gptme`, `llm`, `huggingface`, `deeptutor`, `nanobot`, `agent-reach` |

The menus are declared by patching the stock
`/etc/xdg/menus/xfce-applications.menu` at build time
(`patch-xfce-menu.py`), which inserts the twelve `<Menu>` blocks just before its
`<DefaultMergeDirs/>` node and removes the stock top-level **Internet** (XML
name `Network`) and **Development** menus. Shipping them as a merged menu file under
`/etc/xdg/menus/applications-merged/` does **not** work: garcon copies a merged
file's elements with `g_node_traverse(..., depth=2, G_IN_ORDER)`, which never
descends to the nested `<Menu>` nodes, so the menus are silently dropped and
every entry falls through to the catch-all **Other** menu. The patch script is
idempotent, so re-running it is a no-op.

The menu titles and icons come from the `.directory` files in
`desktop-directories/`, copied to `/usr/share/desktop-directories/`.

### Entry icons

Every entry names its own icon and hands the same name to the terminal, so the
menu row and the window it opens carry the same mark:

```
Exec=xfce4-terminal ... --title=Aider --icon=g4f-sandbox-aider --command="/usr/local/bin/sandbox-cli aider"
Icon=g4f-sandbox-aider
```

The two file managers are the exception: they are stock GUI apps without a
vendor mark of their own, so both use the theme's `folder` icon
(`Icon=folder`, `--icon=folder`) instead of a `g4f-sandbox-<tool>` download.

`fetch-icons.py` materializes those names in the hicolor theme at build time,
pulling each mark from the tool's own site — a direct logo asset where the
vendor publishes one, otherwise its favicon, otherwise its GitHub avatar. It
runs after the Pillow install because GTK icon themes load only `png`, `svg`
and `xpm`, so the JPEG sources (GitHub avatars, some favicons) are converted to
128x128 PNGs; SVGs are kept as SVG but rejected when they are not roughly
square, which is what keeps wordmarks such as aider.chat's 200x60 logo out.
`gtk-update-icon-cache` then refreshes the hicolor cache.

A tool whose sources are all unreachable falls back to the shared
`g4f-sandbox-cli.svg` instead of failing the build, and the build log lists
which ones did. `g4f-go`, `settings`, `reset`, `urls` and `hosts` are g4f's own
entries and always use that shared icon. `remote-desktop` is g4f's own too, but
it ships a dedicated icon: the build copies the app icon the web app itself
uses (`projects/remote-desktop/web/icon.svg`) to
`g4f-sandbox-remote-desktop.svg` in the hicolor theme, so it is not one of
`fetch-icons.py`'s vendor downloads.

The image also installs `librsvg2-common` explicitly: it is only a *Recommends*
of `xfdesktop4`/`adwaita-icon-theme`, and the build uses
`--no-install-recommends`. Without it the gdk-pixbuf SVG loader is missing and
GTK cannot render any SVG icon at all — neither the per-tool ones nor the
Adwaita symbolic icons the category `.directory` files name.

`chrome` is the odd one out: it installs Google's `.deb` through `apt` (the
package pulls its own runtime libraries) and then starts Chrome on the desktop
via `start-browser chrome --once`, detached from the launcher terminal (its
log goes to `/tmp/chrome.log`). `start-browser` (from `bin/start-browser`,
baked into the image) adds the CDP remote-debugging port (9223 on the network,
57011 locally) that the g4f automation connects to; `--once` starts Chrome a
single time instead of the script's default relaunch loop. Nothing about
Chrome is baked into the image, so a sandbox that never opens it carries none
of it.

`start-browser` passes `--no-sandbox` when it is asked to (`--no-sandbox`),
when it runs as root, or when it detects a container (`/.dockerenv`,
`/run/.containerenv` or `$container`). Chrome refuses to start its sandbox as
root, and Docker's default seccomp profile blocks the user namespaces its
zygote needs, so inside the sandbox container Chrome dies with
`Failed to move to new namespace ... Operation not permitted` even as the
unprivileged `pn` user. On a host that allows those namespaces, run
`start-browser chrome` outside the container to keep Chrome's sandbox.

`ensure_chrome` also links the binary into the user's bin so that *any* Chrome
launch inside the sandbox gets `--no-sandbox`, not just the ones going through
`start-browser`: it writes `~/.local/bin/google-chrome`, a two-line wrapper
that execs the real `/usr/bin/google-chrome-stable` with `--no-sandbox`, and
symlinks `google-chrome-stable` to it. `~/.local/bin` is first on `PATH` (the
image sets it, and `sandbox-cli` re-prepends it after applying a saved HOME),
so the shell, `start-browser` and g4f's own browser lookup (`shutil.which` in
`g4f/requests/cdp.py`) all resolve to the wrapper. The real binary is left
untouched, and the wrapper is regenerated on every `chrome` menu click, so it
follows a moved or reinstalled Chrome.

`lightpanda` is the headless counterpart and is likewise **not** baked into
the image: the entry runs `g4f-go browser install` on first click (downloading
and sha256-verifying the pinned Lightpanda release into `~/.g4f/browser/`) and
then `g4f-go browser serve`, which runs the CDP server in the foreground and
prints its endpoint. The terminal stays open as the server console; Ctrl-C
stops it. g4f picks the endpoint up automatically when it runs headless, so
the entry is only needed to pre-install the browser or to run the server
explicitly. `sandbox-cli install lightpanda` installs without serving.

The g4f-go menu entry opens a terminal with the bundled CLI ready to use. Run
`g4f-go help` to see commands, or pass a command directly to
`sandbox-cli g4f-go`, for example `sandbox-cli g4f-go status`.

`remote-desktop` runs the Remote Desktop server that g4f-go ships as a bundled
module (`g4f-go -m remote_desktop`). It serves a host page that shares this
sandbox's screen through the browser's native `getDisplayMedia` API, and a phone
pairs with it by scanning a QR code. On first click `ensure_remote_desktop`
installs the module's extra PyPI packages into the g4f-go runtime: `qrcode` (the
pairing QR SVG), `python-xlib` and `pynput` (pointer and keyboard injection).
`pynput` is installed with `--no-deps` because its Linux metadata pulls in
`evdev`, a C extension that needs kernel headers and a compiler the slim image
does not carry; the sandbox is X11, where `pynput` uses `python-xlib` instead.
Installing up front makes a failure visible in the terminal instead of
surfacing later as a server that has no QR code or no input control. The server runs in the
foreground, so the terminal stays open as its console (Ctrl-C stops it), and its
banner prints the host page (`http://127.0.0.1:8765/host`) plus the phone URL.
Open the host page on this desktop, press **Share**, then scan the QR code from
the phone; extra `remote_desktop` flags pass through, for example
`sandbox-cli remote-desktop --port 8765 --open-browser`.

## Using g4f with OpenCode, Copilot, Hermes, LibreChat and Open WebUI

The OpenCode, Copilot, Hermes, LibreChat and Open WebUI menu launchers
configure an OpenAI-compatible provider to use the g4f API at
`http://127.0.0.1:8080/v1` (or `:9090` when that endpoint is reachable).
OpenCode receives an inline config, Copilot receives its BYOK environment
variables, Hermes gets an alias (and, when a model is chosen, its `model.*`
settings) written through `hermes config set` (its config is YAML and the
image ships no YAML parser), LibreChat gets a custom endpoint in
`librechat.yaml`, and Open WebUI gets `OPENAI_API_BASE_URL` /
`OPENAI_API_KEY` plus its data directory under `~/.open-webui`. The Open WebUI
connection is a first-launch setting: after the admin account exists, change
it under **Settings → Admin → Connections** if the g4f endpoint moves.

OpenCode, Hermes and LibreChat always register the g4f provider with model
`auto`, so it is available in the model picker even before anything is chosen.
A model chosen through **Sandbox Settings** (or `G4F_SANDBOX_MODEL`) is
registered as well and becomes the tool's default; with no model chosen, the
tool keeps its own default model. Copilot is the exception: its BYOK variables
only take effect together with `COPILOT_MODEL`, so the g4f provider and the
chosen model are applied as a pair. OpenCode, Copilot and Hermes keep
credentials in the environment. LibreChat's `librechat.yaml` is generated on
every launch and embeds the endpoint and key, so change the model/provider in
**Sandbox Settings** rather than editing that file.

Set these environment variables before opening the menu item to override the
defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| `G4F_API_BASE_URL` | detected `http://127.0.0.1:{8080,9090}/v1`, otherwise `http://127.0.0.1:8080/v1` | g4f OpenAI-compatible API base URL |
| `G4F_API_KEY` | unset | API key sent as a bearer token and required by the server; saved in the Sandbox Settings editor (or with `sandbox-cli settings --api-key`) |
| `G4F_MODEL` | unset  | Model ID registered with the g4f-aware tools and used as their default when saved in **Sandbox Settings**; setting it only in the environment does not select it |
| `G4F_PROVIDER` | unset | g4f provider the launchers pin by prefixing a model chosen in **Sandbox Settings**; unset lets g4f pick from its own rotation |

The API service must be running and reachable from the desktop container.
`G4F_API_BASE_URL` should include the `/v1` path.

### OmniRoute gateway

When the local OmniRoute instance is running (default
`http://localhost:20128`), g4f exposes it as the `OmniRoute` provider and
forwards the `omniroute.g4f.local` hostname to it:

```bash
curl http://omniroute.g4f.local/v1/models
```

Requests are forwarded to the local instance only; set `OMNIROUTE_BASE_URL`,
`OMNIROUTE_HOST` or `OMNIROUTE_PORT` to point the provider and the hostname
elsewhere.

OmniRoute serves its dashboard and its own API from the gateway root rather
than from a sub-path, which is why it is reached by hostname instead of a path
prefix: the dashboard is at `http://omniroute.g4f.local/dashboard` and its
`/api/*`, `/_next/*` and `/healthz` paths are forwarded unchanged. g4f's own
API keeps its paths (`/v1/*`, `/api/*`) on `g4f.g4f.local`.

### Ollama gateway

The same applies to a local Ollama server (default `http://localhost:11434`).
g4f exposes it as the `Ollama` provider and forwards the `ollama.g4f.local`
hostname to it:

```bash
curl http://ollama.g4f.local/api/tags
curl http://ollama.g4f.local/v1/models
```

Set `OLLAMA_BASE_URL`, or `OLLAMA_HOST` / `OLLAMA_PORT`, to point the provider
and the hostname elsewhere. `OLLAMA_HOST` follows the Ollama convention and may
carry a scheme and port (`http://127.0.0.1:11434`); a bare host works too.

Both hostnames are served by the domain router (see below), which forwards to
the fixed local ports only, so gateway paths such as
`/v1/chat/completions` reach the gateway instead of being captured by g4f's
`/{mode}/{provider}/chat/completions` route.

The **Public URLs** entry (`sandbox-cli urls`) prints the hostname list, and
**Host Names** (`sandbox-cli hosts`) prints the `/etc/hosts` entries the host
needs for them. The noVNC desktop lives at `desktop.g4f.local` on the same
port.

The noVNC client bridges the **browser clipboard** to the VM: `Ctrl+V` in the
desktop pastes into the VM, and text copied in the VM is mirrored into the
browser clipboard where the Async Clipboard API is available (secure origins;
the clipboard panel stays as the fallback). It defaults to **`resize=off`**,
so opening the same session from a different device does not remotely resize
the Xvnc desktop; each viewer can still pick `off`/`scale`/`remote` in the
noVNC settings panel.

Set `G4F_API_KEY` on the container (or `bin/start-sandbox-desktop.sh
--api-key KEY`) to require a password for the API, the noVNC desktop and all
routed apps. The browser's login accepts **any username**; the password is the
key. `G4F_DOMAIN_SUFFIX` changes the `g4f.local` suffix and
`G4F_DOMAIN_ROUTER=0` disables hostname routing. See
[`docs/sandbox-desktop.md`](https://g4f.dev/docs/sandbox-desktop) for all options.

### Domain routing

Every app in the menu is reachable by hostname through the same g4f API port,
which is what the **Public URLs** entry (`sandbox-cli urls`) lists:

| Hostname | App | Port form |
|----------|-----|-----------|
| `g4f.g4f.local` | g4f API / GUI | `8080.g4f.local` |
| `omniroute.g4f.local` | OmniRoute (dashboard + API) | `20128.g4f.local` |
| `ollama.g4f.local` | Ollama | `11434.g4f.local` |
| `litellm.g4f.local` | LiteLLM gateway | `4000.g4f.local` |
| `librechat.g4f.local` | LibreChat | `3080.g4f.local` |
| `sillytavern.g4f.local` | SillyTavern | `8601.g4f.local` |
| `open-webui.g4f.local` | Open WebUI | `8603.g4f.local` |
| `cptr.g4f.local` | Open WebUI Computer | `8000.g4f.local` |
| `code-server.g4f.local` | code-server | `8602.g4f.local` |
| `llamacpp.g4f.local` | llama.cpp server | `11435.g4f.local` |
| `lmstudio.g4f.local` | LM Studio | `1234.g4f.local` |
| `remote-desktop.g4f.local` | Remote Desktop | `8765.g4f.local` |
| `desktop.g4f.local` | noVNC desktop | — |
| `pi-freeflow.g4f.local` | pi-freeflow | `28180.g4f.local` |

SillyTavern is installed with `listen: true` and its IP/host whitelists
disabled (with `securityOverride` for its listen-mode startup check), so the
`*.g4f.local` names above and a `sandbox-cli tunnel` URL work without further
configuration. Remove those settings in `~/.sillytavern/config.yaml` to keep it
on localhost only.

Its chat connection is preconfigured as **Custom (OpenAI-compatible)** at the
sandbox g4f endpoint (`G4F_API_BASE_URL`, normally
`http://127.0.0.1:8080/v1`) with the g4f model — `auto`, or the model chosen in
**Sandbox Settings** — and the sandbox API key when `G4F_API_KEY` is set. The
values are merged into `~/.sillytavern/data/default-user/settings.json` and
`secrets.json` on every launch; use the connection panel in the UI to change
them after that.

The hostname form exists because some apps cannot be served from a sub-path:
they emit absolute URLs, or their client derives its WebSocket URL from the
page origin (the noVNC client at `desktop.g4f.local` connects to
`ws://desktop.g4f.local/websockify`, which the router maps to the relay at
`/desktop/websockify`).

WebSocket connections are forwarded for every hostname in the table, not just
the ones with a path-prefix proxy: `ws://litellm.g4f.local/...` reaches the
same endpoint on the target app as the matching HTTP request would, so apps
that stream over a socket (LiteLLM, code-server, LibreChat, ...) work through
their hostname too.

Inside the container the names are seeded into `/etc/hosts` by the entrypoint,
derived from the router's own registry (`g4f/api/domain_router.py`), so they
cannot drift from the code. On the host, add the same names once — the
**Host Names** entry (`sandbox-cli hosts`) prints the full block, ready to
paste:

```bash
# Linux / macOS
sandbox-cli hosts | sed -n '/^127\.0\.0\.1/p' | sudo tee -a /etc/hosts
```

or use a wildcard resolver such as `dnsmasq` (`address=/g4f.local/127.0.0.1`).
Only the hostnames in the table are routed, and only to the fixed local ports
they name, so the router is not an open proxy. `<port>.g4f.local` accepts 4 to
8 digits and is resolved against the same table.

Two environment variables control it:

| Variable | Default | Purpose |
|----------|---------|---------|
| `G4F_DOMAIN_ROUTER` | `1` | set to `0` to disable host routing (and the `/etc/hosts` seeding) |
| `G4F_DOMAIN_SUFFIX` | `g4f.local` | suffix used for the hostnames, e.g. `g4f.test` |
| `G4F_DOMAIN_AUTOSTART` | `1` | set to `0` to stop apps from being installed and started on demand |

#### Install & run on demand

Opening a hostname for an app that was never installed does not dead-end in a
proxy error: the router starts the app for you. When a request reaches
`<app>.g4f.local` and nothing listens on the app's port, the g4f API runs
`sandbox-cli <app>` in a terminal window on the sandbox desktop — the same
install-and-run flow as the desktop menu entry — and answers the request with
`503` and a `Retry-After` header while the install runs:

```json
{"error": {"message": "LiteLLM is starting", "type": "litellm_starting",
           "detail": "\"sandbox-cli litellm\" is installing and starting LiteLLM on the desktop; retry in a few seconds."}}
```

The request is retried automatically once the app accepts connections within a
few seconds (so a fast start is invisible to the browser); otherwise reload the
page. The install keeps running either way. The window shows the app's output
and stays open as its console, so closing it stops the app (the same as
Ctrl-C in a menu entry). Without a desktop — no `DISPLAY` or no
`xfce4-terminal` — the CLI runs detached instead, with its output in
`/tmp/g4f-autostart-<app>.log`. Only one install per app runs at a time, and
only the apps in the table above that `sandbox-cli` knows how to install are
started — `g4f` (the API itself) and `desktop` (the noVNC client, part of the
image) are never started this way. Outside the sandbox image, where
`sandbox-cli` does not exist, the hook does nothing and the proxy keeps
answering its usual `502` "is not running".

The per-app target can be redirected with the same variables the providers use
(`OMNIROUTE_BASE_URL`, `OLLAMA_HOST`, `LITELLM_PORT`, ...).

#### Multicast discovery (mDNS)

**Multicast Discovery:** When a device (like a printer, a Mac, or a Linux box)
joins a network, it announces its name (e.g., `printer.local`) to the local
subnet using multicast traffic (`224.0.0.251` or `ff02::fb`).

The `*.g4f.local` names are ordinary unicast hostnames, not mDNS names: they
are resolved from `/etc/hosts` (or a wildcard resolver such as `dnsmasq`), so
they work without any multicast traffic and without a responder on the
network. That is deliberate — mDNS only reaches the local link, is blocked by
many container and VPN setups, and would let any device on the subnet claim a
name such as `g4f.local`. If you do want the sandbox to be discoverable by
name from other machines, publish an mDNS record for the host that runs the
g4f API (for example with `avahi-publish -a -R g4f.local <host-ip>` on Linux,
or Bonjour on macOS) and keep the `/etc/hosts` entries as the fallback.

### Proxy server (mitmproxy)

The **Install & Run Proxy Server** entry (`Gateways` menu) installs
[mitmproxy](https://mitmproxy.org) with `pip install --user` on first click and
then runs its console UI, listening on `127.0.0.1:8081`:

```bash
sandbox-cli proxy            # install on first run, then serve on 8081
sandbox-cli install proxy    # install only
```

The terminal stays open as mitmproxy's console (Ctrl-C stops it). The port is
`8081` rather than mitmproxy's default `8080` so it does not collide with the
g4f API. Point **Sandbox Settings → proxy** at `http://127.0.0.1:8081` to send
the menu entries (and g4f's own requests) through it; `sandbox-cli proxy`
also accepts extra mitmproxy flags after the tool id, for example
`sandbox-cli proxy --set console_eventlog_verbosity=debug`.

mitmproxy signs the HTTPS traffic it intercepts with its own CA, which nothing
trusts by default. The entry therefore pins mitmproxy's `confdir` to
`~/.config/g4f-sandbox/mitmproxy` (override with `G4F_SANDBOX_MITM_DIR`) and the
launchers trust the CA from there — see
[Proxy and certificate trust](#proxy-and-certificate-trust).

### Tunnel client (cloudflared)

The **Install & Run Tunnel Client** entry (`Gateways` menu) is the sandbox's
one-click tunnel: it installs Cloudflare's
[cloudflared](https://github.com/cloudflare/cloudflared) from its latest GitHub
release over HTTPS on first click, then opens a public, encrypted
`https://<random>.trycloudflare.com` URL for a local port — no account, DNS
record or open port required:

```bash
sandbox-cli tunnel          # install on first run, then tunnel to :8080
sandbox-cli install tunnel
```

The target defaults to the g4f API at `http://127.0.0.1:8080` (the entry point
the phone/GUI guide uses). Set `TUNNEL_TARGET`, or pass cloudflared flags
after the tool id, to expose something else:

```bash
TUNNEL_TARGET=http://127.0.0.1:3000 sandbox-cli tunnel
sandbox-cli tunnel --url http://127.0.0.1:3000
```

The terminal stays open as cloudflared's console and prints the public URL
(Ctrl-C closes the tunnel).

### Langfuse tracing

The **Install & Open Langfuse CLI** entry (`Gateways` menu) installs the
official [Langfuse CLI](https://github.com/langfuse/langfuse-cli) from npm on
first click and opens it in the terminal:

```bash
sandbox-cli langfuse              # install on first run, then open the CLI
sandbox-cli install langfuse      # install only
sandbox-cli langfuse api prompts list
```

The CLI talks to Langfuse Cloud by default. Point it at another instance with
the usual environment variables, which `sandbox-cli` passes through:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
sandbox-cli langfuse api traces list
```

The g4f API can also send its own traces to Langfuse. Install the optional
extra and set the same credentials before starting the server:

```bash
pip install "g4f[tracing]"        # installs the langfuse SDK
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
python -m g4f --port 8080
```

Every `/v1/chat/completions`, `/v1/responses` and `/v1/messages` request then
becomes one Langfuse generation carrying the model, provider, sampling
parameters, token usage and the assistant output. Streaming responses are
accumulated and reported once the stream ends. Tracing is inert unless both
credentials are present, so the server runs unchanged without Langfuse.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` | – | Required, enables tracing together with the secret key |
| `LANGFUSE_SECRET_KEY` | – | Required, enables tracing together with the public key |
| `LANGFUSE_HOST` / `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse instance to send traces to |
| `G4F_LANGFUSE_TRACING` | `1` | Set to `0` to keep the SDK installed but stop sending traces |
| `G4F_LANGFUSE_TAGS` | – | Comma-separated tags added to every trace |

### Choosing a g4f provider

The g4f-aware launchers always register the provider with model `auto`, so by
default g4f decides which of its providers answers. Set a provider in
**Sandbox Settings** (or `G4F_PROVIDER`) to pin one — useful when a specific
provider is the only one that works, or to compare answers:

```bash
sandbox-cli settings --provider PollinationsAI
sandbox-cli settings --provider ""            # back to g4f's own rotation
```

The provider name is the one g4f uses (`GET /v1/models` lists them, marked
with `"provider": true`). g4f resolves the provider from the **model name**
(`"<Provider>:<model>"`), so each launcher prefixes a model chosen in
**Sandbox Settings**:

- **OpenCode** — the model id becomes `g4f/<Provider>:<model>`; the always
  registered `auto` model stays `g4f/auto`.
- **Copilot** — `COPILOT_PROVIDER_MODEL_ID` and `COPILOT_MODEL` become
  `<Provider>:<model>`.
- **Hermes** — `model.default` becomes `<Provider>:<model>`; the `g4f` alias
  always points at the unprefixed `auto` model.
- **LibreChat** — the chosen model is listed first in the g4f endpoint, with
  `auto` kept after it.

With no provider set, the model is passed through unchanged and g4f picks from
its own rotation. An unknown provider name is not recognised as a prefix, so
g4f treats the whole string as a model name and the request fails with
`Request execution failed`.

The base URL stays at `…/v1` in every case, so model listing
(`GET /v1/models`) keeps working. The prefix is only added for a model chosen in
the menu; the always-registered `auto` model is never prefixed, and
`G4F_PROVIDER` on its own does not repoint a launcher. Note also that
`G4F_PROVIDER` is read only by the g4f **server** process, so exporting it in a
launcher does not select a provider for the request — the model name is what
does.

### System tray

The sandbox image runs `sandbox-tray` as its main process. It starts the g4f
API (`g4f-go api --port 8080 --debug`) as a child and puts a g4f icon in the
Xfce panel's stock **Status Tray** plugin through `yad` — the notification area
at the **top right** of the desktop. Right-click the logo to open the menu, so
the server can be managed from the desktop without a terminal:

| Tray item | What it does |
|-----------|--------------|
| **Open in Browser** | Opens `http://127.0.0.1:8080` on the desktop |
| **Start / Stop Server** | Starts or stops the API child process |
| **Restart Server** | Stops the child and starts it again, e.g. after changing settings |
| **Update g4f Package** | Runs `sandbox-cli package --update` in a terminal |
| **Change g4f Package Version...** | Prompts for a version (or `latest`) and pins it |
| **Sandbox Settings...** | Opens the interactive editor for model, API key, HOME, provider and proxy |
| **Show Logs** | Opens the API's `/logs` page |
| **Quit** | Stops the server and closes the tray |

The menu actions that need input open a terminal; package actions restart the
server when that window closes. Sandbox Settings does not need that: the tray
watches the settings file, so a saved key/model/provider/proxy is picked up by
the API server as well as by the launchers, and the server is restarted
automatically. The same applies to changes made with `sandbox-cli settings` or
to the file by hand.

The server is a child process on purpose: restarting an in-process uvicorn
instance is not reliable, and the container would otherwise have to be
recreated for every package or settings change. When there is no display
(`SANDBOX_DESKTOP=0`), `yad` is unavailable, or the tray exits unexpectedly,
`sandbox-tray` falls back to running the API directly.

### Sandbox Settings (model, API key, HOME, provider and proxy)

A desktop menu entry cannot export environment variables, so the **Sandbox
Settings** entry (`g4f` menu) writes the model, API key, HOME, provider and
proxy to a small settings file that the launchers read before they start:

```
~/.config/g4f-sandbox/env      # G4F_MODEL=…, G4F_API_KEY=…, SANDBOX_HOME=…, G4F_PROVIDER=…, SANDBOX_PROXY=…
```

The file lives outside `$HOME` on purpose: changing HOME must not move the
file that records it. `sandbox-cli settings` opens a full-screen editor
(`sandbox-settings`), validates every value before writing, and shows whether
each effective value comes from an `G4F_SANDBOX_*` environment override, the
saved file or the built-in default:

| Key | Action |
|-----|--------|
| `↑` / `↓` | Move between fields and actions |
| `Enter` | Edit the selected field / run the selected action |
| `s` | Save (writes atomically, mode `0600`) and restart the API |
| `t` | Test the saved key against the running API |
| `r` | Ask the tray to restart the API |
| `R` | Reset to defaults (delete the settings file) |
| `g` | Generate a new random API key |
| `v` | Reveal/hide the selected API key |
| `c` | Clear the selected field |
| `q` | Quit (asks before discarding unsaved changes) |

The editor only writes the file when something changed, preserves unknown
`KEY=value` lines, and never echoes the key while typing. Pressing Enter in the
key field keeps the current key; type a new one to replace it. After a save (or
a reset) the tray restarts the server, so the key the server requires and the
key the launchers send are never out of step.

The same command works non-interactively:

```bash
sandbox-cli settings --show                 # print the saved values (the key is masked)
sandbox-cli settings --json                 # saved + effective values as JSON (key masked)
sandbox-cli settings --check                # validate the file without changing it
sandbox-cli settings --generate-key         # print a new random key (does not save it)
sandbox-cli settings --restart-server       # ask the tray to restart the API
sandbox-cli settings --model gpt-4o         # set the model, keep the rest
sandbox-cli settings --api-key sk-...       # set the server/tool API key
sandbox-cli settings --api-key ""           # clear it
sandbox-cli settings --home /home/pn/work   # set HOME, keep the rest
sandbox-cli settings --provider PollinationsAI
sandbox-cli settings --proxy http://127.0.0.1:8081
sandbox-cli settings --unset                # back to the defaults
```

The API key is exported as `G4F_API_KEY` to the menu launchers (OpenCode,
Copilot, Hermes, LibreChat, assistant-ui) and passed to the server child, so
the same saved key authenticates both sides. The model, provider and proxy are
passed to the server too. With no key saved, the server starts without
authentication, as before; saving an empty key deliberately disables
authentication even if `G4F_API_KEY` is set in the ambient environment.

The **Reset Sandbox Settings** entry (`g4f` menu) is the same reset behind a
menu item: it asks for confirmation, then removes the settings file so every
launcher falls back to the ambient environment and the built-in defaults. It
touches nothing else — installed tools, agent configs and the g4f runtime stay
as they are. Non-interactively:

```bash
sandbox-cli reset            # asks before removing the file
sandbox-cli reset --yes      # no prompt
sandbox-cli reset --show     # print what would be cleared
```

`HOME` applies to **every** menu entry, not just the g4f-aware ones: agents
keep their config, credentials and caches under `$HOME`, so pointing it at a
scratch directory gives a throwaway agent profile. The directory is created if
it does not exist, and the launcher puts its `~/.local/bin`,
`~/.opencode/bin`, `~/.openclaw/bin`, `~/.kimi-code/bin` and `~/.qoder/bin`
back on `PATH`.

`PROXY` also applies to **every** menu entry. It is exported as `G4F_PROXY`
(which g4f itself reads for its own requests) and as the conventional
`HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` variables (which the agents and the
tools they shell out to honour, in both upper and lower case). Loopback
(`127.0.0.1`, `localhost`, `::1`) is appended to `NO_PROXY` so the launchers
still reach the local g4f API and the other local servers directly instead of
looping back through the proxy. The natural value is the **Proxy Server** entry
below, `http://127.0.0.1:8081`.

### Proxy and certificate trust

An intercepting proxy terminates TLS with its own CA, so a tool that is routed
through it fails every HTTPS request with `unable to verify the first
certificate` unless it is told to trust that CA. When a proxy is configured the
launchers therefore build a bundle from the system store plus mitmproxy's CA
(`~/.config/g4f-sandbox/mitmproxy/ca-bundle.pem`) and point the tools at it via
`NODE_EXTRA_CA_CERTS`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`,
`GIT_SSL_CAINFO` and `AWS_CA_BUNDLE`. The CA is also installed into the system
store (`/usr/local/share/ca-certificates/g4f-sandbox-proxy.crt` plus
`update-ca-certificates`) so tools that read the store directly — Chrome and the
other GUI apps — trust it as well. The bundle keeps the system CAs, so direct
(`NO_PROXY`) connections keep working.

The CA only exists once mitmproxy has run at least once; until then the
launchers skip the trust setup and start the tool anyway.

OpenCode needs one extra step. It performs its network traffic in a persistent
background server (`opencode serve --service`) that the CLI only talks to over
loopback, and that server is started once and reused by every later invocation
without inheriting the CLI's environment. Exporting the proxy for the CLI alone
would leave the server connecting directly. The launcher writes the proxy and CA
variables into OpenCode's own service configuration (`opencode service set env`)
and restarts the server when they change, so the process that makes the requests
is proxied too. With no proxy configured the variables are removed again and the
server restarted, so turning the setting off takes effect as well.

Precedence, highest first:

1. `G4F_SANDBOX_MODEL` / `G4F_SANDBOX_API_KEY` / `G4F_SANDBOX_HOME` /
   `G4F_SANDBOX_PROVIDER` / `G4F_SANDBOX_PROXY` — explicit environment override
2. the settings file — what the menu entry saved
3. `G4F_MODEL` / `G4F_API_KEY` / `$HOME` / `G4F_PROVIDER` / `G4F_PROXY` —
   ambient environment, then the built-in defaults

A model counts as chosen only at levels 1 or 2. If neither provides one, the
g4f-aware launchers still register the g4f provider with model `auto`, but
leave the tool's own default model alone. The provider is independent of that:
it is only passed on when a model is chosen.

### Finding new tools (AI Tool Scout)

**Find & Add AI Tools** (`g4f` menu) installs OpenCode if needed, registers the
g4f provider, then runs `sandbox-discover`. That script hands the OpenCode agent
a prompt that tells it to:

1. read the current menu with `sandbox-cli list`;
2. search the web (GitHub Trending, Show HN, or `curl` when web tools are
   unavailable) for up to three open-source AI CLI tools released or trending
   recently that the menu does not have yet;
3. wire each one in following the "Adding a tool" recipe — a `run_tool` case
   using the `ensure_*` helpers, a `list_tools` line, a
   `g4f-sandbox-<tool>.desktop` entry in an existing category, and (when the
   gpt4free checkout is present) the repo copies of the launcher, the desktop
   file, `fetch-icons.py` and the README tables;
4. validate with `bash -n` and `sandbox-cli list`, and write a short report.

The agent never installs or runs the tools it adds; the new entries do that on
first click. Run it by hand with `sandbox-cli discover`, add a topic bias with
`sandbox-cli discover audio`, or inspect the prompt with
`sandbox-cli discover --dry-run`. The prompt is kept in
`~/.cache/sandbox-discover/prompt.md` and the report in
`~/.cache/sandbox-discover/report.md`; `SANDBOX_DISCOVER_MODEL` picks another
OpenCode model (default `g4f/auto`).

### g4f docs in the browser and in glow

The **Open g4f Docs** entry (`docs`) opens
`https://g4f.dev/docs/` in Google Chrome on the desktop (Chrome is installed
on first open if it is not there yet). The browser gives every code block the
site's copy button and keeps the docs' normal navigation.

The **Open g4f Docs in Glow** entry (`glow`) is the terminal reader. It
installs `glow` and `chafa` with apt on first open, then downloads the docs
and opens them in glow's TUI. Glow renders every link as a real URL, so
`Ctrl+click` opens it in the browser, and `c` copies the open page — code
blocks included — to the clipboard (`?` lists the keys). Code can still be
copied the usual way by selecting it in the terminal; `--pager` opens with
`glow -p` instead for terminals where the TUI is not wanted.

The same renderer is available from a shell with `sandbox-docs open`:

1. it downloads `https://g4f.dev/docs/README.md` and every relative `.md` that
   page links (breadth first, depth 3, at most 120 pages) into
   `~/.cache/g4f-docs/raw`;
2. it rewrites each page into `~/.cache/g4f-docs/docs`, turning `<img>` and
   `![...]()` references into chafa ANSI art so pictures show up in the
   terminal instead of a URL (the art uses at most half the terminal width and
   never more than 50 columns; it follows the link when the image is wrapped
   in one, and otherwise links to the full image; shields.io badges are skipped,
   and images that cannot be fetched fall back to a plain link);
3. it starts glow on the rendered docs (glow 3 opens the index in its TUI;
   glow 2's TUI starts on the rendered file list with every page one Enter
   away; `--pager` uses `glow -p` on the index instead).

The cache is reused on later opens, so only new pages and images are fetched;
pages are downloaded again after 24 hours (`G4F_DOCS_TTL`, in seconds) or when
`--refresh` is passed. Images are rendered on the index only by default to keep
the first open quick; run `sandbox-docs render --all` (or `sandbox-docs open
--all`) to render every page. `G4F_DOCS_DIR` moves the cache.

## Files

- `bin/sandbox-cli` — installed as `/usr/local/bin/sandbox-cli` (the image
  copies the whole repo `bin/` into `/usr/local/bin`); also usable by
  hand: `sandbox-cli list`, `sandbox-cli install <tool>`, `sandbox-cli <tool>`,
  `sandbox-cli settings`, `sandbox-cli package`, `sandbox-cli reset`,
  `sandbox-cli urls`, `sandbox-cli hosts`.
- `bin/sandbox-tray` — installed as `/usr/local/bin/sandbox-tray`; the image's
  main process. It runs the g4f API as a child and shows the tray menu (server
  restart, package update/pin, Sandbox Settings). See **System tray** above.
- `bin/sandbox-settings` — installed as `/usr/local/bin/sandbox-settings`; the
  curses editor behind `sandbox-cli settings`. The line prompts remain as a
  fallback when curses or a terminal is unavailable.
- `g4f-sandbox-*.desktop` — the menu entries, copied to
  `/usr/share/applications/`. Each names its own `g4f-sandbox-<tool>` icon and
  passes it to `xfce4-terminal` as `--icon`.
- `g4f-sandbox-cli.svg` — the shared icon, used by the `g4f-go`, `package`,
  `settings`, `reset`, `urls` and `hosts` entries and as the fallback for any
  tool whose icon sources are unreachable.
- `bin/fetch-icons.py` — installed as `/usr/local/bin/fetch-icons.py`; fetches and
  normalizes the per-tool icons into the hicolor theme at build time. Run it by
  hand with `--dest <dir>` to inspect the result, or `--only <tool>` to retry a
  single tool.
- `desktop-directories/g4f-sandbox-*.directory` — the menu titles and icons,
  copied to `/usr/share/desktop-directories/`.
- `bin/patch-xfce-menu.py` — inserts the category menus into
  `/etc/xdg/menus/xfce-applications.menu` at build time.
- `bin/sandbox-docs` — installed as `/usr/local/bin/sandbox-docs`; the terminal
  renderer for https://g4f.dev/docs/ (downloads the markdown sources, rewrites
  their images into chafa ANSI art and opens the result in glow). The
  **Open g4f Docs** menu entry opens the site in Chrome; **Open g4f Docs in
  Glow** runs `sandbox-docs open`.
- `bin/sandbox-discover` — installed as `/usr/local/bin/sandbox-discover`; the
  "AI Tool Scout" behind the **Find & Add AI Tools** entry (see below).

## Fonts

The image installs `fonts-dejavu-core` plus `fonts-noto-core` and
`fonts-noto-color-emoji`, so CJK, Arabic, Devanagari and emoji render in
terminals, editors and browser pages instead of showing as boxes. Add more
Noto packages (for example `fonts-noto-cjk`) to the `apt-get install` list in
`docker/Dockerfile-sandbox-desktop` if a tool needs them.

## Adding a tool

1. Add a case to `bin/sandbox-cli` (`ensure_npm` / `ensure_pip` /
   `ensure_script` / `ensure_chrome` / `ensure_lightpanda` + `launch`) and a
   line to `list_tools`.
   The saved HOME is applied to every entry automatically; a tool that should
   also talk to the g4f API calls `configure_g4f` before `launch`.
2. Copy an existing `.desktop` file, swap the tool id, name, comment and
   keywords, and set `Categories=X-G4F-Sandbox;X-G4F-<Cat>;` to the category it
   belongs to. Set `Icon=g4f-sandbox-<tool>` and add the matching
   `--icon=g4f-sandbox-<tool>` to the `Exec` line.
3. Add the tool id to `ICONS` in `bin/fetch-icons.py` with its candidate URLs, best
   first (a direct logo asset, then `google("<domain>")`, then
   `github("<org>")`). A tool with no vendor site goes into `SHARED` instead and
   keeps `g4f-sandbox-cli`.
4. Rebuild the image.

## Adding a category

1. Add a `.directory` file in `desktop-directories/` (copy an existing one and
   swap `Name`, `Comment` and `Icon`).
2. Add the matching `<Menu>` block to `MENUS` in `bin/patch-xfce-menu.py`, using
   the new `X-G4F-<Cat>` category.
3. Rebuild the image.
