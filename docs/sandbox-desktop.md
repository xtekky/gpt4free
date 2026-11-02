# Sandbox desktop

The sandbox desktop is a full Xfce session (TigerVNC + noVNC) with the g4f API
and a one-click menu for coding agents, browsers, gateways,
local runtimes and assistants. Everything runs in one container and the API,
the web desktop and the routed apps share a single published port.

For the complete tool list and per-entry details, see
[`docker/sandbox-menu/README.md`](../docker/sandbox-menu/README.md). This page
covers the operational features: the start script, the app volume, user ids,
clipboard and scaling, password protection, domain routing, the tray, the
`sandbox-cli` commands and the Polli sandbox install.

## Quick start

```bash
# Build if needed, then run in the background
bin/start-sandbox-desktop.sh -d

# NoVNC desktop: http://127.0.0.1:8080/desktop/
# g4f API:       http://127.0.0.1:8080/
# Direct VNC:    127.0.0.1:5900
```

The script stores no state of its own. `docker compose` is used when available;
pass `--no-compose` to use a direct `docker run` instead.

## Sandbox launcher (`g4f-go sandbox`)

The `g4f-go` release binary includes the sandbox launcher. Download `g4f-go`
for Windows, Linux, macOS or Termux (amd64/arm64); the `sandbox` subcommand
requires only Docker, Podman or, on Termux/Android, udocker. It does not embed
the image, so the download is a few MB and the image is pulled on demand.

```bash
g4f-go sandbox                    # small API-only sandbox (latest-slim)
g4f-go sandbox --desktop          # full Xfce/VNC desktop, pulled on first use
g4f-go sandbox --desktop install chrome
g4f-go sandbox-cli list           # shorthand for `g4f-go sandbox --desktop cli`
g4f-go sandbox status
g4f-go sandbox down
```

- **API-only is the default.** It uses the multi-arch `latest-slim` image and
  maps the API to `http://127.0.0.1:8080/`.
- **The desktop image is opt-in.** `--desktop` pulls
  `latest-sandbox-desktop` (linux/amd64 or linux/arm64, selected by Docker) and
  serves the desktop at `/desktop/`, with direct VNC on `5900`.
- **Tools install on demand.** The image ships the menu and `sandbox-cli`, not
  the individual tools. `g4f-go sandbox --desktop install <tool>` installs one
  when it is first needed; `g4f-go sandbox --desktop cli list` lists them.
- **State is persistent** under `~/.g4f-sandbox` (`--data-dir` to change it):
  cookies and media in API mode, `.config`/`.local` in desktop mode.
- `--app-dir DIR` mounts a checkout at `/app/gpt4free`; `--port`,
  `--vnc-port`, `--bind`, `--api-key`, `--pull`, `--no-pull`, `--engine`,
  `--recreate` and `--no-open` mirror the start script where they overlap.
- **Termux/Android** is detected automatically. `g4f-go sandbox` uses
  [udocker](https://github.com/indigo-dc/udocker), the user-space container
  runner, so the same slim and desktop images run without root or a Docker
  daemon. udocker is installed with pip and set up with `udocker install` on
  first use. The older native Termux Python backend remains available with
  `--backend termux` (API only). `G4F_SANDBOX_UDOCKER_DIR` moves the udocker
  root (default `~/.udocker`) and `G4F_SANDBOX_UDOCKER_EXECMODE` selects the
  execution mode (`P1`, `P2`, `F1`-`F4`, `R1`-`R3`, `S1`).

### Hosted Polli sandbox

`g4f-go polli-sandbox --desktop` installs the Pollinations CLI locally, creates
a hosted Polli (E2B) sandbox, and installs `g4f-go`, `sandbox-cli`, Xfce,
TigerVNC and noVNC inside it. No Docker is needed on the host:

```bash
g4f-go polli-sandbox --desktop
g4f-go polli-sandbox install-menu <id> --desktop
g4f-go polli-sandbox connect <id>
```

The command prints the SSH tunnels for the browser desktop
(`http://127.0.0.1:8080/desktop/`) and a direct VNC client.

## `start-sandbox-desktop.sh`

```text
Usage: start-sandbox-desktop.sh [OPTIONS] [COMMAND]

Commands:
  up, start       Start the sandbox desktop container (default)
  stop, down      Stop the sandbox desktop container
  restart         Restart the sandbox desktop container
  status          Show the status of the sandbox container
  shell, exec     Open a bash shell inside the running container
  logs            View the container logs (-f to follow)
  build           Build or rebuild the sandbox desktop image
```

Useful options:

| Option | Meaning |
|--------|---------|
| `-d`, `--detach` | Run the container in the background |
| `-b`, `--build` | Build/rebuild the image before starting |
| `--pull` | Pull the prebuilt image instead of building |
| `-p`, `--port PORT` | API + noVNC port (default `8080`) |
| `--bind ADDR` | Host address to publish on (default `127.0.0.1`) |
| `--vnc-port PORT` | Direct VNC port (default `5900`) |
| `--user-id UID` | uid/gid of the sandbox user (default `1000`) |
| `--app-dir DIR` | Host directory mounted at `/app/gpt4free` |
| `--api-key KEY` | Require KEY as the browser/API password |
| `--domain-suffix SUFFIX` | Suffix for `<app>.<suffix>` routing (default `g4f.local`) |
| `--no-domain-router` | Disable hostname routing |
| `--local-g4f` | Install this checkout as an editable g4f (default: PyPI) |
| `--no-desktop` | Run the API only (`SANDBOX_DESKTOP=0`) |
| `--allow-local-network` | Let the sandbox reach private/link-local networks |
| `--image IMAGE`, `--name NAME` | Override the image or container name |

The same settings are read from the environment when the options are not used:
`G4F_APP_DIR`, `G4F_USER_ID`, `G4F_API_KEY`, `G4F_DOMAIN_SUFFIX`,
`G4F_DOMAIN_ROUTER`, `SANDBOX_BIND_ADDR`, `SANDBOX_BLOCK_LOCAL_NETWORK` and
`SANDBOX_FIREWALL_STRICT`.

## App volume

The g4f app tree is bind-mounted at `/app/gpt4free`, the path the image's
editable g4f install points at. The API serves that code and the agents edit
the same files.

```bash
# Default: the checkout the script lives in
bin/start-sandbox-desktop.sh -d

# A different checkout or working copy
bin/start-sandbox-desktop.sh -d --app-dir /srv/gpt4free

# Equivalent environment override
G4F_APP_DIR=/srv/gpt4free bin/start-sandbox-desktop.sh -d
```

Mount at `/app/gpt4free`, never `/app`: the image keeps its embedded Python and
the g4f runtime under `/app/g4f`, and mounting over `/app` hides them. The
compose file uses `${G4F_APP_DIR:-..}` for the same mount.

## Image user id

The container runs as the unprivileged `pn` user with uid/gid **1000** by
default. Matching the host user keeps bind-mounted files (`/app/gpt4free`,
`sandbox-home`) owned by your account instead of root or an unrelated uid.

```bash
# Build (or rebuild) the image for the current host user
bin/start-sandbox-desktop.sh build --user-id "$(id -u)"

# Run with the matching id
bin/start-sandbox-desktop.sh up --user-id "$(id -u)"
```

The start script inspects the image's `G4F_USER_ID` environment and rebuilds
when it differs from the requested id. Pulled prebuilt images always use
uid 1000, so `--pull` with a different id prints a warning.

## Clipboard and scaling (noVNC)

The web desktop is the stock noVNC client, served by the g4f API at
`/desktop/`, with g4f branding applied at image build.

**Clipboard.** `bin/novnc-brand.sh` patches `app/ui.js` to bridge the browser
clipboard and the VM clipboard:

- Pasting with `Ctrl+V` (or the context menu) inside the noVNC window sends the
  text to the VM.
- Text copied inside the VM is mirrored into the browser clipboard.
- The noVNC clipboard panel (the clipboard button in the control bar) remains
  available, and is the fallback when the browser blocks the Async Clipboard
  API. Browsers only grant that API on secure origins: `http://127.0.0.1` and
  `https://` work, a plain `http://` LAN address does not.

**Scaling.** The client defaults to `resize=remote`: noVNC asks the VNC server
to resize the desktop to the viewer's window, so the same session fills a
phone, laptop or tablet without manual zooming. The overlay's **Fit** button
switches a viewer between local scaling (`scale`) and remote resize (`remote`);
the stock noVNC settings panel also offers `off`/`scale`/`remote` for that
viewer. To use a different default, edit the `defaults.json` written by
`novnc-brand.sh` or pass `resize=<mode>` in the `vnc.html` URL. The direct VNC
port (`5900`) is unaffected.

**Touch controls.** The `/desktop/` landing page also overlays the input
controls from the Remote Desktop viewer, because the stock noVNC control bar
is awkward on a phone:

- **Type** opens a text field that sends the whole string to the VM, plus
  Enter, Backspace, Tab, Esc, arrow, navigation and F5/F11 keys.
- **Ctrl**, **Alt**, **Shift** and **Win** are sticky modifier buttons; tap one,
  then a key or a shortcut, and tap it again to release it.
- **Keys** lists one-tap shortcuts (Alt+Tab, Ctrl+C/V/X/Z, Win+R, Alt+F4,
  PrintScreen, and the rest of the Remote Desktop set), pressing the modifiers
  around the key and releasing them in reverse.
- **View/Control** toggles noVNC's view-only mode, **med** cycles the JPEG
  quality, **Fit** toggles local scaling against the default remote resize,
  **Full** enters fullscreen and **Leave** disconnects. Leave also breaks a
  connect/reconnect loop, so it works while the desktop is unreachable and not
  only during a live session. The bar hides after a few idle seconds; tap the
  arrow at the bottom to bring it back.
- The Type/Keys panels follow the on-screen keyboard (`visualViewport`) so
  Send and the key rows stay above it, and closing Type releases the keyboard
  instead of covering the next panel.

The overlay is served from `docker/novnc-controls.{js,css}` and talks to
noVNC through `window.g4fUI`, exposed by `bin/novnc-brand.sh`; the stock
client is otherwise unchanged.

## Install the desktop as an app (PWA)

`/desktop/` is an installable progressive web app. The page links a web app
manifest (`/desktop/manifest.webmanifest`), registers a service worker
(`/desktop/sw.js`) and uses the g4f SVG as its icon. Chrome, Edge, Brave and
other Chromium browsers offer **Install app** in the address bar or browser
menu. When the browser fires `beforeinstallprompt`, the touch toolbar also
shows an **Install** button; on iPhone and iPad use **Share → Add to Home
Screen** instead, because Safari does not support the install prompt.

The installed app opens at `/desktop/` in a standalone window, with the same
noVNC client and touch controls as the browser tab. It also works through the
`desktop.g4f.local` host route: the manifest, service worker and icons use
relative URLs, so the host rewrite does not change their scope.

The service worker caches only the page shell (the landing page, controls,
manifest and icons) so an installed app can start while the network is down.
The VNC stream is live: `/desktop/websockify` and every WebSocket are excluded
from the cache, and reconnecting still needs the sandbox to be reachable. The
browser's **Install** affordance requires a secure context, so use
`http://127.0.0.1`/`http://localhost` or HTTPS; `http://` on a plain LAN
address cannot install the app.

## Password protection (`G4F_API_KEY`)

Set `G4F_API_KEY` to protect the API, the noVNC desktop and every routed app
with HTTP Basic authentication:

```bash
bin/start-sandbox-desktop.sh -d --api-key 'choose-a-strong-key'

# or
G4F_API_KEY='choose-a-strong-key' bin/start-sandbox-desktop.sh -d
```

When the browser asks for credentials, **the username can be anything**; the
password is the value of `G4F_API_KEY`. The key is also accepted through the
`g4f-api-key` header and the `Authorization: Bearer`/Basic headers used by the
API clients.

Inside the container, the same setting can be saved from the desktop:
**Sandbox Settings** → `sandbox-cli settings --api-key` (this writes
`~/.config/g4f-sandbox/env`, used by both the tray-started API and the menu
launchers). `sandbox-cli settings --generate-key` prints a random key, and
`sandbox-cli settings --api-key ""` clears it. The tray watches the file and
restarts the server when the key changes.

> The key travels over the published port. Bind to `127.0.0.1` (the default)
> or put the container behind TLS if the port is reachable from other hosts.

## Domain suffix and routing

Apps in the menu are reachable by hostname through the same API port:

| Hostname | Destination |
|----------|-------------|
| `g4f.g4f.local` | g4f API / GUI |
| `<port>.g4f.local` | the app listening on that local port, e.g. `20128.g4f.local` |
| `desktop.g4f.local` | the noVNC desktop |
| `opencode.g4f.local` | the headless OpenCode server API (`opencode serve`) |
| `omniroute.g4f.local`, `litellm.g4f.local`, `ollama.g4f.local`, … | the matching local app |

`bin/sandbox-entrypoint.sh` seeds `/etc/hosts` inside the container from
`g4f/api/domain_router.py`, so no extra `extra_hosts` are needed. On the host,
add the names to `/etc/hosts` or use the output of `sandbox-cli hosts`:

```text
127.0.0.1 g4f.g4f.local desktop.g4f.local 8080.g4f.local
```

**Change the suffix** when `g4f.local` collides with something else:

```bash
bin/start-sandbox-desktop.sh -d --domain-suffix sandbox.test
# or
G4F_DOMAIN_SUFFIX=sandbox.test bin/start-sandbox-desktop.sh -d
```

**Disable routing** entirely when you only use the published port:

```bash
bin/start-sandbox-desktop.sh -d --no-domain-router   # G4F_DOMAIN_ROUTER=0
```

The router only forwards the known hostnames to their fixed local ports, so it
cannot be used as an open proxy. Environment overrides are
`G4F_DOMAIN_SUFFIX` and `G4F_DOMAIN_ROUTER` (both are also passed through the
compose file).

## Tray

The container's main process is `sandbox-tray`. It starts the g4f API as a
child and places the **g4f logo in the panel's notification area (top right)**
with `yad`. Right-click the logo for:

- Open in Browser
- Start/Stop Server
- **Restart Server**
- Update g4f Package
- Change g4f Package Version…
- Sandbox Settings…
- Show Logs
- Quit

The tray watches the sandbox settings file, so a key, model, provider or proxy
change restarts the server automatically. Without a display, without `yad`, or
when the tray cannot start, it falls back to running the API directly.

## `sandbox-cli`

`sandbox-cli` is the installer behind the menu entries and also the terminal
interface for the sandbox:

```text
sandbox-cli <tool>          install <tool> if missing, then run it here
sandbox-cli install <tool>  install only
sandbox-cli list            list the known tools
sandbox-cli settings        model, API key, HOME, provider and proxy
sandbox-cli reset           clear the saved settings
sandbox-cli package         show, update or pin the g4f package
sandbox-cli update          update the g4f-go launcher itself
sandbox-cli upgrade         update the g4f package in the runtime
sandbox-cli uninstall       remove g4f-go and its managed data
sandbox-cli urls            print the public URLs served by the API
sandbox-cli hosts           print the /etc/hosts entries for the app domains
sandbox-cli harness         pick a coding agent and a model from the g4f API
sandbox-cli polli-sandbox   create/connect/manage hosted Polli sandboxes
```

Tools install on first click (vendor installer, npm package,
`pip install --user`, or Chrome's `.deb`), so the image stays small. The
settings commands accept flags for scripting, for example
`sandbox-cli settings --api-key 'key' --model gpt-4o`,
`sandbox-cli settings --show` and `sandbox-cli settings --json`.

The launcher and its Python runtime are owned by `g4f-go`, so the management
commands forward to it: `sandbox-cli update` updates the `g4f-go` binary
(`--check` reports without installing, `--version <tag>` picks a release),
`sandbox-cli upgrade` updates the g4f package in the runtime, and
`sandbox-cli uninstall` removes the launcher and its managed runtime, browser
and caches (`--keep-binary`, `--keep-data` and `--dry-run` are passed through).

## Polli sandbox install

A Polli sandbox is a headless Debian VM (E2B) reached as `<id>.polli` over
ssh. The menu can install itself there:

```bash
# Menu only (terminal use)
sandbox-cli polli-sandbox install-menu <id>

# Menu + Xfce + TigerVNC + g4f-go + noVNC
sandbox-cli polli-sandbox install-menu <id> --desktop
```

`install-menu` copies the menu assets (the CLI, `.desktop` entries, desktop
directories, Xfce menu patch and icons) into the sandbox. With `--desktop` it
also copies `g4f-go` and `/usr/share/novnc`, installs the Xfce/TigerVNC
packages there, patches the Applications menu, starts the desktop on
`127.0.0.1:5900`, and starts `g4f-go api` on `127.0.0.1:8080` with
`G4F_VNC_PORT=5900`, `G4F_NOVNC_DIR=/usr/share/novnc` and
`G4F_DOMAIN_ROUTER=0` (the sandbox has no app hostnames in `/etc/hosts`).

The bootstrap prints the tunnels to open from your machine:

```bash
ssh -L 8080:127.0.0.1:8080 <id>.polli   # http://127.0.0.1:8080/desktop/
ssh -L 5900:127.0.0.1:5900 <id>.polli   # direct VNC client
```

The first `g4f-go api` start downloads its CPython + g4f runtime into the
sandbox user's home; watch `/tmp/g4f-sandbox-api.log` inside the sandbox if
the page is not up yet. The applications menu carries the g4f categories
(Media > Polli, Coding Agents, …).

## Applications menu

The image patches `/etc/xdg/menus/xfce-applications.menu` with
`bin/patch-xfce-menu.py`, which:

- inserts the g4f categories (g4f, Browsers, Files & Editors, Coding Agents,
  Gateways, Chat UIs, Assistants, …) as their own top-level menus;
- removes the stock **Internet** (XML name `Network`) and **Development**
  menus, which the sandbox entries replace.

The patch is idempotent: re-running it does not duplicate menus. The sandbox
entries use `X-G4F-*` categories, so they never fall into Internet or
Development.

## Notes

- The image starts an egress firewall. It needs `--cap-add NET_ADMIN`; without
  it the container still starts with a warning. Set
  `SANDBOX_FIREWALL_STRICT=1` to make a firewall failure fatal. The filter
  blocks private/link-local/reserved networks, file-sharing protocols and
  crypto-mining ports while allowing public internet access and replies to
  host-initiated connections.
- Keep `~/.config` and `~/.local` bind-mounted (see the README example) to
  preserve agent logins, caches and sandbox settings across container
  restarts. Do not mount over `/home/pn` itself: the embedded runtime lives in
  `/home/pn/.g4f`.
- Logs: `bin/start-sandbox-desktop.sh logs -f`, or inside the container
  `/tmp/g4f-sandbox-*.log` for the desktop processes and the API.
