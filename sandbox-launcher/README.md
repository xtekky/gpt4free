# g4f-sandbox

A small, standalone launcher for the [g4f sandbox](../docs/sandbox-desktop.md)
container image. It is a single executable for Windows, Linux and macOS (amd64
and arm64) and requires only Docker or Podman (Docker Desktop on Windows and
macOS). When neither is installed it installs the Podman CLI with the host
package manager (apt, dnf, pacman, zypper, apk, Homebrew, winget, Chocolatey or
Scoop); set `G4F_SANDBOX_INSTALL_PODMAN=0` to opt out. On **Termux/Android** it
detects the environment and runs the API natively with Termux's Python — no
Docker required.

The executable does **not** embed the image. On first run it pulls only the
image for the selected mode and the host architecture:

| Mode | Runtime | Approx. download | Use |
| --- | --- | --- | --- |
| API only (default) | Docker/Podman + `hlohaus789/g4f:latest-slim` | ~400 MB image | API, GUI, provider work |
| Desktop (`--desktop`) | Docker/Podman + `hlohaus789/g4f:latest-sandbox-desktop` | ~675 MB image | Screen, coding agents, one-click tools |
| Termux (auto) | Termux Python + pip `g4f[slim]` | Python + g4f wheels | API on Android, no Docker |

Both tags are multi-arch (`linux/amd64` and `linux/arm64`; arm64 is slightly
smaller). Docker selects the matching manifest and downloads only those
layers. The launcher binary itself is under 1 MB compressed, so a first
API-only run downloads roughly half of the desktop image.

API-only mode runs the regular slim server image, so it does not include the
sandbox entrypoint, egress firewall or desktop stack. Use `--desktop` when you
want the isolated sandbox with the menu and tools.

## Quick start

```bash
# Small, API-only sandbox on http://127.0.0.1:8080/
./g4f-sandbox

# Full desktop, pulled only when it is needed
./g4f-sandbox --desktop

# Install a sandbox-menu tool on demand (nothing is baked into the image)
./g4f-sandbox --desktop install chrome

# Run sandbox-cli itself
./g4f-sandbox --desktop cli list
```

On Windows, run `g4f-sandbox.exe` from Explorer or a terminal. The launcher
opens the web UI in the default browser unless `--no-open` is set.

### Termux / Android

Download the `android-arm64` zip (a static Linux/arm64 binary), extract it in
Termux, and run it. Termux is detected automatically; Docker is not needed.

```bash
pkg install python          # optional: the launcher runs pkg for you
./g4f-sandbox               # installs g4f[slim] on first run, starts the API
./g4f-sandbox status
./g4f-sandbox logs
./g4f-sandbox install pillow
./g4f-sandbox down
```

- The Termux backend is **API-only**. `--desktop` still requires Docker/Podman
  on a machine that can run the desktop image.
- First run installs `g4f[slim]` with pip into Termux's Python. Later runs start
  in a second or two.
- The server runs detached by default and logs to
  `~/.g4f-sandbox/g4f.log`; `-f`/`--foreground` runs it in the terminal.
- Use `termux-wake-lock` before a long run so Android does not suspend Termux.
- `--engine docker` (or `--backend docker`) overrides autodetection if you want
  to drive a remote Docker daemon from Termux.

## Commands

```text
g4f-sandbox [COMMAND] [OPTIONS]

up, start       Start the sandbox (default)
down, stop      Stop and remove the container (--keep only stops it)
restart         Restart the container
status          Show status, image and URL
logs            Follow the container logs
shell           Open a bash shell in the container
exec <cmd...>   Run a command in the container
cli <args...>   Run sandbox-cli in a desktop container
install <tool>  Install a sandbox-menu tool on demand (desktop mode)
open            Open the web UI in a browser
pull            Pull the image for the selected mode
build           Build the desktop image from a source checkout
version         Print the launcher version
help            Show all options
```

Useful options: `--port`, `--vnc-port`, `--bind`, `--data-dir`, `--app-dir`,
`--api-key`, `--no-open`, `--recreate`, `--pull`, `--no-pull`, `--engine`.

The environment variables are the same ones the shell start script uses
(`PORT`, `VNC_PORT`, `SANDBOX_BIND_ADDR`, `G4F_API_KEY`, `G4F_APP_DIR`,
`G4F_DOMAIN_SUFFIX`, `G4F_DOMAIN_ROUTER`, `SANDBOX_BLOCK_LOCAL_NETWORK`,
`SANDBOX_FIREWALL_STRICT`) plus `G4F_SANDBOX_MODE`, `G4F_SANDBOX_IMAGE`,
`G4F_SANDBOX_NAME`, `G4F_SANDBOX_DATA_DIR`, `G4F_SANDBOX_ENGINE` and
`G4F_SANDBOX_NO_OPEN`.

## Persistence

State lives in `${G4F_SANDBOX_DATA_DIR:-~/.g4f-sandbox}`:

- API mode: `har_and_cookies/` and `generated_media/`.
- Desktop mode: `config/` and `local/` (mounted at `/home/pn/.config` and
  `/home/pn/.local`), plus the optional `--app-dir` checkout mounted at
  `/app/gpt4free`.

Do not mount over `/home/pn` itself in desktop mode: the embedded Python and
g4f runtime live in `/home/pn/.g4f`.

On Linux, the container user is uid/gid 1000 `pn` (desktop) or `g4f` (API). If
your host user has a different id, make the data directory writable by 1000:

```bash
sudo chown -R 1000:1000 ~/.g4f-sandbox
```

For a desktop image built for your own id, use
`g4f-sandbox --desktop --build --user-id "$(id -u)"` and pass the matching
`--image` tag. Docker Desktop on Windows and macOS handles ownership through
its file-sharing layer.

## Building

```bash
go test ./...
./build-all.sh              # all targets into dist/
./build-all.sh windows      # only windows
./build-all.sh android      # Termux (static linux/arm64)
G4F_VERSION=1.2.3 ./build-all.sh
```

Releases are built by `.github/workflows/build-packages.yml` and attached to
GitHub releases as `g4f-sandbox-<version>-<os>-<arch>.zip`.
