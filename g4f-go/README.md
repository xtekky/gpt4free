# g4f-go

A small, self-contained Go launcher for [gpt4free](https://github.com/xtekky/gpt4free).
Unlike the original design, the CPython runtime is **not embedded** in the
binary: `g4f-go` is a few MB, and downloads the correct CPython for the host
platform on first run (with live progress feedback), then installs `g4f` into
it. No system Python required.

```
g4f-go client "What is gpt4free?"
g4f-go api --port 8080
g4f-go -m remote_desktop --port 8000
```

## How it works

1. The binary embeds only a small manifest (`runtime.json`) that pins the
   CPython archive URL, size, and sha256 per platform
   (pbs install-only tarballs for desktop, the official python.org Android
   package for Termux).
2. On first run `g4f-go` picks the entry for its host OS/arch, downloads the
   archive into `~/.g4f/python-embed/` (or the app dir on Android), printing a
   `\r` progress line (percent, bytes, throughput, ETA), verifies the sha256,
   and extracts it.
3. It then bootstraps pip (`ensurepip`), installs `g4f[slim]` into the
   interpreter, writes a `.installed` stamp, and finally runs your command.
4. Subsequent runs skip straight to step 3 — the runtime is cached until the
   pinned version changes.
5. The embedded Python bundle (see below) is extracted into the runtime on
   first use and refreshed whenever its revision changes.

Downloads go to:

| Platform | Location |
|---|---|
| Linux / macOS / Windows | `~/.g4f/python-embed/` |
| Android (Termux) | app-private dir (`G4F_ANDROID_FILES_DIR`, defaults to `$HOME/g4f-go-runtime`) |

Pip downloads used by the launcher (including bundled module dependencies) are
cached in `~/.g4f/cache/pip` (`G4F_CACHE_DIR/pip` when overridden). To remove
them, delete that directory or run `g4f-go cache clear`.

`G4F_PYTHON_ONLY=1 g4f-go --version` prints the downloaded interpreter path
without running gpt4free (useful for wrapping the runtime from other tools).

On Windows, if `python-home/python.exe` is missing, check whether antivirus
quarantined it and restore it only if you trust the downloaded runtime. Retry
`g4f-go` to extract the verified cached archive again.

Alternatively, `uv tool install "g4f[all]"` installs the `g4f` CLI separately;
run `g4f api` or `g4f gui` directly. This does not replace the isolated runtime
used by `g4f-go`.

## Installing on Termux

The installer detects Android/Termux and installs the `android-arm64` release
into `$PREFIX/bin` (`/data/data/com.termux/files/usr/bin`), which Termux
already puts on `PATH`:

```
pkg install curl unzip
curl -fsSL https://raw.githubusercontent.com/xtekky/gpt4free/main/bin/install.sh | bash
g4f-go sandbox
```

Pass `--prefix DIR` (or `G4F_GO_PREFIX`) to install somewhere else. On Termux
the sandbox launcher uses udocker automatically; it is installed with pip on
first use (run `pkg install python` first).

## Building

```
go build -o g4f-go .        # linux host build (fast iteration)
./build-all.sh              # cross-compile + zip releases for all targets
./build-all.sh android      # only the android target
./sync-bundle.sh            # refresh g4f-go/bundle/ from projects/
./fetch-python.sh           # optional: re-pin sizes + sha256 in runtime.json
```

The manifest is embedded via `go:embed runtime.json` and the Python bundle via
`go:embed all:bundle`; the binary builds without network access.

## Usage

```
g4f-go <g4f args...>        run gpt4free (e.g. g4f-go client "hello")
g4f-go -m <module> [args...] run a Python module (bundled or installed)
g4f-go script.py [args...]  run a .py file with the bundled Python
g4f-go api --port 8080      start the OpenAI-compatible API server
g4f-go gui                  launch the web GUI
g4f-go status               show runtime download/install status
g4f-go browser install      install the headless browser (Lightpanda) for this OS
g4f-go browser serve        run the browser's CDP server in the foreground
g4f-go sandbox [command]    run the g4f sandbox container (embedded launcher)
g4f-go start-desktop-sandbox [--app TOOL] [options]  start the desktop sandbox
g4f-go sandbox-cli [args]   run sandbox-cli in the desktop container
g4f-go polli-sandbox        create a Polli cloud sandbox with g4f-go + sandbox-cli
g4f-go freeflow             run the pi-freeflow proxy without Pi/OMP
g4f-go freeflow status      show the standalone freeflow endpoint and API key
g4f-go turn serve           run the embedded STUN/TURN server in the foreground
g4f-go turn status          show the configured/running STUN/TURN server
g4f-go turn credentials     mint TURN REST credentials for a client
g4f-go install g4f          (re)install the g4f package (network)
g4f-go install              install the g4f-go launcher into a bin on PATH
g4f-go install --prefix DIR install the launcher into DIR
g4f-go upgrade              upgrade the g4f package to the latest release
g4f-go update               update the g4f-go launcher itself
g4f-go uninstall            remove g4f-go, its runtime, browser and caches
g4f-go -v <version> <cmd>   pin the g4f package version (e.g. g4f-go -v 1.5.0 api)
g4f-go -v latest            stop pinning (track the latest g4f release)
g4f-go help                 show help
```

## Pinning the g4f version (`-v`)

By default `g4f-go` installs the latest `g4f` release on first run and
auto-upgrades it for `api`/`gui`. `-v <version>` pins it to an exact release
instead:

```
g4f-go -v 1.5.0 client "hello"   run a command with g4f 1.5.0
g4f-go -v 1.5.0 api --port 8080  the pin works in front of any subcommand
g4f-go -v 1.5.0                  just (re)install the pinned version
g4f-go -v latest                 drop the pin again
```

The pin is persisted in
`~/.g4f/python-embed/.g4f-runtime/.g4f-version`, so later runs without `-v`
keep using that version — including the automatic upgrades that `api`/`gui`
would otherwise trigger. Passing `-v` again replaces the pin (a changed pin
reinstalls g4f on the next run), `-v latest` clears it, and `g4f-go status`
shows the active pin. A bare `-v` (no version) still prints the g4f-go
version, as does `--version`.

An explicit `g4f-go install g4f` (unpinned) or `g4f-go install g4f==X.Y.Z`
updates the stored pin to match what was installed.

Installing a local checkout (`g4f-go install -e /path/to/gpt4free`) marks the
runtime as **builtin**. The launcher then serves that package as-is: the
`api`/`gui` auto-upgrade never replaces it with PyPI's latest. This is what the
sandbox desktop image uses, so its first `g4f-go api` call does not download or
upgrade g4f. An explicit `g4f-go install g4f`, `g4f-go upgrade` or `g4f-go
bootstrap` opts back into PyPI and clears the marker.

## Installing the launcher on PATH (`install`)

`g4f-go install` copies the running launcher into a bin directory that is
already on `PATH`, so a downloaded or locally built binary can be put on
`PATH` without re-downloading a release:

```
g4f-go install                     # install into a writable bin directory on PATH
g4f-go install --prefix ~/bin      # install into a specific directory
g4f-go install --dry-run           # show the destination without writing
```

The default destination is `$G4F_GO_PREFIX` when set, otherwise `$PREFIX/bin`
on Termux, `~/.local/bin` when it is on `PATH`, or the first writable
directory on `PATH`. The binary is copied atomically and keeps its executable
mode; the downloaded CPython runtime and the installed g4f package are not
touched. `g4f-go install g4f` remains the pip passthrough for the Python
package.

## Updating and uninstalling

`g4f-go update` is for the launcher, `g4f-go upgrade` is for the Python
package, and `g4f-go uninstall` removes both:

```
g4f-go update                  install the latest g4f-go GitHub release
g4f-go update --check          only report whether a newer launcher exists
g4f-go update --force          reinstall even when already up to date
g4f-go update --version TAG    install a specific release (e.g. v0.2.0)

g4f-go upgrade                 upgrade g4f to the latest release and clear the pin
g4f-go -v 1.5.0 upgrade        install exactly g4f 1.5.0 and keep that pin

g4f-go uninstall               remove the binary, runtime, browser and caches
g4f-go uninstall --keep-binary keep the executable, remove only managed data
g4f-go uninstall --keep-data   keep the runtime/data, remove only the executable
g4f-go uninstall --dry-run     show what would be removed
```

`update` replaces the executable in place; the downloaded CPython runtime and
the installed `g4f` package are left untouched. It selects the
`g4f-go-<version>-<os>-<arch>.zip` asset for the current platform and verifies
the published sha256 digest when GitHub provides one. A copy installed through
a package manager may need that manager's update command instead.

`uninstall` removes `python-embed` (which contains the g4f package and the
extracted bundle), the browser directory, and the g4f cache. User
configuration such as cookies and model settings in the g4f config directory is
left in place, and `g4f-go uninstall g4f` still works as a pip passthrough for
removing just the Python package.

## Sandbox container (`sandbox`, `sandbox-cli`)

`g4f-go` embeds the sandbox launcher, so one binary can also run the
sandbox desktop/API container:

```
g4f-go sandbox                    # API-only sandbox on http://127.0.0.1:8080/
g4f-go sandbox --desktop          # full desktop image, pulled on first use
g4f-go start-desktop-sandbox       # shorthand for starting the desktop image
g4f-go start-desktop-sandbox --app opencode  # run a tool instead of the live wallpaper
g4f-go start-desktop-sandbox --no-open --port 9090
g4f-go sandbox --desktop connect  # interactive bash shell in the running desktop
g4f-go sandbox --desktop exec ls /home/pn
g4f-go sandbox --desktop cli list # list installable sandbox tools
g4f-go sandbox --desktop install chrome
g4f-go sandbox status
g4f-go sandbox down
```

`connect` is an alias for `shell`; both it and `exec` require the selected
container to exist. Use `--desktop` for the desktop container (without it,
`sandbox` targets the API-only container). Start it first with
`start-desktop-sandbox` or `sandbox --desktop up`.

`--app TOOL` starts a `sandbox-cli` desktop tool (for example `opencode` or
`chrome`) in a maximized terminal instead of opening the live wallpaper. The
tool is installed on first use if needed. Changing or removing `--app` when
starting again recreates the desktop container so the startup choice takes
effect; persistent sandbox data remains in the data directory. It also works
with `g4f-go sandbox --desktop --app TOOL`.

`g4f-go sandbox-cli <args...>` is shorthand for
`g4f-go sandbox --desktop cli <args...>`: it starts the desktop container when
needed (without opening a browser) and runs `sandbox-cli` inside it, for
example `g4f-go sandbox-cli settings` or `g4f-go sandbox-cli hosts`.

The launcher uses Docker when it is installed and falls back to Podman. If
neither CLI is present, `g4f-go sandbox` installs Podman with the host package
manager (`apt`, `dnf`, `pacman`, `zypper`, `apk`, Homebrew, `winget`, Chocolatey
or Scoop) before pulling the image. Set `G4F_SANDBOX_INSTALL_PODMAN=0` to
disable that automatic install. The full launcher options and environment
variables are available with `g4f-go sandbox help` and in the
[sandbox desktop documentation](../docs/sandbox-desktop.md).

On **Termux/Android** the launcher selects [udocker](https://github.com/indigo-dc/udocker)
instead: it pulls and runs the same sandbox images in user space, without root
or a Docker daemon. On first use the launcher installs the pip package (with
`pkg install python` first) and runs `udocker install` for the tools, so the
first sandbox start does not stop halfway through. The desktop image works with
`g4f-go sandbox --desktop`. The udocker root defaults to `~/.udocker` and can
be moved with `G4F_SANDBOX_UDOCKER_DIR` (useful when `$HOME` is small);
`G4F_SANDBOX_UDOCKER_EXECMODE` selects the execution mode (`P1`, `P2`, `F1`-`F4`,
`R1`-`R3` or `S1`, default `P1`). Set `G4F_SANDBOX_INSTALL_UDOCKER=0` to require
a preinstalled binary and tools. The older native Termux Python backend remains
available with `--backend termux` (API-only). Selecting another backend
explicitly, for example `--engine docker`, still wins over the Termux
autodetection.

## Standalone pi-freeflow (`freeflow`)

The free-model catalog behind `bin/dev-pipeline` is normally loaded as an
extension by Pi or Oh My Pi (OMP). `g4f-go freeflow` installs the same
[`pi-freeflow`](https://github.com/trefeon/pi-freeflow) npm package into
`~/.g4f/freeflow` and runs its proxy daemon directly, with no Pi/OMP host and
no Python runtime:

```
g4f-go freeflow                    # install (first run) and serve on :28180
g4f-go freeflow --port 9090        # use another loopback port
g4f-go freeflow install --force    # update the npm package
g4f-go freeflow status             # package, base URL, API key, daemon health
```

It exposes the OpenAI-compatible endpoint `http://127.0.0.1:28180/v1` and
prints the bearer token from `~/.pi/agent/token`. The daemon stays in the
foreground until Ctrl-C, and logs to `~/.pi/agent/pi-freeflow.log`. Existing
OMP/Pi installs are reused when present; otherwise the package is installed
with `npm` or Bun. Node.js >= 22.19 is preferred; Bun >= 1.0 runs the
TypeScript daemon when Node.js is unavailable or too old. Set
`G4F_FREEFLOW_HOME` to move the private install, or
`PI_FREEFLOW_DATA_DIR`/`PI_FREEFLOW_PORT` to match the extension's own settings.

## Polli cloud sandbox (`polli-sandbox`)

`g4f-go polli-sandbox` installs the [Pollinations CLI](https://github.com/pollinations/pollinations)
and the OpenSSH client, creates a hosted Polli (E2B) sandbox, and installs
`g4f-go` plus `sandbox-cli` inside it:

```
g4f-go polli-sandbox                    # create a sandbox, install g4f-go + sandbox-cli
g4f-go polli-sandbox --desktop          # also install Xfce + TigerVNC + noVNC
g4f-go polli-sandbox --template pollinations-large --timeout 3600 --desktop
g4f-go polli-sandbox install-menu <id> --desktop   # set up an existing sandbox
g4f-go polli-sandbox list|connect|logs|pause|kill|timeout|ssh-config ...
```

`--connect` opens an SSH session after setup. `--no-setup` only creates the
sandbox. When g4f-go is running as a Linux binary and the remote architecture
matches, the running binary is uploaded; otherwise (or with `--remote`) the
latest release is installed remotely with `bin/install.sh`. Use
`--g4f-go-version V` to pin that release.

With `--desktop`, the sandbox gets the Xfce + TigerVNC desktop and the noVNC web
client. The command prints the SSH tunnels:

```
ssh -L 8080:127.0.0.1:8080 <id>.polli   # http://127.0.0.1:8080/desktop/
ssh -L 5900:127.0.0.1:5900 <id>.polli   # direct VNC client
```

The embedded `sandbox-cli` and desktop bootstrap are synced from `bin/`; the
`g4f-go sandbox` and `polli-sandbox` commands share the same generated images
and environment variables.

## Headless browser (`browser`)

g4f drives a real browser over CDP for cookie fetching and scraping. `g4f-go`
can install and manage that browser for you — the
[g4f-copilot/browser](https://github.com/g4f-copilot/browser) build of
[Lightpanda](https://github.com/lightpanda-io/browser), a small headless CDP
browser with no GUI that ships a Chrome-impersonating fingerprint out of the
box.

```
g4f-go browser install      # download + verify + install for this OS
g4f-go browser serve        # run the CDP server in the foreground
g4f-go browser status       # show install state
g4f-go browser path         # print the installed binary path
g4f-go browser uninstall    # remove it
```

`browser serve` accepts `--host` (default `127.0.0.1`) and `--port` (default
`9222`; `0` picks a free port). Any other flag is forwarded to
`lightpanda serve`.

Both `browser serve` and the automatic startup always pass the same extra
flags, matching what the Python package uses:

```
--block-private-networks
--load-resources iframe
--http-cache-dir <config dir>/cache/http_cache
```

The cache directory is shared with `g4f.requests.cdp.get_http_cache_dir()`.

No user-agent flag is passed. The g4f-copilot build already sends a full
Chrome UA (`Mozilla/5.0 (Windows NT 10.0; Win64; x64) ... Chrome/152.0.0.0
Safari/537.36`) together with matching `Sec-Ch-Ua` client hints, and
`--user-agent-suffix` would append a second `Chrome/152.0.0.0 Safari/537.36`
to it. `g4f/requests/cdp.py` still passes the suffix, which is harmless there
because it masks the UA over CDP anyway.

### Automatic startup when headless

g4f runs its browser headless by default (`BrowserConfig.headless = True`).
When g4f-go runs g4f — or a `.py` script — and headless mode is active, it starts
the installed browser on a free port and points g4f at it via `G4F_BROWSER_HOST` /
`G4F_BROWSER_PORT`, then shuts it down when the child process exits:

```
$ g4f-go client "hello"
browser: started Lightpanda on 127.0.0.1:64188 (headless)
...
```

The automatic startup is skipped when:

- the browser is not installed (`g4f-go browser install` first),
- `G4F_BROWSER_PORT` is already set (an existing endpoint is reused),
- `G4F_BROWSER_MODE` is set to something other than `cdp`,
- headless mode is off — via `--no-headless` or `G4F_BROWSER_HEADLESS=false`.

A PID file (`.autostart.pid`) lets the next run reclaim a browser that outlived
a hard-killed g4f-go.

### Automatic startup from Python

The g4f package performs the same detection on its own, so `python -m g4f`,
`g4f client` and the API server use the installed browser without going through
g4f-go. When headless mode is on, `g4f.requests.cdp` looks for the binary in
`~/.g4f/browser/` (the directory `g4f-go browser install` writes to) and starts
`lightpanda serve` on a free port, preferring it over a locally installed
Chrome. The same skip conditions apply, plus:

- `G4F_BROWSER_LIGHTPANDA_PATH` overrides the binary location,
- a `lightpanda` executable on `PATH` is used as a fallback.

The process is stopped again by the regular shutdown paths (idle timer, `atexit`,
API lifespan). Lightpanda does not implement the CDP `Browser.close` command
(it answers `-32601`), so g4f kills the process it started instead.

### Downloads

Binaries are pinned per platform and verified against a SHA-256 before install:

| OS | Arch | Source |
|----|------|--------|
| Linux | amd64, arm64 | `g4f-copilot/browser` release (zip) |
| Windows | amd64 | `qidiai/lightpanda-windows-port` (upstream has no native Windows build) |

Installed into `~/.g4f/browser/` (next to the CPython runtime). The zips are
unpacked with their top-level directory stripped (the Windows package has one,
the Linux zips are flat) and the binary is `chmod 0755`-ed, since the archives
carry no unix permission bits.

The Linux amd64 build links against `libc_v8.so`, which is not part of the
release zip. `g4f-go browser install` downloads the matching
`lightpanda-io/zig-v8-fork` library into the same directory and rewrites the
binary's ELF `RUNPATH` to `$ORIGIN`, because the shipped value points at the
build machine's absolute path. The Linux arm64 build is self-contained.

## STUN/TURN server (`turn`)

g4f-go embeds [pion/turn](https://github.com/pion/turn), so a full STUN/TURN
relay ships inside the binary — no coturn, no extra process to install:

```
g4f-go turn serve --public-ip 203.0.113.7
g4f-go turn status
g4f-go turn env --json
```

One server answers STUN Binding requests and TURN allocations on the same
port (UDP and TCP by default, TLS with `--cert`/`--key` or `--tls-self-signed`).
Authentication uses the TURN REST scheme (`use-auth-secret` in coturn terms):
the username carries the expiry and the password is its HMAC-SHA1 digest keyed
with a shared secret, so nothing is stored server side. The secret is generated
on first use into `~/.g4f/turn/secret` (mode `0600`) and can be rotated with
`g4f-go turn secret --new`.

| Flag | Meaning |
|---|---|
| `--public-ip <IP>` | address peers should send media to (default: STUN probe, then local IPv4) |
| `--bind <IP>` | local address to bind (default: all interfaces) |
| `--port <PORT>` | UDP/TCP port (default `3478`) |
| `--tls-port <PORT>` | TLS port (default `5349`) |
| `--realm <REALM>` | authentication realm (default: the public IP) |
| `--secret <SECRET>` | shared secret (default: the persisted one) |
| `--min-port` / `--max-port` | relay port range (default `49160`–`49200`) |
| `--no-tcp` | disable the TCP listener |
| `--cert` / `--key` / `--tls-self-signed` | TLS listener |
| `--max-allocations N` | per-IP allocation quota, `0` disables (default `100`) |
| `--log-level <LEVEL>` | `disable`, `error`, `warn`, `info`, `debug`, `trace` |

Every flag also has an environment variable (`G4F_TURN_PUBLIC_IP`,
`G4F_TURN_BIND`, `G4F_TURN_PORT`, `G4F_TURN_TLS_PORT`, `G4F_TURN_REALM`,
`G4F_TURN_SECRET`, `G4F_TURN_MIN_PORT`, `G4F_TURN_MAX_PORT`, `G4F_TURN_QUOTA`,
`G4F_TURN_CERT`, `G4F_TURN_KEY`); flags win over the environment. `g4f-go turn
env` prints the `RD_TURN_*` variables that point the remote desktop server at a
running instance.

`g4f-go turn credentials` mints a username/password pair for a client without
starting a server, which is handy for testing a relay or for handing
credentials to a third-party client:

```
g4f-go turn credentials --ttl 1h --json
```

`--ttl` accepts Go durations (`1h`, `30m`) as well as bare seconds (`600`), and
defaults to `G4F_TURN_TTL` or one hour.

### Remote desktop relay

`g4f-go -m remote_desktop` starts the embedded server automatically and passes
`RD_TURN_URL`, `RD_TURN_SECRET` and `RD_STUN_URL` to the Python process, so
pairing a phone over mobile data works out of the box. Set
`G4F_TURN_AUTOSTART=0` to opt out, or set `RD_TURN_URL`/`RD_ICE_SERVERS`
yourself to use an external relay instead.

Because both sides implement the same TURN REST scheme, credentials minted by
`remote_desktop.config.turn_credentials()` authenticate against the embedded
server and vice versa — the shared secret is the only thing that has to match.

`projects/remote-desktop/deploy/setup-turn.sh` remains available for hosts
that prefer a standalone coturn instance; it is no longer required.

## Bundled modules (`-m`)

`g4f-go -m <module>` runs `python -m <module>` with the downloaded runtime.
Modules that ship inside the binary are extracted into the runtime first and
put on `PYTHONPATH`; any other module name is forwarded to the interpreter
unchanged (so `g4f-go -m pip list` works too).

| Module | Description |
|---|---|
| `remote_desktop` | Remote desktop server: shares this screen with a phone through the browser's native screencast API, with QR pairing and an optional input bridge. |

```
g4f-go -m remote_desktop --port 8000
g4f-go -m remote_desktop --help
```

To reach the phone over mobile data (no shared Wi-Fi) the two browsers need a
relay, because they otherwise only learn their LAN addresses. `g4f-go` starts
its own embedded STUN/TURN server for this (see
[STUN/TURN server](#stuntturn-server-turn)); to use an external one instead,
point the server at it and it hands time-limited credentials to both pages:

```
g4f-go -m remote_desktop --turn-url turn:turn.example.com:3478 --turn-secret <shared-secret>
```

`projects/remote-desktop/deploy/setup-turn.sh` installs and configures coturn
with the matching `use-auth-secret` setting for that external case.

Extra dependencies of a bundled module that are not part of `g4f[slim]`
(`qrcode`, `pynput` for `remote_desktop`) are pip-installed into the runtime on
first use. If that install fails, the module still starts with reduced
functionality (no QR code / no input control).

### Bundle sources

The bundle lives in `g4f-go/bundle/` and is generated from the repository:

```
./sync-bundle.sh            # projects/remote-desktop -> g4f-go/bundle/
```

`build-all.sh` and `make all` run it automatically, so releases always ship the
current sources. Bump `BundleRevision` in `version.go` when the bundle changes
so existing installations re-extract it.

## Supported platforms

| OS | Arch | Runtime source |
|----|------|----------------|
| Linux | amd64, arm64 | python-build-standalone install-only tarball |
| Windows | amd64 | python-build-standalone install-only tarball |
| macOS | amd64, arm64 | python-build-standalone install-only tarball |
| Android | arm64 (Termux) | official python.org `*-linux-android` package |

Android note: the python.org Android package ships `libpython3.15.so` +
stdlib but no `python` executable. `g4f-go` detects Termux
(`pm list packages`), merges the tarball into the app dir, and compiles a tiny
C runner with Termux's clang that `dlopen`s libpython — the same technique as
CPython's own android testbed.

## Layout after first run (`~/.g4f/python-embed/`)

```
python-home/bin/python     interpreter (pbs layout)
python-home/lib/python3.15 stdlib + site-packages (g4f installed here)
python (launcher)          shell wrapper that sets PYTHONHOME/PYTHONPATH
bundle/                    extracted embedded Python bundle (remote_desktop, web)
.g4f-runtime/.runtime-ok   stamp: download+extract complete
.g4f-runtime/.installed    stamp: g4f pip-installed
.g4f-runtime/.g4f-version  stamp: g4f version pinned via -v (absent: latest)
.g4f-runtime/.bundle-revision  stamp: extracted bundle revision
```

The headless browser lives next to it in `~/.g4f/browser/`:

```
lightpanda(.exe)           browser binary
VCRUNTIME140*.dll          Windows runtime DLLs (Windows package only)
.version                   stamp: installed platform + version
.autostart.pid             PID of the browser g4f-go started (while running)
```

## Limitations

- The runtime is materialized on disk (CPython cannot run a 100%-in-memory
  interpreter reliably); first run downloads ~50–800 MB depending on platform.
- macOS builds must be signed/notarized by the distributor for Gatekeeper.
- Android builds need Termux installed to compile the dlopen runner on device.
