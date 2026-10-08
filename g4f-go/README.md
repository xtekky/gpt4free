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

`G4F_PYTHON_ONLY=1 g4f-go --version` prints the downloaded interpreter path
without running gpt4free (useful for wrapping the runtime from other tools).

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
g4f-go turn serve           run the embedded STUN/TURN server in the foreground
g4f-go turn status          show the configured/running STUN/TURN server
g4f-go turn credentials     mint TURN REST credentials for a client
g4f-go install g4f          (re)install the g4f package (network)
g4f-go help                 show help
```

## Headless browser (`browser`)

g4f drives a real browser over CDP for cookie fetching and scraping. `g4f-go`
can install and manage that browser for you — [Lightpanda](https://github.com/lightpanda-io/browser),
a small headless CDP browser with no GUI.

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
| Linux | amd64, arm64 | `lightpanda-io/browser` release |
| macOS | amd64, arm64 | `lightpanda-io/browser` release |
| Windows | amd64 | `qidiai/lightpanda-windows-port` (upstream has no native Windows build) |

Installed into `~/.g4f/browser/` (next to the CPython runtime). The Windows
package is a zip whose top-level directory is stripped and whose binary is
`chmod 0755`-ed, since the archive carries no unix permission bits.

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

Android note: the python.org Android package ships `libpython3.14.so` +
stdlib but no `python` executable. `g4f-go` detects Termux
(`pm list packages`), merges the tarball into the app dir, and compiles a tiny
C runner with Termux's clang that `dlopen`s libpython — the same technique as
CPython's own android testbed.

## Layout after first run (`~/.g4f/python-embed/`)

```
python-home/bin/python     interpreter (pbs layout)
python-home/lib/python3.14 stdlib + site-packages (g4f installed here)
python (launcher)          shell wrapper that sets PYTHONHOME/PYTHONPATH
bundle/                    extracted embedded Python bundle (remote_desktop, web)
.g4f-runtime/.runtime-ok   stamp: download+extract complete
.g4f-runtime/.installed    stamp: g4f pip-installed
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
