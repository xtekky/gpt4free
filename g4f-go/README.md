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
