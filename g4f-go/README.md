# g4f-go

A small, self-contained Go launcher for [gpt4free](https://github.com/xtekky/gpt4free).
Unlike the original design, the CPython runtime is **not embedded** in the
binary: `g4f-go` is a few MB, and downloads the correct CPython for the host
platform on first run (with live progress feedback), then installs `g4f` into
it. No system Python required.

```
g4f-go client "What is gpt4free?"
g4f-go api --port 8080
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
./fetch-python.sh           # optional: re-pin sizes + sha256 in runtime.json
```

The manifest is embedded via `go:embed runtime.json`; the binary builds
without network access.

## Usage

```
g4f-go <g4f args...>        run gpt4free (e.g. g4f-go client "hello")
g4f-go api --port 8080      start the OpenAI-compatible API server
g4f-go gui                  launch the web GUI
g4f-go status               show runtime download/install status
g4f-go install g4f          (re)install the g4f package (network)
g4f-go help                 show help
```

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
.g4f-runtime/.runtime-ok   stamp: download+extract complete
.g4f-runtime/.installed    stamp: g4f pip-installed
```

## Limitations

- The runtime is materialized on disk (CPython cannot run a 100%-in-memory
  interpreter reliably); first run downloads ~50–800 MB depending on platform.
- macOS builds must be signed/notarized by the distributor for Gatekeeper.
- Android builds need Termux installed to compile the dlopen runner on device.
