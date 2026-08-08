# g4f-go

A single self-contained executable for [gpt4free](https://github.com/xtekky/gpt4free):
a small Go launcher with an embedded, relocatable CPython runtime that has
gpt4free pre-installed. No system Python required.

```
g4f-go client "What is gpt4free?"
g4f-go api --port 8080
```

## How it works

1. `fetch-python.sh` downloads the official CPython *embeddable* zips from
   python.org for every supported platform, copies the `g4f` package into the
   interpreter's `python-home/`, drops prebuilt wheels in `wheels/`, patches
   `python311._pth` for relocation, and re-packs everything into
   `embed/<os>/<os>-<arch>-embed-<ver>.zip`.
2. `go:embed` (build tags per OS) bakes that archive into the launcher binary.
3. On first run the launcher extracts the archive next to itself, finishes the
   g4f install from the bundled wheels (offline), then execs
   `python -m g4f <args>`.
4. `build-all.sh` cross-compiles the launcher with `CGO_ENABLED=0` for every
   target and zips each release.

```
┌──────────────────────────────┐
│  g4f-go (Go, ~10 MB binary)  │
│  ┌────────────────────────┐  │
│  │ CPython 3.14 runtime   │  │  ← baked in via go:embed
│  │ + g4f + wheels          │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
   │  first run: extract next to binary
   ▼
g4f-go.exe/ + python-home/ + wheels/ + .g4f-runtime/.installed
   │
   ▼
exec python -m g4f <args>
```

## Build

Requires Go 1.22+ and (for `fetch-python.sh`) bash, curl, unzip, zip, python3.

```bash
cd g4f-go

# 1. Download & embed the Python runtimes (all platforms, ~200 MB on disk)
./fetch-python.sh
./fetch-python.sh linux          # or just one platform

# 2. Build launchers for every OS/arch
./build-all.sh                   # -> dist/g4f-go-<ver>-<os>-<arch>.zip
# or a single one:
go build -o g4f-go .
```

Release zips live in `dist/`. Each zip is a portable folder: drop it anywhere,
run `g4f-go`, and the runtime is extracted next to it on first launch.

> **Size note:** CPython runtimes are ~40-60 MB per platform, so release zips
> are large (Windows ≈ 160 MB, Linux ≈ 820 MB). The `.zip` is stored inside the
> Go binary, and extracted on first run; the Go executable itself stays small.

## Usage

```
g4f-go <g4f args...>        run gpt4free (e.g. g4f-go client "hello")
g4f-go api --port 8080      start the OpenAI-compatible API server
g4f-go gui                  launch the web GUI
g4f-go status               show embedded runtime status
g4f-go install g4f          (re)install the g4f package (network)
g4f-go help                 show help
```

Environment: `G4F_PYTHON_ONLY=1 g4f-go --version` prints the embedded python
path (useful for wrapping the runtime from other tools).

## Supported platforms

| OS | Arch | Notes |
|----|------|-------|
| Linux | amd64, arm64 | official python.org x64 embed |
| Windows | amd64, x86 | official python.org win64/win32 embed |
| macOS | arm64 (universal2 embed), amd64 | universal2 zip runs on both |
| FreeBSD | amd64 | reuses the Linux x64 embed (Python-only, no C extensions) |

`fetch-python.sh` can be extended to other OSes; add a row and a matching
`runtime_<os>.go` build-tagged file.

## Layout inside a release folder

```
g4f-go (binary)
python.exe / python (launcher)
python-home/  g4f package + site-packages
wheels/       prebuilt g4f dependency wheels
.g4f-runtime/.installed   (stamp written on first run)
```

## Limitations

- The runtime is extracted to disk next to the executable (this is what makes
  it *relocatable* — CPython cannot run a 100%-in-memory interpreter reliably).
- Bundled dependency versions are pinned by `requirements-min.txt` at build
  time; use `g4f-go install g4f` for network installs.
- macOS builds must be signed/notarized by the distributor for Gatekeeper.
