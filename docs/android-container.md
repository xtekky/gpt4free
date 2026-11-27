# Android container app (v2)

The Android app has two modes:

1. **Container mode** (default launcher): a rootless Debian 12 environment runs
   gpt4free under PRoot. An in-app terminal controls shell, API and desktop
   sessions. XFCE is installed only if the user opts in.
2. **Classic mode**: the original Chaquopy-powered chat WebView from
   [`projects/android/README.md`](../projects/android/README.md).

This document explains the container implementation. It is inspired by the
*architecture* of [LinuxOnAndroid](https://github.com/devwithzachary/LinuxOnAndroid)
(PRoot plus an Android PTY bridge) but is an independent implementation; no code
or binaries from that project are used.

## Layout on the device

All state lives in the app's private `filesDir`:

```
filesDir/
├── app/                    # Classic mode's extracted source (separate)
└── container/
    ├── g4f-src/            # bundled gpt4free source, bound at /g4f-src
    │   ├── g4f/
    │   ├── setup.py
    │   └── README.md
    ├── bin/                # repo bin commands + Linux g4f-go, bound at /g4f-bin
    ├── downloads/          # verified .deb and rootfs archives
    ├── resolv.conf         # bind-mounted at /etc/resolv.conf in every session
    ├── rootfs/             # Debian 12 bookworm filesystem
    ├── runtime/            # libtalloc.so.2 and libandroid-shmem.so for PRoot
    │   └── lib/
    ├── tmp/                # PRoot scratch directory (PROOT_TMP_DIR)
    └── scripts/            # bound at /g4f-scripts, provision markers
        ├── provision-base.sh
        ├── provision-desktop.sh
        ├── start-api.sh
        └── start-desktop.sh
```

`assets/container/catalog.json` pins every remote artifact by SHA-256. A
checksum mismatch aborts installation; no unsigned or floating binary is used.
The optional `g4f.prebuilt` entry can pin a gzip tarball containing
`opt/g4f-venv` and `usr/local/bin/g4f-api`. The current pin is the exact
v8.9.4 release asset, not a floating `latest/download` URL. When present, the
overlay replaces the apt/pip pass entirely; when it is `null`,
`provision-base.sh` runs as the fallback.

`runtime.python.packages` pins the exact Debian arm64/amd64 packages used to
provide `/usr/bin/python3` and its standard library. `prepare_python.py`
verifies every package and extracts only the needed files at build time, so the
prebuilt venv works without running apt on the phone. `g4f.desktop_prebuilt`
can similarly pin an optional XFCE overlay, downloaded only when **Desktop** is
checked. A single APK asset tree
can hold only one architecture; the build defaults to `arm64-v8a` and accepts
`-PcontainerPythonAbi=x86_64` for x86 builds. The same build task copies
all repository `bin/` commands plus the Linux arm64 `g4f-go` binary into
`container/bin`, which is bound at `/g4f-bin` and placed first on `PATH`.
`G4F_PYTHON=/opt/g4f-venv/bin/python` is exported in every session and in
`/etc/profile.d/g4f.sh`, so `g4f-go` inside the container reuses the bundled
venv interpreter instead of trying to download a CPython archive over HTTPS
(which can fail when the container has no CA certificates). `g4f-go` also
detects that explicit `G4F_PYTHON` and no longer points `PYTHONHOME` at its own
downloaded runtime, which would break the venv interpreter in the same way.
`writeLaunchers()` also refreshes `g4f-go` from the APK on reconnect, so an app
update repairs an existing container without reinstalling it. The same profile
and every launcher also clear `PYTHONHOME`; a stale value makes
CPython fail at startup with `No module named encodings` / `Failed to get codec
of filesystem encoding`. `runtime.curl`
pins a static musl curl with an embedded CA bundle; `prepare_curl.py` verifies
its archive and binary hashes and places it in `container/bin` so `sandbox-cli`
can download installers without apt or system certificates. The Android
`g4f-go` build stays separate for Termux; it uses the Termux `python` package
and cannot run inside the Debian rootfs. The release workflow publishes it as
`g4f-go-<version>-android-arm64.zip` alongside the desktop launchers.

Android denies `execve()` of binaries in app-writable storage. The build-time
helper `prepare_proot.py` therefore extracts `libproot.so` and
`libproot_loader.so` from the pinned Termux package into `app/src/main/jniLibs`.
They are installed in the APK's read-only `nativeLibraryDir`; `PROOT_LOADER`
and `PROOT_TMP_DIR` point PRoot at that loader and the writable `container/tmp`
scratch directory. Only the non-executable support libraries remain in
`container/runtime/lib`. A small native preload, `libg4f_linkfix.so`, is also
packaged there: PRoot cannot create hard links in app data on some Android
kernels, so it makes `link()` copy the file instead. This is what lets dpkg
create its `/var/lib/dpkg/status-old` backup.

## Startup sequence

1. `ContainerActivity` starts `ContainerService` as a foreground service with a
   partial wake lock, so downloads and long-running sessions survive screen-off.
2. `ContainerManager.install()` downloads the rootfs and the two Termux support
   packages, verifies their hashes, extracts them (with path-traversal checks),
   writes the bind-mounted resolver, copies the provisioning scripts, the
   bundled Python runtime and the `bin/` commands. PRoot itself and its loader
   were already packaged as native libraries during the build.
3. If `g4f.prebuilt` is pinned, the verified overlay is extracted directly into
   the rootfs and `.provisioned-base` is written—no apt/pip pass. Otherwise a
   **setup** terminal session runs `provision-base.sh` inside PRoot: it clears
   stale dpkg/apt locks, repairs dpkg state, rewrites `/etc/resolv.conf`, retries
   `apt-get update` up to three
   times, installs Python 3, then creates `/opt/g4f-venv` and installs the
   bundled source with the `api` extra plus Flask dependencies.
4. If the user checked **Desktop**, a pinned `desktop_prebuilt` overlay is
   extracted immediately; otherwise a setup session runs
   `provision-desktop.sh`. It delegates the apt work to the shared
   `install-desktop.sh`, which is also used by the Polli sandbox bootstrap.
   At build time `prepare_novnc.py` unpacks pinned noVNC 1.6.0, layers the
   sandbox-desktop landing page/touch controls on top and runs
   `novnc-brand.sh`, so the APK carries a **complete g4f-branded client** —
   not just an overlay. The full tree replaces `/usr/share/novnc`, including
   the `vnc.html` and icon logos; no stock noVNC mark is left.
5. Once the filesystem is ready, **API** is started automatically with
   `--debug`, which mounts the bundled GUI. With `G4F_VNC_PORT` exported by the
   launcher, the same server serves the noVNC client at `/desktop/`.
6. Chat, API and Desktop are the first three single-instance tabs. **Chat**
   starts the API session if needed and opens its bundled GUI at
   `http://127.0.0.1:1337/chat/`; **Desktop** opens
   `http://127.0.0.1:1337/desktop/`. Desktop mode launches `Xvnc :1` and
   `startxfce4` in the background through `dbus-run-session` (with `dbus-x11`
   installed), so XFCE has a session bus and does not get stuck on
   "Unable to contact settings server". It waits for the X socket before it
   reports readiness, reusing a live display (and recovering stale lock files)
   on reconnect. Its PTY is not rendered as a terminal tab, so the button always
   shows the VNC page instead of the display-server console. No second HTTP
   server is needed. The desktop add-on is never downloaded unless selected.
   The noVNC **Full** button uses the WebView's HTML fullscreen custom view;
   when an iframe request is blocked, the `AndroidDesktop` bridge hides the
   app chrome and Android system bars instead. Closing an overlay hides its
   Close button, and stopping the container shows a stopped message rather than
   an exit error.

## Terminal implementation

`app/src/main/cpp/pty.cpp` allocates a pseudo-terminal with `posix_openpt()`,
`grantpt()` and `unlockpt()`, forks a child that calls `setsid()` and `execve()`s
PRoot with the slave PTY attached to stdin/stdout/stderr. The Java layer reads
the master side on a dedicated thread and forwards bytes (base64-encoded) to
`assets/container/terminal.html`, which renders them with a locally bundled
xterm.js 5.5.0 client (plus the Fit addon) — no CDN is fetched at runtime.
Keystrokes and resize events travel back through the `AndroidTerminal`
JavaScript bridge. `/etc/profile.d/g4f.sh` keeps `/g4f-bin` on `PATH` for login
shells, so the repo `bin/` commands and `g4f-go` remain available after
`bash -l` resets `PATH`.

The terminal needs no CDN: xterm.js renders ANSI colors and cursor movement.
The fixed Chat/API/Desktop tabs stay visible while any number of shell tabs
scroll separately and are restored on reconnect. Each new shell opens with a
short banner pointing to `sandbox-cli list`, `sandbox-cli <tool>` and `g4f-go`.
A Termux-style key row is pinned below the terminal (Esc, Tab, Ctrl, Alt,
arrows, Home/End, `^C`, `^D`). The sticky Ctrl/Alt keys also modify characters
typed on the hardware keyboard, so selecting Ctrl and pressing `c` sends `^C`.
A mostly-horizontal finger drag selects text (and copies it), while vertical
drags scroll the terminal. Touching either end of the last selection and dragging
adjusts that range. Copy/All keep the selection when tapped, and the dynamic
shell tab strip scrolls horizontally while Close/Close All stay pinned in view.

## Why the desktop is optional

A GUI environment adds roughly 300–500 MB and several hundred packages to a
phone. Keeping it opt-in means a typical install only needs Python, pip and the
API dependencies. The base and desktop provisioning scripts are separate, so an
installed container can be upgraded with just:

```sh
bash /g4f-scripts/provision-desktop.sh
```

from any terminal tab. It writes `.provisioned-desktop` in the app-visible
scripts directory, which is how the UI knows the desktop is ready.

## Limitations

- PRoot needs ptrace; some hardened OEM firmwares block it. In that case the
  setup terminal shows a permission/ptrace error from PRoot.
- Only `arm64-v8a` and `x86_64` are shipped, matching the Chaquopy ABIs.
- The terminal renderer is line-oriented, but keeps full PTY input/output and
  the Termux-style key row, so shell editors remain usable without a CDN.
- Classic chat and the container API both use port 1337. Stop one before
  starting the other.
- The PRoot, talloc and shmem binaries are GPL/BSD licensed third-party
  artifacts from the Termux package repository; PRoot and its loader are
  extracted at build time, talloc and shmem at install time. Their licenses are
  listed in the app README.

## Verifying a release

```bash
# APK contains the native PTY library, PRoot and its loader for both ABIs
unzip -l app/build/outputs/apk/release/app-release.apk | grep -E 'libg4f_pty|libg4f_linkfix|libproot'

# Catalog hashes match the pinned upstream artifacts
python3 -m unittest etc.unittest.test_android_container_catalog
```
