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
No Python runtime or venv is shipped in the APK or the catalog. The optional
`g4f.prebuilt` entry can still pin a gzip overlay with pre-provisioned
launchers, but it is `null` so a fresh install uses `provision-base.sh`; the
old overlay contained `/opt/g4f-venv` and is intentionally not used.
`g4f.desktop_prebuilt` can pin an optional XFCE overlay, downloaded only when
**Desktop** is checked. A single APK asset tree can hold only one architecture;
the build defaults to `arm64-v8a` and accepts `-PcontainerPythonAbi=x86_64` for
x86 builds. The same build task copies all repository `bin/` commands plus the
Linux arm64 `g4f-go` binary into `container/bin`, which is bound at `/g4f-bin`
and placed first on `PATH`.

Python is supplied by `g4f-go`, not by the app. On first use inside the
container, `g4f-go` downloads its pinned CPython archive into
`/root/.g4f/python-embed`, verifies the SHA-256, and installs the bundled
`/g4f-src` checkout with an editable pip install. `provision-base.sh` only
installs `ca-certificates`, `sudo` and `git` via apt (the download needs a CA
bundle; git and sudo are part of the interactive container); it never
runs `python3 -m venv` or pip directly. The editable install marks the runtime
as builtin, so later `g4f-go api` runs do not replace the APK's source with
PyPI's latest release. `writeLaunchers()` refreshes `g4f-go` from the APK on
reconnect, so an app update repairs an existing container without reinstalling
it. Every launcher and the shell profile clear `PYTHONHOME`; a stale value makes
CPython fail at startup with `No module named encodings` / `Failed to get codec
of filesystem encoding`. `runtime.curl` pins a static musl curl with an embedded
CA bundle; `prepare_curl.py` verifies its archive and binary hashes and places it
in `container/bin` so `sandbox-cli` can download installers without apt or system
certificates. The Android `g4f-go` build stays separate for Termux; it uses the
Termux `python` package and cannot run inside the Debian rootfs. The release
workflow publishes it as `g4f-go-<version>-android-arm64.zip` alongside the
desktop launchers.

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
   writes the bind-mounted resolver, copies the provisioning scripts and the
   `bin/` commands. PRoot itself and its loader were already packaged as native
   libraries during the build.
3. If `g4f.prebuilt` is pinned, the verified overlay is extracted directly into
   the rootfs and `.provisioned-base` is written. The current catalog leaves it
   `null`, so a **setup** terminal session runs `provision-base.sh` inside
   PRoot: it clears stale dpkg/apt locks, repairs dpkg state, rewrites
   `/etc/resolv.conf`, retries `apt-get update` up to three times, installs
   `ca-certificates`, `sudo` and `git`, and then runs
   `g4f-go install -e /g4f-src[api]` plus the
   small Flask GUI dependencies. `g4f-go` downloads its own CPython (~60 MB)
   and owns the pip install; the rootfs gets no Python, pip or venv.

   The first run therefore needs a few minutes and a working connection. A
   dropped mobile transfer no longer hangs forever: `g4f-go` aborts a stalled
   read after 90 s, retries three times, and resumes from the bytes already on
   disk with an HTTP Range request. The provisioning script retries the
   `g4f-go install` once more and mirrors its full output to
   `/g4f-scripts/provision-base.log` (app files dir), which survives the
   terminal session and can be pulled with `adb` when something fails. The
   setup tab shows the same progress live, including the download bar.
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
   `g4f-go api --debug`, which mounts the bundled GUI from the installed g4f
   package. With `G4F_VNC_PORT` exported by the launcher, the same server serves
   the noVNC client at `/desktop/`.
6. Home, Chat, API, Desktop and Store are pinned before the shell tabs. **Home** opens
   the locally served site home page at `http://127.0.0.1:1337/`; **Chat**
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
   an exit error. **Store** reads `sandbox-cli list` in a hidden PTY, renders a
   searchable catalog of every sandbox tool, and runs
   `sandbox-cli install <tool>` in a visible terminal when **Install** is tapped.
   `list` also appends a `g4f-bin commands` group with every file bound at
   `/g4f-bin`, using each script's first comment or docstring line as its
   description; **Terminal** opens one in a shell with its name at the prompt.
   New tools and commands appear without an APK update.

## Terminal implementation

`app/src/main/cpp/pty.cpp` allocates a pseudo-terminal with `posix_openpt()`,
`grantpt()` and `unlockpt()`, forks a child that calls `setsid()` and `execve()`s
PRoot with the slave PTY attached to stdin/stdout/stderr. The Java layer reads
the master side on a dedicated thread and forwards the bytes to the native
`TerminalView`
(`app/src/main/java/dev/g4f/android/container/TerminalView.java`). The view
decodes UTF-8 incrementally, interprets the ANSI/VT100 subset produced by bash,
apt and the bundled tools, and paints its cell grid with `Canvas`. No WebView,
CDN or bundled JavaScript is involved in the terminal path.
Keystrokes and resize events travel back through `ContainerService.writeBytes()`
and `ContainerService.resize()`. `/etc/profile.d/g4f.sh` keeps `/g4f-bin` on
`PATH` for login shells, so the repo `bin/` commands and `g4f-go` remain
available after `bash -l` resets `PATH`.

The renderer handles 16/256/truecolour SGR, cursor addressing and movement,
erase/insert/delete line and character operations, scroll regions, the alternate
screen, wide characters and up to 5000 lines of scrollback. The fixed
Home/Chat/API/Desktop/Store tabs stay visible while any number of shell tabs scroll
horizontally so all of them remain reachable. Shell tabs are restored on
reconnect. Each new shell opens with a short banner pointing to
`sandbox-cli list`, `sandbox-cli <tool>` and `g4f-go`.

A Termux-style key row is pinned below the terminal (Esc, Tab, Ctrl, Alt,
arrows, Home/End, `^C`, `^D`, Select, Copy, All and Paste). The sticky Ctrl/Alt
keys also modify characters typed on the hardware keyboard or through the IME,
so selecting Ctrl and pressing `c` sends `^C`. A long-press selects the word
under the finger and dragging extends the range; in Select mode every drag
selects. A mostly vertical drag scrolls back through the buffer, and a tap
returns focus and opens the keyboard. Copy and All use the Android clipboard.

The **Full** tab switches the current Chat/Desktop page or shell to immersive
fullscreen: the status line, action bar, tab bar and Android system bars are
hidden. A fullscreen shell keeps the key row, so Ctrl/Alt, arrows, Esc and the
clipboard buttons remain available while the tab bar is gone; Chat, Home and
Desktop use the whole stage. Tap **✕ Exit** or press Back to restore the chrome.

## Provider browser automation (CDP)

There is no installable Chrome on Android, so providers that need a browser
attach to the app's own WebView over the abstract Unix socket
`@webview_devtools_remote_<app pid>`. This works in the container as well as in
Classic mode:

- `ContainerActivity` enables WebView debugging (honouring the `WebViewDebug`
  manifest switch) and keeps a hidden control WebView with a `127.0.0.1` base
  URL. It exposes `window.G4FAutomation`, the same bridge the Classic chat page
  provides.
- `ContainerManager.prootEnv()` exports
  `G4F_WEBVIEW_DEVTOOLS_SOCKET=webview_devtools_remote_<app pid>` into every
  PRoot session. The container's Python process has a different pid, so it
  cannot derive the socket name itself.
- The API launcher and shell profile export the same markers, and
  `writeLaunchers()` records the socket name in
  `/g4f-scripts/webview-devtools-socket`. This keeps the container API working
  when a session was started without `prootEnv()`; g4f also probes the live
  socket if a recorded pid became stale after an app restart.
- `g4f.requests.cdp` treats `G4F_CONTAINER=1` as Android, resolves the socket
  from `G4F_WEBVIEW_DEVTOOLS_SOCKET` (or `G4F_WEBVIEW_PID`), and connects with
  `aiohttp.UnixConnector` exactly as in Classic mode. `CDPSession` creates a
  dedicated automation WebView through the bridge, so the container API is not
  hijacked, and closes it again when the session ends.

Automation therefore needs the app UI to be running: the DevTools socket and
the bridge live in `ContainerActivity`. If the activity is destroyed while the
foreground service keeps the API alive, a provider that needs the browser
reports that no WebView target is reachable.

## Why the desktop is optional

A GUI environment adds roughly 300–500 MB and several hundred packages to a
phone. Keeping it opt-in means a typical install only needs `ca-certificates`,
`sudo`, `git`, the `g4f-go` CPython runtime and the API dependencies. The base
and desktop provisioning scripts are separate, so an
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
- The terminal renderer implements the ANSI subset that shells and package
  managers need; full-screen applications that require every terminal extension
  may still render imperfectly.
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
