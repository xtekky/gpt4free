"""Checks for the Android v2 container assets.

The Android app cannot be built in this environment, so these tests guard the
data and scripts that drive its installer and provisioning flow.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "projects" / "android" / "app" / "src" / "main" / "assets" / "container"
CATALOG = ASSETS / "catalog.json"
JAVA = (REPO / "projects" / "android" / "app" / "src" / "main" / "java"
        / "dev" / "g4f" / "android" / "container")
CPP = REPO / "projects" / "android" / "app" / "src" / "main" / "cpp"
ANDROID_PROJECT = REPO / "projects" / "android"
WORKFLOW = REPO / ".github" / "workflows" / "build-packages.yml"
ABIS = ("arm64-v8a", "x86_64")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class AndroidContainerCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    def assert_asset(self, asset, label):
        self.assertTrue(asset["url"].startswith(("https://", "http://")), label)
        self.assertRegex(asset["sha256"], SHA256, label)
        self.assertGreater(asset.get("bytes", 0), 0, label)

    def test_schema_and_distribution(self):
        self.assertEqual(self.catalog["schema"], 1)
        self.assertEqual(self.catalog["distribution"]["id"], "debian-bookworm")
        self.assertIn("Debian", self.catalog["distribution"]["name"])

    def test_rootfs_assets_are_pinned(self):
        rootfs = self.catalog["distribution"]["rootfs"]
        self.assertEqual(set(rootfs), set(ABIS))
        for abi, asset in rootfs.items():
            with self.subTest(abi=abi):
                self.assert_asset(asset, abi)
                commit = asset["url"].split("/")[5]
                self.assertRegex(commit, COMMIT)
                self.assertTrue(asset["url"].endswith("/rootfs.tar.gz"))

    def test_runtime_assets(self):
        runtime = self.catalog["runtime"]
        for name in ("proot", "libtalloc", "libandroid-shmem"):
            packages = runtime[name]["packages"]
            self.assertEqual(set(packages), set(ABIS))
            for abi, asset in packages.items():
                with self.subTest(name=name, abi=abi):
                    self.assert_asset(asset, f"{name}/{abi}")
                    self.assertTrue(asset["url"].endswith(".deb"))
                    self.assertTrue(runtime[name]["license"])

    def test_static_curl_is_pinned_for_sandbox_cli(self):
        curl = self.catalog["runtime"]["curl"]
        self.assertEqual(curl["version"], "8.23.0")
        self.assertEqual(set(curl["packages"]), set(ABIS))
        for abi, asset in curl["packages"].items():
            with self.subTest(abi=abi):
                self.assert_asset(asset, f"curl/{abi}")
                self.assertTrue(asset["url"].endswith(".tar.xz"))
                self.assertRegex(asset["binary_sha256"], SHA256, abi)

        script = (ANDROID_PROJECT / "prepare_curl.py").read_text(encoding="utf-8")
        for needle in ('r:xz', 'getmember("curl")', 'binary_sha256',
                       'write_executable(output / "curl", binary)'):
            with self.subTest(needle=needle):
                self.assertIn(needle, script)

        gradle = (ANDROID_PROJECT / "app" / "build.gradle").read_text(encoding="utf-8")
        for needle in ('prepareCurlBinary', 'prepare_curl.py',
                       "dependsOn 'copyG4fAssets'",
                       'build/curl-bin/${containerPythonAbi}/curl',
                       'rename { "curl" }'):
            with self.subTest(needle=needle):
                self.assertIn(needle, gradle)

    def test_python_runtime_packages_are_pinned(self):
        runtime = self.catalog["runtime"]
        packages = runtime["python"]["packages"]
        self.assertEqual(set(packages), set(ABIS))
        for abi, entries in packages.items():
            with self.subTest(abi=abi):
                names = {entry["name"] for entry in entries}
                self.assertIn("python3.11-minimal", names)
                self.assertIn("libpython3.11-stdlib", names)
                self.assertIn("libexpat1", names)
                self.assertIn("libssl3", names)
                for entry in entries:
                    self.assert_asset(entry, f"python/{abi}/{entry['name']}")

    def test_ports_are_distinct(self):
        g4f = self.catalog["g4f"]
        ports = [g4f["api_port"], g4f["vnc_port"], g4f["novnc_port"]]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertGreater(min(ports), 1024)
        self.assertLess(max(ports), 65536)
        launcher = (ASSETS / "scripts" / "start-api.sh").read_text(encoding="utf-8")
        self.assertIn("export G4F_VNC_PORT=" + str(g4f["vnc_port"]), launcher)

    def test_package_lists(self):
        g4f = self.catalog["g4f"]
        self.assertIn("python3", g4f["base_packages"])
        self.assertIn("python3-venv", g4f["base_packages"])
        # git/curl are not needed by the editable pip install: the version is
        # supplied by G4F_VERSION. curl is bundled as a pinned static binary
        # on /g4f-bin instead, so sandbox-cli has HTTPS without an apt pass.
        self.assertNotIn("git", g4f["base_packages"])
        self.assertNotIn("curl", g4f["base_packages"])
        self.assertIn("xfce4", g4f["desktop_packages"])
        self.assertIn("tigervnc-standalone-server", g4f["desktop_packages"])
        self.assertIn("novnc", g4f["desktop_packages"])
        self.assertIn("dbus-x11", g4f["desktop_packages"])
        self.assertIn("procps", g4f["desktop_packages"])

    def test_prebuilt_overlay_is_optional_and_pinned(self):
        prebuilt = self.catalog["g4f"].get("prebuilt")
        if prebuilt is None:
            self.skipTest("no prebuilt overlay pinned yet")
        self.assert_asset(prebuilt, "prebuilt")
        self.assertIn("/releases/download/", prebuilt["url"])
        self.assertNotIn("/latest/download/", prebuilt["url"])

    def test_desktop_prebuilt_overlay_is_pinned_when_present(self):
        desktop = self.catalog["g4f"].get("desktop_prebuilt")
        if desktop is None:
            self.skipTest("no desktop overlay pinned yet")
        self.assert_asset(desktop, "desktop_prebuilt")
        self.assertIn("/releases/download/", desktop["url"])
        self.assertNotIn("/latest/download/", desktop["url"])

    def test_novnc_release_is_pinned_and_branded_at_build_time(self):
        novnc = self.catalog["runtime"]["novnc"]
        self.assertEqual(novnc["version"], "1.6.0")
        self.assert_asset(novnc, "novnc")
        self.assertTrue(novnc["url"].endswith("/v1.6.0.tar.gz"))

        generator = (ANDROID_PROJECT / "prepare_novnc.py").read_text(encoding="utf-8")
        for needle in ("catalog[\"runtime\"][\"novnc\"]", "asset[\"bytes\"]",
                       "novnc-brand.sh", "novnc-index.html", "g4f-controls.js",
                       "g4f.svg", "UI_FILES"):
            with self.subTest(needle=needle):
                self.assertIn(needle, generator)

        gradle = (ANDROID_PROJECT / "app" / "build.gradle").read_text(encoding="utf-8")
        for needle in ("prepareNovncClient", "prepare_novnc.py", "copyNovncAssets",
                       'outputs.dir("${rootDir}/build/novnc-client")',
                       'from("${rootDir}/build/novnc-client")',
                       'into("src/main/assets/container/scripts/novnc")'):
            with self.subTest(needle=needle):
                self.assertIn(needle, gradle)

        provision = (ASSETS / "scripts" / "provision-desktop.sh").read_text(
            encoding="utf-8")
        self.assertIn("G4F_DESKTOP_NOVNC_ASSETS=/g4f-scripts/novnc", provision)

        installer = (REPO / "bin" / "install-desktop.sh").read_text(encoding="utf-8")
        self.assertIn('[ -f "$NOVNC_ASSETS/vnc.html" ]', installer)
        self.assertIn("rm -rf /usr/share/novnc", installer)

        index = (REPO / "docker" / "novnc-index.html").read_text(encoding="utf-8")
        self.assertIn('href="g4f.svg"', index)

    def test_provisioning_scripts(self):
        scripts = ASSETS / "scripts"
        expected = {
            "provision-base.sh": ["/g4f-src", "/opt/g4f-venv", "g4f-api",
                                  "export G4F_PYTHON=/opt/g4f-venv/bin/python",
                                  "export G4F_VNC_PORT=5901",
                                  "export G4F_NOVNC_DIR=/usr/share/novnc",
                                  "dpkg --configure -a", "apt-get -f install -y",
                                  "write_resolv", "clear_apt_locks",
                                  "--bind 127.0.0.1:1337 --debug --no-reload",
                                  "unset PYTHONHOME",
                                  "for attempt in 1 2 3",
                                  "apt-get update -qq", "exit 1"],
            "provision-desktop.sh": ["xfce4", "dbus-x11", "procps",
                                     "cp /g4f-scripts/g4f-desktop",
                                     "export G4F_PYTHON=/opt/g4f-venv/bin/python",
                                     "G4F_DESKTOP_REPAIR=1",
                                     "G4F_DESKTOP_NOVNC_ASSETS=/g4f-scripts/novnc",
                                     "/g4f-bin/install-desktop.sh"],
            "g4f-desktop": ["Xvnc :1", "dbus-run-session -- startxfce4",
                            "XDG_RUNTIME_DIR=/tmp/g4f-desktop-runtime",
                            "while kill -0 \"$vnc_pid\"",
                            "Xvnc :1 did not start"],
            "start-api.sh": ["g4f-api", "export PYTHONPATH=/g4f-src",
                             "export VIRTUAL_ENV=/opt/g4f-venv",
                             "export G4F_PYTHON=/opt/g4f-venv/bin/python",
                             "export G4F_VNC_PORT=5901",
                             "export G4F_NOVNC_DIR=/usr/share/novnc",
                             "unset PYTHONHOME"],
            "start-desktop.sh": ["g4f-desktop",
                                 "export G4F_PYTHON=/opt/g4f-venv/bin/python",
                                 "unset PYTHONHOME"],
        }
        for name, needles in expected.items():
            with self.subTest(script=name):
                path = scripts / name
                self.assertTrue(path.exists())
                content = path.read_text(encoding="utf-8")
                self.assertTrue(content.startswith("#!/bin/sh"))
                for needle in needles:
                    self.assertIn(needle, content)
        # noVNC is served by the API at /desktop/, so the desktop launcher must
        # not start a second websockify HTTP server.
        self.assertNotIn("websockify",
                         (scripts / "provision-desktop.sh").read_text(encoding="utf-8"))

    def test_native_pty_bindings(self):
        java = (JAVA / "TerminalSession.java").read_text(encoding="utf-8")
        cpp = (CPP / "pty.cpp").read_text(encoding="utf-8")
        methods = set(re.findall(r"native\s+\w+\s+(native\w+)\(", java))
        self.assertTrue(methods)
        for method in methods:
            with self.subTest(method=method):
                self.assertIn(
                    f"Java_dev_g4f_android_container_TerminalSession_{method}", cpp)
        self.assertIn('System.loadLibrary("g4f_pty")', java)
        self.assertIn("add_library(g4f_pty SHARED pty.cpp)",
                      (CPP / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_linkfix_preload(self):
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")
        for needle in ('nativeLibraryDir.getAbsolutePath() + ":/g4f-native"',
                       '"export LD_PRELOAD=/g4f-native/libg4f_linkfix.so; "'):
            with self.subTest(needle=needle):
                self.assertIn(needle, manager)

        cmake = (CPP / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('set(CMAKE_CXX_STANDARD_LIBRARIES "")', cmake)
        self.assertIn("add_library(g4f_linkfix SHARED linkfix.cpp)", cmake)
        self.assertIn("-nostdlib", cmake)
        self.assertIn("-Wl,--unresolved-symbols=ignore-all", cmake)

        shim = (CPP / "linkfix.cpp").read_text(encoding="utf-8")
        self.assertIn('extern "C" int link(const char* oldpath, const char* newpath)', shim)
        self.assertIn('open(newpath, O_WRONLY_VALUE | O_CREAT_VALUE | O_TRUNC_VALUE', shim)

    def test_terminal_bridge(self):
        html = (ASSETS / "terminal.html").read_text(encoding="utf-8")
        for method in ("ensureSession", "newSession", "send", "resize", "close",
                       "closeAll", "copy", "__g4fCopySelection", "selectAll",
                       "__g4fTermWrite", "__g4fTermExit"):
            with self.subTest(method=method):
                self.assertIn(method, html)
        self.assertIn("AndroidTerminal", html)
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        checks = (
            (activity, "public String ensureSession(String kind)"),
            (activity, "public void closeAll()"),
            (activity, "public void copy(String base64)"),
            (activity, "window.closeAll()"),
            (html, "bridge.ensureSession ? bridge.ensureSession(kind)"),
        )
        for source, needle in checks:
            with self.subTest(needle=needle):
                self.assertIn(needle, source)

    def test_container_ui_tabs_and_lifecycle(self):
        html = (ASSETS / "terminal.html").read_text(encoding="utf-8")
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")

        # Chat, API and Desktop are pinned before the scrollable shell tabs, so
        # adding a shell can never hide the Desktop button.
        order = ["id=\"fixed-chat\"", "id=\"fixed-api\"", "id=\"fixed-desktop\""]
        fixed_start = html.index('<div id="fixed-tabs">')
        fixed_end = html.index("</div>", fixed_start)
        dynamic_start = html.index('<div id="tabs">')
        actions_start = html.index('<div id="actions">')
        positions = [html.index(needle) for needle in order]
        self.assertEqual(positions, sorted(positions))
        for position in positions:
            self.assertLess(fixed_start, position)
            self.assertLess(position, fixed_end)
        self.assertLess(fixed_end, dynamic_start)
        self.assertLess(dynamic_start, actions_start)
        self.assertIn("#fixed-tabs { display: flex; gap: 4px; flex: 0 0 auto; }", html)
        self.assertIn('<div id="tabs-scroll"><div id="tabs"></div></div>', html)
        self.assertIn("#tabs-scroll {", html)
        self.assertIn("overflow-x: auto;", html)
        self.assertIn("touch-action: pan-x;", html)
        self.assertIn("#tabs { display: flex; gap: 4px; min-width: max-content; }", html)
        self.assertIn("#actions { display: flex; gap: 4px; flex: 0 0 auto; margin-left: auto; }", html)
        self.assertIn("scrollTabIntoView(active)", html)
        self.assertIn("singletonIds = { api: null, desktop: null }", html)
        self.assertIn("removeSingletonStale(kind)", html)

        # The top button group keeps only Install, Classic and Stop; Chat lives
        # in the pinned tab bar and opens the API's bundled GUI.
        self.assertNotIn('button("Shell"', activity)
        self.assertNotIn('button("API"', activity)
        self.assertNotIn('button("VNC"', activity)
        self.assertNotIn('button("Chat"', activity)
        self.assertIn('id="fixed-chat"', html)
        chat_start = activity.index("private void openChat()")
        chat_end = activity.index("private void openDesktop()", chat_start)
        chat_body = activity[chat_start:chat_end]
        self.assertIn('service.ensureSession("api")', chat_body)
        self.assertIn('http://127.0.0.1:1337/chat/', chat_body)

        # Extra input buttons sit below the terminal, not in the header.
        header_start = html.index('<div id="header">')
        stage_end = html.index('<div id="stage"></div>')
        keys_start = html.index('<div id="keys">')
        self.assertNotIn('id="keys"', html[header_start:stage_end])
        self.assertLess(stage_end, keys_start)
        self.assertIn("header.offsetHeight", html)
        self.assertIn("keys.offsetHeight", html)

        # Closing an overlay hides its Close button as well as the WebView.
        self.assertIn("closeButton.setVisibility(android.view.View.GONE)", activity)
        self.assertIn('http://127.0.0.1:1337/desktop/', activity)
        self.assertNotIn('http://127.0.0.1:6080/', activity)
        # Xvnc/startxfce4 runs as a background session; its PTY must never be
        # rendered as a terminal tab that hides the VNC page.
        self.assertIn("if (label !== 'desktop') createSession(id, label, true);", html)
        self.assertIn("if (kind !== 'desktop') createSession(id, kind, true);", html)
        self.assertIn("} else if (label === 'shell' && !sessions[id]) {", html)
        self.assertIn("createSession(id, 'shell', false);", html)
        new_shell = html.index("window.newSession = function (kind)")
        new_shell_end = html.index("window.ensureSession", new_shell)
        self.assertIn("bridge.newSession(kind)", html[new_shell:new_shell_end])
        self.assertNotIn("ensureSession", html[new_shell:new_shell_end])

        # API and Desktop come up when the container finishes loading, including
        # when an already-installed app reconnects to the foreground service.
        for needle in ("autoStartIfReady()", "autoStartServices()",
                       'ensureSession("api")', "manager.isDesktopInstalled()",
                       "Container stopped", "boolean stopped",
                       "[stopped]"):
            with self.subTest(needle=needle):
                source = html if needle == "[stopped]" else service
                self.assertIn(needle, source)

        # The API runs with --debug so the bundled GUI is mounted too, and it
        # serves the included noVNC client itself.
        for needle in ("G4F_VNC_PORT", "G4F_NOVNC_DIR", "/usr/share/novnc",
                       "writeLaunchers()", "--debug --no-reload",
                       '"unset PYTHONHOME\\n"',
                       "installNovncAssets()",
                       'copyAssetTree("container/scripts/novnc", novnc)'):
            with self.subTest(needle=needle):
                self.assertIn(needle, manager)
        launchers = manager[manager.index("public void writeLaunchers()"):
                            manager.index("private void writeApiLauncher()")]
        self.assertIn("writeShellProfile();", launchers)

    def test_desktop_fullscreen_and_display_launcher(self):
        controls = (REPO / "docker" / "novnc-controls.js").read_text(encoding="utf-8")
        for needle in ("btn-fullscreen", "toggleFullscreen", "AndroidDesktop",
                       "setFullscreen", "nativeFullscreen"):
            with self.subTest(needle=needle):
                self.assertIn(needle, controls)

        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        for needle in ('addJavascriptInterface(new DesktopBridge(), "AndroidDesktop")',
                       "onShowCustomView", "onHideCustomView", "showFullscreen",
                       "hideFullscreen", "setAppChromeVisible(false)",
                       "View.SYSTEM_UI_FLAG_FULLSCREEN"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)

        launcher = (ASSETS / "scripts" / "g4f-desktop").read_text(encoding="utf-8")
        for needle in ("socket=/tmp/.X11-unix/X1", "lock=/tmp/.X1-lock",
                       "kill -0 \"$vnc_pid\"", "Xvnc :1 -geometry 1280x720",
                       "dbus-run-session -- startxfce4",
                       "XDG_RUNTIME_DIR=/tmp/g4f-desktop-runtime",
                       "Xvnc :1 did not start"):
            with self.subTest(needle=needle):
                self.assertIn(needle, launcher)

        # One launcher is used by both the apt setup path and writeLaunchers(),
        # which also runs after a prebuilt overlay or a service reconnect.
        provision = (ASSETS / "scripts" / "provision-desktop.sh").read_text(
            encoding="utf-8")
        self.assertIn("cp /g4f-scripts/g4f-desktop /usr/local/bin/g4f-desktop",
                      provision)
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")
        self.assertIn('copyAssetFile("container/scripts/g4f-desktop", '
                      'new File(bin, "g4f-desktop"))', manager)

    def test_terminal_is_local_and_termux_style(self):
        html = (ASSETS / "terminal.html").read_text(encoding="utf-8")
        for forbidden in ("cdn.jsdelivr.net", "https://", "textarea",
                          "cleanOutput"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, html)
        for needle in ("xterm/xterm.min.css", "xterm/xterm.min.js",
                       "xterm/addon-fit.min.js", "new Terminal(",
                       "new FitAddon.FitAddon()", "ClipboardManager",
                       "bridge.copy(encoded(text))",
                       "sandbox-cli list", "sandbox-cli <tool>", "g4f-go",
                       "welcomeMessage(session)"):
            with self.subTest(needle=needle):
                source = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8") \
                    if needle == "ClipboardManager" else html
                self.assertIn(needle, source)
        for name, digest in (
            ("xterm.min.css", "f7f724aea2bb620a6482bfb8e4bdecfae1152b0c7facef55fbda61f3b6cfedb2"),
            ("xterm.min.js", "4196e242ef1cf4c2adead8d97f4a772a69576076f70b095e004b4abbb049e7bf"),
            ("addon-fit.min.js", "a6a7bbb33569f16aa3e18d71425e34d035fc89a0b7e8cba084f8855f91aa38f1"),
        ):
            with self.subTest(asset=name):
                self.assertEqual(
                    hashlib.sha256((ASSETS / "xterm" / name).read_bytes()).hexdigest(),
                    digest,
                )
        for key in ("esc", "tab", "ctrl", "alt", "left", "up", "down", "right",
                    "home", "end", "ctrlc", "ctrld"):
            with self.subTest(key=key):
                self.assertIn('data-key="' + key + '"', html)
        self.assertIn("applyModifiers", html)

    def test_terminal_selection_modifiers_and_scrolling(self):
        html = (ASSETS / "terminal.html").read_text(encoding="utf-8")
        for needle in ("session.selection", "term.onSelectionChange",
                       "installTouchSelection(session)", "touchmove",
                       "new MouseEvent", "stopPropagation()",
                       "mouse('mousedown', anchor, element)",
                       "mouse('mousemove', p, element)",
                       "mouse('mouseup', last || start, element)",
                       "near(p, range.focus)", "range = { anchor: start, focus:",
                       'onmousedown="event.preventDefault(); return false;"',
                       "touch-action: pan-y", "overscroll-behavior: contain",
                       "visualViewport", "requestAnimationFrame(relayout)",
                       "session.term.rows !== m.rows"):
            with self.subTest(needle=needle):
                self.assertIn(needle, html)
        # xterm only listens on its own element; document events never reach it.
        self.assertNotIn("mouse('mousemove', p, document)", html)
        self.assertNotIn("mouse('mouseup', last || start, document)", html)
        # The sticky on-screen Ctrl/Alt keys must also apply to characters
        # typed on the hardware keyboard (Ctrl+c => 0x03), not just key buttons.
        modifier_block = html[html.index("function applyModifiers")
                              :html.index("function clearModifiers")]
        self.assertIn("applyModifiers(data, null)", html)
        self.assertIn("char.toUpperCase().charCodeAt(0) & 0x1f", modifier_block)

    def test_native_build_cache_is_not_committed(self):
        ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/projects/android/app/.cxx/", ignore)
        for name in ("CMakeLists.txt", "pty.cpp", "linkfix.cpp"):
            self.assertTrue((CPP / name).is_file(), name)
        cmake = (CPP / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("add_library(g4f_pty SHARED pty.cpp)", cmake)
        self.assertIn("add_library(g4f_linkfix SHARED linkfix.cpp)", cmake)

    def test_proot_is_shipped_as_native_libraries(self):
        # Android forbids execve() from app-writable storage, so PRoot and its
        # static loader must come from the APK's read-only nativeLibraryDir.
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")
        for needle in ('"libproot.so"', '"libproot_loader.so"',
                       '"PROOT_LOADER="', '"PROOT_TMP_DIR="', 'nativeLibraryDir'):
            with self.subTest(needle=needle):
                self.assertIn(needle, manager)
        self.assertNotIn('new File(runtime, "bin/proot")', manager)
        # Debian images contain absolute symlinks; they must be rewritten to
        # relative links so extraction cannot escape the rootfs.
        self.assertIn('relativeLink(name, entry.getLinkName())', manager)
        self.assertNotIn('getCanonicalPath()', manager)
        # Debian archives start with a "./" root marker; it must be skipped.
        self.assertIn('name == null || name.isEmpty()', manager)
        # Debian packages contain escaped filenames with a literal backslash.
        # It is data on Android, not a path separator, so extraction must not
        # reject it as unsafe.
        self.assertIn('system-systemd\\x2dcryptsetup.slice', manager)
        self.assertIn('if (name == null || name.isEmpty() || name.startsWith("/")) {', manager)
        # Any other unusual entry is skipped with a visible message rather than
        # aborting the whole Desktop overlay install.
        self.assertIn('Log.w("g4f-container", "Skipping unsafe tar entry:', manager)
        self.assertNotIn('throw new IOException("Unsafe tar entry:', manager)
        # Classic mode owns filesDir/app; the container needs its own source
        # tree so a pip install still finds setup.py after Classic extraction.
        self.assertIn('return new File(base, "g4f-src")', manager)
        # assets/g4f holds package contents; wrap them so import g4f works.
        self.assertIn('File packageDir = new File(appDir, "g4f")', manager)
        self.assertIn('copyAssetTree("g4f", packageDir)', manager)
        self.assertIn('"setup.py", "README.md", "requirements.txt"', manager)
        self.assertIn('"PYTHONPATH=/g4f-src"', manager)
        self.assertIn('"G4F_PYTHON=/opt/g4f-venv/bin/python"', manager)
        self.assertIn('"PYTHONHOME="', manager)
        self.assertIn('"export G4F_PYTHON=/opt/g4f-venv/bin/python; "', manager)
        self.assertIn('"export G4F_PYTHON=/opt/g4f-venv/bin/python\\n"', manager)
        self.assertIn('"unset PYTHONHOME\\n"', manager)
        self.assertIn('copyAssetFile("container/bin/g4f-go"', manager)
        # Re-extracting over a half-configured rootfs is what caused repeated
        # apt exit 100; every install starts from a clean filesystem.
        self.assertIn("            reset();\n            File rootfsArchive", manager)
        # Optional prebuilt overlay: no apt/pip pass when pinned in the catalog.
        for needle in ('boolean prebuilt = catalog.prebuilt != null;',
                       'download(catalog.prebuilt, overlayArchive, listener)',
                       'extractTarGz(overlayArchive, rootfs, "")',
                       'writeFile(baseMarker, "base provisioning complete\\n")',
                       'copyAssetTree("container/python", rootfs)',
                       'copyAssetTree("container/bin", binDir)',
                       '"PATH=/g4f-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"',
                       'resolvConf.getAbsolutePath() + ":/etc/resolv.conf"',
                       'writeShellProfile()',
                       'writeApiLauncher()',
                       '" --debug --no-reload\\n"',
                       'etc/profile.d',
                       'desktopPrebuilt',
                       'catalog.desktopPrebuilt.bytes'):
            with self.subTest(needle=needle):
                self.assertIn(needle, manager)

        script = (ANDROID_PROJECT / "prepare_proot.py").read_text(encoding="utf-8")
        self.assertIn('"libproot.so": "bin/proot"', script)
        self.assertIn('"libproot_loader.so": "libexec/proot/loader"', script)
        self.assertIn('expected_sha256', script)

        gradle = (ANDROID_PROJECT / "app" / "build.gradle").read_text(encoding="utf-8")
        self.assertIn('prepareProotLibs', gradle)
        self.assertIn("prepare_proot.py", gradle)
        self.assertIn('preparePythonRuntime', gradle)
        self.assertIn("prepare_python.py", gradle)
        self.assertIn("'--abi', containerPythonAbi", gradle)
        self.assertIn('project.findProperty("containerPythonAbi") ?: "arm64-v8a"', gradle)
        self.assertIn('copyContainerBinAssets', gradle)
        self.assertIn('g4f-go/dist/linux-arm64/g4f-go', gradle)
        self.assertIn('preBuild.dependsOn', gradle)
        # PRoot must be executable on disk, so AGP has to extract native libs.
        self.assertIn('useLegacyPackaging true', gradle)

    def test_release_publishes_termux_g4f_go_asset(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for needle in ('go-version: "1.23"',
                       'name: g4f-go-android-arm64',
                       'g4f-go-*-android-arm64.zip',
                       'Android/Termux arm64'):
            with self.subTest(needle=needle):
                self.assertIn(needle, workflow)

    def test_release_android_sdk_setup_avoids_removed_tools_package(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        setup = workflow[workflow.index("      - name: Install Android NDK and CMake"):
                         workflow.index("      - name: Decode and verify release keystore")]
        self.assertIn("uses: android-actions/setup-android@v4.0.4", setup)
        self.assertIn("packages: 'platform-tools'", setup)
        self.assertNotIn("tools platform-tools", setup)
        self.assertIn('sdkmanager "ndk;26.3.11579264" "cmake;3.22.1"', setup)

    def test_failed_provisioning_is_visible(self):
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        for needle in ('private volatile String provisioningError;',
                       'return provisioningError;',
                       'tap Install to retry',
                       'isSetupRunning()',
                       'Setup is still running; it cannot be closed',
                       'sessionLabels.put(id, label)',
                       'if (!manager.isProvisioned()) {',
                       'provisionDesktopOnly()',
                       'isProvisioningRunning()',
                       'Desktop setup failed'):
            with self.subTest(needle=needle):
                self.assertIn(needle, service)


if __name__ == "__main__":
    unittest.main()
