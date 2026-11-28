"""Checks for the Android v2 container assets.

The Android app cannot be built in this environment, so these tests guard the
data and scripts that drive its installer and provisioning flow.
"""

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
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")
        for method in ("public interface Listener", "void onInput(int sessionId, byte[] data)",
                       "public synchronized void write(byte[] data)",
                       "public void sendText(String text)", "public void sendKey(String name)",
                       "public String getSelectedText()", "public void selectAll()",
                       "public boolean copySelection()", "public void toggleSelectionMode()",
                       "public void showBanner()", "public void showProcessExit("):
            with self.subTest(method=method):
                self.assertIn(method, view)
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        for needle in ("new TerminalView(this)", "terminalListener",
                       "service.writeBytes(sessionId, data)",
                       "service.resize(sessionId, cols, rows)",
                       "service.close(activeTerminal.sessionId())",
                       "service.closeAllSessions()"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)
        self.assertNotIn("AndroidTerminal", activity)
        self.assertNotIn("file:///android_asset/container/terminal.html", activity)
        self.assertNotIn("addJavascriptInterface(new TerminalBridge", activity)
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        self.assertIn("public void writeBytes(int sessionId, byte[] data)", service)

    def test_install_confirm_and_busy_guard(self):
        # A second Install tap during provisioning used to reset the rootfs
        # (and restart the 60 MB runtime download), which looked like a hang.
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        self.assertIn("// Refresh after every setup/desktop-setup exit.", service)
        for needle in ("public boolean isBusy()", "public boolean isReady()",
                       "installing || isProvisioningRunning()"):
            with self.subTest(needle=needle):
                self.assertIn(needle, service)
        for needle in ("private void confirmInstall(final boolean withDesktop)",
                       "if (service.isBusy())", "if (service.isReady())",
                       "Reinstall container?",
                       "confirmInstall(desktopCheckBox.isChecked())"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)

    def test_provisioning_output_is_logged(self):
        # A download that stalls on a phone must leave evidence behind, and a
        # failed script must keep its own exit status (pipefail, not tee's).
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        self.assertIn("private static String loggedScript(String script)", service)
        self.assertIn("bash -o pipefail -c", service)
        self.assertIn('+ " 2>&1 | tee -a " + log', service)
        self.assertIn('loggedScript("provision-base.sh")', service)
        self.assertIn('loggedScript("provision-desktop.sh")', service)
        self.assertIn("provision-base.log", service)

    def test_container_webview_cdp_bridge(self):
        # Provider automation in the container attaches to the app's WebView
        # DevTools socket instead of trying to launch Chrome.
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        cdp = (REPO / "g4f" / "requests" / "cdp.py").read_text(encoding="utf-8")
        self.assertIn("G4F_WEBVIEW_DEVTOOLS_SOCKET=webview_devtools_remote_", manager)
        self.assertIn("android.os.Process.myPid()", manager)
        # The API launcher and the shell profile must carry the hand-off even
        # when a session was started without prootEnv(), and the scripts dir
        # gets a socket file for sessions that lose the environment entirely.
        self.assertIn('"export G4F_CONTAINER=1\\n" +', manager)
        self.assertIn(
            '"export G4F_WEBVIEW_DEVTOOLS_SOCKET=webview_devtools_remote_"',
            manager,
        )
        self.assertIn('new File(scripts, "webview-devtools-socket")', manager)
        for needle in ("G4FAutomation",
                       "setWebContentsDebuggingEnabled(isWebViewDebugEnabled())",
                       'loadDataWithBaseURL("http://127.0.0.1:1337/"',
                       "createAutomationWebView", "destroyAutomationWebView",
                       "addJavascriptInterface(new AutomationBridge()"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)
        for needle in ("G4F_WEBVIEW_DEVTOOLS_SOCKET", "G4F_WEBVIEW_PID",
                       "G4F_CONTAINER"):
            with self.subTest(needle=needle):
                self.assertIn(needle, cdp)

    def test_container_ui_tabs_and_lifecycle(self):
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        manager = (JAVA / "ContainerManager.java").read_text(encoding="utf-8")
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")

        # Home, Chat, API and Desktop are pinned before the scrollable shell
        # tabs, so adding a shell can never hide the navigation buttons.
        order = ['addFixedButton("home"', 'addFixedButton("chat"',
                 'addFixedButton("api"', 'addFixedButton("desktop"']
        positions = [activity.index(needle) for needle in order]
        self.assertEqual(positions, sorted(positions))
        fixed_end = activity.index('tabBar.addView(tabButton("+ Shell"')
        self.assertLess(positions[-1], fixed_end)
        self.assertIn("tabBar.addView(tabButton(\"+ Shell\"", activity)
        self.assertIn("tabBar.addView(tabButton(\"Close\"", activity)
        self.assertIn("tabBar.addView(tabButton(\"Close All\"", activity)
        self.assertIn("tabScroller = new HorizontalScrollView(this);", activity)
        self.assertIn("tabScroller.setHorizontalScrollBarEnabled(false);", activity)
        self.assertIn("scrollActiveTabIntoView()", activity)
        self.assertIn("tabScroller.smoothScrollTo(", activity)
        self.assertIn("styleTabs()", activity)

        # The top button group keeps only Install, Classic and Stop; Chat lives
        # in the pinned tab bar and opens the API's bundled GUI.
        self.assertNotIn('button("Shell"', activity)
        self.assertNotIn('button("API"', activity)
        self.assertNotIn('button("VNC"', activity)
        self.assertNotIn('button("Chat"', activity)
        home_start = activity.index("private void openHome()")
        home_end = activity.index("private void openChat()", home_start)
        home_body = activity[home_start:home_end]
        self.assertIn('service.ensureSession("api")', home_body)
        self.assertIn('http://127.0.0.1:1337/', home_body)
        chat_start = activity.index("private void openChat()")
        chat_end = activity.index("private void openDesktop()", chat_start)
        chat_body = activity[chat_start:chat_end]
        self.assertIn('service.ensureSession("api")', chat_body)
        self.assertIn('http://127.0.0.1:1337/chat/', chat_body)

        # Extra input buttons sit below the terminal stage, not in the tab bar.
        stage_add = activity.index("root.addView(stage, new LinearLayout.LayoutParams(")
        key_add = activity.index("root.addView(keyRow);")
        self.assertLess(stage_add, key_add)
        key_block = activity[activity.index("private View buildKeyRow()"):
                             activity.index("private Button keyButton(")]
        for name in ("esc", "tab", "ctrl", "alt", "left", "up", "down", "right",
                     "home", "end", "ctrlc", "ctrld", "select", "copy", "all",
                     "paste"):
            with self.subTest(key=name):
                self.assertIn('"' + name + '"', key_block)

        # Closing an overlay hides its Close button as well as the WebView.
        self.assertIn("closeButton.setVisibility(View.GONE)", activity)
        self.assertIn('http://127.0.0.1:1337/desktop/', activity)
        self.assertNotIn('http://127.0.0.1:6080/', activity)
        # Xvnc/startxfce4 runs as a background session; its PTY must never be
        # rendered as a terminal tab that hides the VNC page.
        self.assertIn('if ("desktop".equals(label)) continue;', activity)
        self.assertIn("private void newShell()", activity)
        self.assertIn('service.newSession("shell")', activity)
        self.assertIn("private void showApi()", activity)
        self.assertIn("private void selectSession(int id)", activity)
        self.assertIn("private void syncSessions()", activity)

        # API and Desktop come up when the container finishes loading, including
        # when an already-installed app reconnects to the foreground service.
        for needle in ("autoStartIfReady()", "autoStartServices()",
                       'ensureSession("api")', "manager.isDesktopInstalled()",
                       "Container stopped", "boolean stopped"):
            with self.subTest(needle=needle):
                self.assertIn(needle, service)
        self.assertIn("[stopped]", view)
        self.assertIn("showProcessExit", view)

        # The API runs with --debug so the bundled GUI is mounted too, and it
        # serves the included noVNC client itself.
        for needle in ("G4F_VNC_PORT", "G4F_NOVNC_DIR", "/usr/share/novnc",
                       "writeLaunchers()", "--debug --no-reload",
                       '"unset PYTHONHOME G4F_PYTHON VIRTUAL_ENV PYTHONPATH\\n"',
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

    def test_fullscreen_for_chat_and_shell(self):
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        for needle in ('tabButton("Full"', "private void toggleFullscreen()",
                       "private void updateChromeVisibility()",
                       "private void setNativeFullscreen(boolean enabled)",
                       "fullscreenExitButton", "View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY",
                       "keyRow.setVisibility(View.VISIBLE)",
                       "setNativeFullscreen(false)"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)

        # The Full control sits with the fixed tabs, before the shell actions.
        desktop_button = activity.index('addFixedButton("desktop"')
        full_button = activity.index('tabButton("Full"')
        shell_button = activity.index('tabButton("+ Shell"')
        self.assertLess(desktop_button, full_button)
        self.assertLess(full_button, shell_button)

        # Chat/desktop hide every row; a fullscreen shell keeps its key row.
        chrome = activity[activity.index("private void updateChromeVisibility()"):
                          activity.index("/** Fallback for WebViews")]
        self.assertIn("activeTerminal != null && !contentVisible()", chrome)
        self.assertIn("keyRow.setVisibility(View.VISIBLE)", chrome)
        self.assertIn("setAppChromeVisible(false)", chrome)

        # Back leaves fullscreen before leaving the activity or the page.
        back = activity[activity.index("public void onBackPressed()"):
                        activity.index("protected void onDestroy()")]
        self.assertIn("if (nativeFullscreen)", back)
        self.assertIn("setNativeFullscreen(false)", back)

        # The exit overlay is brought back above the terminal/content view.
        self.assertIn("fullscreenExitButton.bringToFront()", activity)
        self.assertIn('fullscreenExitButton.setText("✕ Exit")', activity)

    def test_native_terminal_is_local_and_termux_style(self):
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")

        for forbidden in ("cdn.jsdelivr.net", "https://", "WebView",
                          "new Terminal("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, view)
        for needle in ("decodeByte", "charWidth", "handleCsi", "applySgr",
                       "ansi256", "drawText", "Typeface.MONOSPACE",
                       "scrollback", "putCodePoint", "cursorVisible",
                       "sandbox-cli list", "sandbox-cli <tool>", "g4f-go",
                       "showBanner"):
            with self.subTest(needle=needle):
                self.assertIn(needle, view)
        for needle in ("ClipboardManager", "setPrimaryClip", "copyToClipboard",
                       "InputMethodManager", "new TerminalView(this)",
                       "buildKeyRow"):
            with self.subTest(needle=needle):
                self.assertIn(needle, activity)

        # A useful VT100/ANSI subset: SGR colours, cursor addressing, erase and
        # truecolour, which bash, apt and the bundled tools all rely on.
        for sequence in ("case 'm':", "case 'H':", "case 'J':", "case 'K':",
                         "case 38:", "case 48:", "mode == 1049"):
            with self.subTest(sequence=sequence):
                self.assertIn(sequence, view)

        # Termux-style keys are real Android buttons.
        self.assertIn("keys.addView(keyButton(", activity)
        self.assertIn("keyButton(\"Esc\", \"esc\")", activity)
        self.assertIn("keyButton(\"Tab\", \"tab\")", activity)
        self.assertIn("keyButton(\"^C\", \"ctrlc\")", activity)
        self.assertIn("keyButton(\"^D\", \"ctrld\")", activity)

    def test_native_terminal_selection_modifiers_and_scrolling(self):
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        for needle in ("startLongPressSelection", "draggingSelection",
                       "draggingScroll", "scrollOffset", "getSelectedText",
                       "copySelection()", "selectAll()", "isSelected(",
                       "toggleSelectionMode", "selAnchorLine", "selFocusLine",
                       "longPress", "Math.abs(dx) > Math.abs(dy) * 1.5f",
                       "scrollAccum", "cellHeight", "performClick()",
                       "onTouchEvent", "selectionMode"):
            with self.subTest(needle=needle):
                self.assertIn(needle, view)
        # The sticky Ctrl/Alt keys must also apply to characters typed on the
        # hardware keyboard or through the IME (Ctrl+c => 0x03).
        for needle in ("public void sendText(String text)", "c & 0x1f",
                       "\\u001b[1;5", "\\u001b[1;3",
                       "public void sendKey(String name)"):
            with self.subTest(needle=needle):
                self.assertIn(needle, view)
        self.assertIn("void onCopy(String text)", view)
        self.assertIn("copyToClipboard(text)", activity)
        self.assertIn("\"copy\".equals(name)", activity)
        self.assertIn("\"all\".equals(name)", activity)
        self.assertIn("showSoftInput(view", activity)

    def test_android_terminal_has_draggable_selection_pips_and_normal_scroll(self):
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")

        # A selection shows two pips that can be grabbed and moved independently.
        for needle in ("draggingHandle", "handleDragsAnchor", "endpointIsAnchor",
                       "handleAt(float x, float y)", "nearHandle(", "moveHandle(",
                       "drawSelectionHandles(Canvas canvas)", "drawHandle(",
                       "handlePaint", "handleTouchRadius", "selectionBounds()",
                       "autoScrollSelection(float y)"):
            with self.subTest(needle=needle):
                self.assertIn(needle, view)

        # Double tapping a word is a quicker way to start a selection than
        # holding the long-press timeout.
        self.assertIn("DOUBLE_TAP_TIMEOUT_MS", view)
        self.assertIn("boolean doubleTap = now - lastTapTime", view)

        # Natural scrolling: dragging down reveals older lines. The old
        # inverted accumulator must not come back.
        self.assertIn("scrollAccum += y - lastTouchY", view)
        self.assertNotIn("scrollAccum += lastTouchY - y", view)

    def test_android_terminal_selection_opens_context_menu(self):
        view = (JAVA / "TerminalView.java").read_text(encoding="utf-8")

        for needle in ("import android.widget.PopupWindow;",
                       "private PopupWindow selectionMenu;",
                       "private void showSelectionMenu()",
                       "private void dismissSelectionMenu()",
                       "new PopupWindow(content,",
                       'menuButton("Copy", () ->',
                       'menuButton("Select all", () ->',
                       "menu.setOutsideTouchable(true)",
                       "menu.showAtLocation(this, Gravity.TOP | Gravity.START",
                       "if (up && getSelectedText() != null) showSelectionMenu();"):
            with self.subTest(needle=needle):
                self.assertIn(needle, view)

        # Gestures must offer actions instead of silently writing the clipboard.
        self.assertNotIn("if (getSelectedText() != null) copySelection();", view)
        # The key-row Copy button remains an explicit way to copy.
        self.assertIn("public boolean copySelection()", view)

    def test_app_store_lists_and_installs_sandbox_tools(self):
        activity = (JAVA / "ContainerActivity.java").read_text(encoding="utf-8")
        service = (JAVA / "ContainerService.java").read_text(encoding="utf-8")
        store = (JAVA / "StoreView.java").read_text(encoding="utf-8")

        # The store is a pinned tab beside Home/Chat/API/Desktop, not a shell
        # command the user has to discover in the terminal banner.
        self.assertIn('addFixedButton("store", "Store", v -> openStore())', activity)
        for needle in ("private void openStore()", "private void ensureStoreView()",
                       "private void reloadStore()",
                       "private void installStoreTool(String tool)",
                       "private void openBinCommand(String command)",
                       "private boolean storeVisible()", "private void hideStore()",
                       "new StoreView.Listener()", "public void onOpenCommand(String command)",
                       'label.startsWith("store-")', 'label.startsWith("bin-")',
                       "if (storeVisible())", "storeView.setVisibility(View.GONE)"):
            with self.subTest(activity=needle):
                self.assertIn(needle, activity)

        # The catalog is read from the container, so new sandbox-cli tools show
        # up without an APK update. Installs run in a visible PTY session.
        for needle in ("public interface StoreListCallback",
                       "public void listStoreTools(",
                       "public int installStoreTool(String tool)",
                       'manager.prootArgv("sandbox-cli list")',
                       "captureSessions", "SAFE_TOOL.matcher(tool).matches()",
                       '"sandbox-cli install " + tool + " </dev/null; exec bash -l"',
                       'startSession("store-" + tool,',
                       "SAFE_BIN = Pattern.compile", "public int openBinCommand(String command)",
                       "SAFE_BIN.matcher(command).matches()",
                       'startSession("bin-" + command, "exec bash -l")',
                       'writeBytes(id, (command + " ").getBytes(StandardCharsets.UTF_8))'):
            with self.subTest(service=needle):
                self.assertIn(needle, service)

        # Every listed id must pass the shell-command validation in the service.
        source = (REPO / "bin" / "sandbox-cli").read_text(encoding="utf-8")
        listing = source[source.index("list_tools() {"):source.index("trap publish_user_bins")]
        listed = re.findall(r"^\s{8}([a-z0-9-]+)\s+\S", listing, re.MULTILINE)
        pattern = re.search(r'SAFE_TOOL = Pattern\.compile\("([^"]+)"\)', service)
        self.assertIsNotNone(pattern)
        self.assertTrue(listed)
        for tool in listed:
            with self.subTest(tool=tool):
                self.assertRegex(tool, "^" + pattern.group(1) + "$")

        for needle in ("Sandbox App Store", "public int getToolsCount()",
                       "public void showTools(String output)",
                       "static List<Tool> parse(String output)",
                       'search.setHint("Search tools")',
                       'badge.setText("built in")',
                       "install.setOnClickListener(v -> listener.onInstall(id))",
                       'BIN_GROUP = "g4f-bin commands"',
                       'terminal.setText("Terminal")',
                       "listener.onOpenCommand(command)"):
            with self.subTest(store=needle):
                self.assertIn(needle, store)

        # The listing must append the container's whole /g4f-bin inventory.
        for needle in ("list_bin_tools", "G4F_BIN_DIR", "g4f-bin commands",
                       "bin_command_description"):
            with self.subTest(sandbox_cli=needle):
                self.assertIn(needle, source)

        android_doc = (REPO / "docs" / "android-container.md").read_text(
            encoding="utf-8")
        self.assertIn("**Store** reads `sandbox-cli list`", android_doc)
        self.assertIn("g4f-bin commands", android_doc)

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
        self.assertIn('"G4F_CONTAINER=1"', manager)
        self.assertIn('"PYTHONHOME="', manager)
        self.assertIn('"exec /g4f-bin/g4f-go api --bind 127.0.0.1:"', manager)
        self.assertNotIn('g4f-venv', manager)
        # The old venv image exported these; launchers and the shell profile
        # must clear them so g4f-go uses its own runtime.
        self.assertIn('"unset PYTHONHOME G4F_PYTHON VIRTUAL_ENV PYTHONPATH\\n"', manager)
        self.assertNotIn('copyAssetTree("container/python"', manager)
        self.assertNotIn('PYTHONPATH=/g4f-src', manager)
        self.assertIn('copyAssetFile("container/bin/g4f-go"', manager)
        # Re-extracting over a half-configured rootfs is what caused repeated
        # apt exit 100; every install starts from a clean filesystem.
        self.assertIn("            reset();\n            File rootfsArchive", manager)
        # Optional prebuilt overlay: no apt/pip pass when pinned in the catalog.
        for needle in ('boolean prebuilt = catalog.prebuilt != null;',
                       'download(catalog.prebuilt, overlayArchive, listener)',
                       'extractTarGz(overlayArchive, rootfs, "")',
                       'writeFile(baseMarker, "base provisioning complete\\n")',
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
        self.assertNotIn('preparePythonRuntime', gradle)
        self.assertNotIn("prepare_python.py", gradle)
        self.assertNotIn('container/python', gradle)
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
