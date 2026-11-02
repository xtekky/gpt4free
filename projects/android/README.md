# G4F Android App

Native Android app embedding the g4f Python server via [Chaquopy](https://chaquo.com/chaquopy).
The Flask GUI (`g4f.gui`) runs in-process and a WebView loads `http://127.0.0.1:1337/chat/`.

## Install (prebuilt APK)

Download `g4f-<version>.apk` (e.g. `g4f-v8.5.9.apk`) from the
[releases page](https://github.com/xtekky/gpt4free/releases) — the APK version
matches the release tag — then:

```bash
adb install -r g4f-v8.5.9.apk
```

Or copy the APK to a phone and open it (enable "Install unknown apps").

Release APKs are signed with the build runner's debug key, so a locally built
APK (signed with your own `~/.android/debug.keystore`) cannot be installed over
one. Android reports this as "App not installed". Uninstall the existing app
once, then install the local build:

```bash
adb uninstall dev.g4f.android
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Installs coming from the same source keep working afterwards (`adb install -r`).
The same applies in reverse when switching back to a release APK.

Watch startup logs:

```bash
adb logcat | grep -E "g4f-bootstrap|python.std"
```

If startup fails, the WebView shows the error message instead of a white screen.

## Build from source

Toolchain expected in `~/android-tools` (JDK 17, Gradle 8.7, Android SDK 34):

```bash
export JAVA_HOME=~/android-tools/jdk
export ANDROID_HOME=~/android-tools/android-sdk
cd android
~/android-tools/gradle-8.7/bin/gradle assembleDebug --no-daemon
# output: app/build/outputs/apk/debug/app-debug.apk
```

The build checks that Brotli is included in the APK for both bundled ABIs.
To check an existing APK: `python3 check_brotli.py app/build/outputs/apk/debug/app-debug.apk`.

## Signing key

Android only installs an APK that is signed with the same key as the installed
app. Without a configured key every build (local and CI) uses a throwaway debug
key, so switching builds always requires an uninstall first.

Create one key and reuse it everywhere:

```bash
cd projects/android
./create-keystore.sh              # prompts for a password, default ~/keystores/g4f-release.jks
base64 -w0 ~/keystores/g4f-release.jks   # paste into a GitHub secret
```

Or manually (same password for store and key — PKCS12 keystores, the JDK
default, cannot hold two different ones):

```bash
keytool -genkeypair -v -keystore ~/keystores/g4f-release.jks -alias g4f \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -storepass '<password>' -keypass '<password>' -dname "CN=G4F, O=gpt4free, C=DE"
```

`keytool: keystore password was incorrect` when *creating* a key means the
target file already exists and is opened with a different password. Remove the
leftover file (only if nothing was signed with it yet) or point at a new path;
`keytool -list -keystore <file>` tells you whether the password works.

Keep the keystore out of the repository (`*.jks` / `*.keystore` are ignored).
Then either export the values locally:

```bash
export G4F_KEYSTORE=~/keystores/g4f-release.jks
export G4F_KEYSTORE_PASSWORD=...
export G4F_KEY_ALIAS=g4f                   # optional, defaults to "g4f"
export G4F_KEY_PASSWORD=...                # optional, defaults to the store password
./gradlew assembleRelease                   # or: ../../android-tools/gradle-8.7/bin/gradle assembleRelease
```

or pass `-Pg4fKeystore=... -Pg4fKeystorePassword=... -Pg4fKeyAlias=...`.

For CI, add these repository secrets and `build-android` uses them
automatically when `ANDROID_KEYSTORE_BASE64` is present:

| Secret | Value |
| --- | --- |
| `ANDROID_KEYSTORE_BASE64` | base64 of `g4f-release.jks` |
| `ANDROID_KEYSTORE_PASSWORD` | store password |
| `ANDROID_KEY_ALIAS` | key alias |
| `ANDROID_KEY_PASSWORD` | key password |

Without a keystore the build falls back to the debug key, which keeps plain
`assembleDebug` working. Builds signed with a different key than the installed
app still need `adb uninstall dev.g4f.android` once.

The CI step *Decode and verify release keystore* validates the secrets before
building and names the failing one (`ANDROID_KEYSTORE_PASSWORD`,
`ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`). Check the same thing locally:

```bash
# store password opens the keystore?
keytool -list -keystore ~/keystores/g4f-release.jks -storepass "$G4F_KEYSTORE_PASSWORD"
# key alias + key password can read the private key?
keytool -certreq -alias g4f -keystore ~/keystores/g4f-release.jks \
  -storepass "$G4F_KEYSTORE_PASSWORD" -keypass "$G4F_KEY_PASSWORD" >/dev/null && echo key OK
```

`KeytoolException: Failed to read key <alias> ... keystore password was
incorrect` means the key password does not match the store (PKCS12 stores
verify both together). Regenerate with `create-keystore.sh`, which sets both to
the same value, and update all secrets.

## Architecture

```
MainActivity (Java)
 ├─ extracts assets/g4f + assets/g4.dev → filesDir/app   (versioned .extracted-N marker)
 ├─ starts Chaquopy Python 3.12
 └─ bootstrap.main(app_root, 1337)  →  g4f.gui Flask server on 127.0.0.1:1337
WebView polls /chat/ until 200, then loads it (2-thread executor: server + poller)
```

## Notes

- `chaquopy {}` block is top-level in `app/build.gradle` (Chaquopy 17 DSL)
- Only arm64-v8a and x86_64 ABIs are bundled
- `assembleRelease` uses the optional stable key (see [Signing key](#signing-key)),
  falling back to the debug key; `assembleDebug` always uses the debug key
- Asset extraction version: bump `EXTRACTION_VERSION` in `MainActivity.java` when bundled assets change
- Startup polls the page it displays (`/chat/`) on a fixed deadline and shows the
  failure (with the HTTP status) instead of spinning forever; the bundled
  `bootstrap.py` sets `G4F_NO_VERSION_CHECK=1` so rendering never waits on PyPI

## Media & clipboard support

- **Image/file upload**: `WebChromeClient.onShowFileChooser` opens the Android picker
- **Mic / camera**: `onPermissionRequest` → runtime permission dialog → `request.grant()`
- **Clipboard**: `navigator.clipboard.writeText/readText` are patched on page load to use a
  native JS bridge (`window.AndroidClipboard` → `ClipboardManager`), so copy/paste works
  reliably inside the WebView
- **Text selection**: long-press works natively; WebView text selection menu is enabled by default

## Provider browser automation (CDP)

Providers that need a browser (e.g. Cloudflare) use `CDPSession`. On Android there is no
installable Chrome, so `CDPSession` attaches to the app's own **WebView**:

- `MainActivity` enables `WebView.setWebContentsDebuggingEnabled(true)`, which exposes the
  DevTools abstract socket `@webview_devtools_remote_<pid>`
- `CDPSession` (mode `webview`, auto-detected on Android or forced with `G4F_BROWSER_MODE=webview`)
  connects to that socket with an aiohttp `UnixConnector` and speaks plain CDP
- On close, the session navigates the WebView back to the page it showed before automation
- Limitation: there is only one WebView, so sessions cannot run in parallel tabs and
  automation navigation is visible in the app UI
