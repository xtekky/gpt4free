# G4F Android App

Native Android app embedding the g4f Python server via [Chaquopy](https://chaquo.com/chaquopy).
The Flask GUI (`g4f.gui`) runs in-process and a WebView loads `http://127.0.0.1:1337/chat/`.

## Install (prebuilt APK)

```bash
adb install -r ../g4f-android-debug.apk   # from repo root: adb install -r g4f-android-debug.apk
```

Or copy `g4f-android-debug.apk` to a phone and open it (enable "Install unknown apps").

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
- Debug-signed; for release, add a signing config and run `assembleRelease`
- Asset extraction version: bump `EXTRACTION_VERSION` in `MainActivity.java` when bundled assets change

## Media & clipboard support

- **Image/file upload**: `WebChromeClient.onShowFileChooser` opens the Android picker
- **Mic / camera**: `onPermissionRequest` → runtime permission dialog → `request.grant()`
- **Clipboard**: `navigator.clipboard.writeText/readText` are patched on page load to use a
  native JS bridge (`window.AndroidClipboard` → `ClipboardManager`), so copy/paste works
  reliably inside the WebView
- **Text selection**: long-press works natively; WebView text selection menu is enabled by default
