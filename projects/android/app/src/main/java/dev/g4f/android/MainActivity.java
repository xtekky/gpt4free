package dev.g4f.android;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.WebSettings;
import android.view.KeyEvent;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends Activity {

    private WebView webView;
    private ExecutorService executor;
    public static final int PORT = 1337;
    private static final String APP_DIR = "app";
    private static final String EXTRACTION_VERSION = "3"; // bump to force re-extraction

    // File chooser (image / file upload from the chat UI)
    private static final int FILE_CHOOSER_REQUEST = 1001;
    private ValueCallback<Uri[]> filePathCallback;

    // Mic / camera permissions (audio recording, camera capture)
    private static final int WEB_PERMISSION_REQUEST = 1002;
    private PermissionRequest pendingWebPermissionRequest;
    private static final String[] NEEDED_PERMISSIONS = {
        Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        executor = Executors.newFixedThreadPool(2);

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(getApplicationContext()));
        }

        webView = new WebView(this);
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setDatabaseEnabled(true);
        ws.setAllowFileAccess(true);
        ws.setAllowContentAccess(true);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setJavaScriptCanOpenWindowsAutomatically(true);
        ws.setCacheMode(WebSettings.LOAD_DEFAULT);
        ws.setUserAgentString("Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36");

        webView.addJavascriptInterface(new ClipboardBridge(), "AndroidClipboard");
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                // Patch navigator.clipboard to use the native bridge (reliable
                // read+write in WebView, no permission quirks)
                view.evaluateJavascript(CLIPBOARD_PATCH_JS, null);
            }
        });
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback,
                                             FileChooserParams params) {
                if (filePathCallback != null) {
                    filePathCallback.onReceiveValue(null);
                }
                filePathCallback = callback;
                try {
                    startActivityForResult(params.createIntent(), FILE_CHOOSER_REQUEST);
                } catch (Exception e) {
                    filePathCallback = null;
                    return false;
                }
                return true;
            }

            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                // Must run on UI thread; grant mic/camera to the WebView after app-level check
                runOnUiThread(() -> {
                    pendingWebPermissionRequest = request;
                    requestPermissions(NEEDED_PERMISSIONS, WEB_PERMISSION_REQUEST);
                });
            }
        });

        setContentView(webView);

        startServer();
    }

    private static final String CLIPBOARD_PATCH_JS =
        "(function() {" +
        "  if (window.__g4fClipboardPatched) return;" +
        "  window.__g4fClipboardPatched = true;" +
        "  var bridge = window.AndroidClipboard;" +
        "  if (!bridge) return;" +
        "  if (!navigator.clipboard) navigator.clipboard = {};" +
        "  navigator.clipboard.writeText = function(text) {" +
        "    try { bridge.copy(String(text == null ? '' : text)); } catch (e) {}" +
        "    return Promise.resolve();" +
        "  };" +
        "  navigator.clipboard.readText = function() {" +
        "    var v = '';" +
        "    try { v = bridge.paste() || ''; } catch (e) {}" +
        "    return Promise.resolve(v);" +
        "  };" +
        "  console.log('[g4f-clipboard] bridge ready');" +
        "})();";

    /** Native clipboard access exposed to JavaScript as window.AndroidClipboard. */
    class ClipboardBridge {
        @JavascriptInterface
        public void copy(String text) {
            ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (cm != null) {
                cm.setPrimaryClip(ClipData.newPlainText("text", text));
            }
        }

        @JavascriptInterface
        public String paste() {
            ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (cm == null || cm.getPrimaryClip() == null) return "";
            ClipData.Item item = cm.getPrimaryClip().getItemAt(0);
            CharSequence t = item == null ? null : item.getText();
            return t == null ? "" : t.toString();
        }
    }

    private File appDir() {
        return new File(getFilesDir(), APP_DIR);
    }

    /** Recursively copy APK assets under assetPath into target. */
    private void extractAssets(String assetPath, File target) throws IOException {
        String[] entries = getAssets().list(assetPath);
        if (entries == null || entries.length == 0) {
            File outFile = new File(target, fileName(assetPath));
            outFile.getParentFile().mkdirs();
            InputStream in = getAssets().open(assetPath);
            OutputStream out = new FileOutputStream(outFile);
            byte[] buf = new byte[65536];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            out.close();
            in.close();
            return;
        }
        File sub = new File(target, fileName(assetPath));
        sub.mkdirs();
        for (String entry : entries) {
            extractAssets(assetPath + "/" + entry, sub);
        }
    }

    private static String fileName(String path) {
        int idx = path.lastIndexOf('/');
        return idx >= 0 ? path.substring(idx + 1) : path;
    }

    private static void deleteRecursive(File file) {
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) {
                for (File child : children) deleteRecursive(child);
            }
        }
        file.delete();
    }

    private void startServer() {
        // Thread 1: extract assets + start the Python server (blocks forever in app.run)
        executor.execute(() -> {
            try {
                File root = appDir();
                File markerV = new File(root, ".extracted-" + EXTRACTION_VERSION);
                if (!markerV.exists()) {
                    // Clean slate: remove stale files from previous extraction versions
                    deleteRecursive(root);
                    root.mkdirs();
                    extractAssets("g4f", root);
                    extractAssets("g4f.dev", root);
                    markerV.createNewFile();
                }

                Python py = Python.getInstance();
                py.getModule("bootstrap").callAttr(
                    "main", root.getAbsolutePath(), PORT);
            } catch (Exception e) {
                e.printStackTrace();
                showError("Server failed to start: " + e);
            }
        });

        // Thread 2: poll until the server answers, then load the WebView
        loadChatWhenReady();
    }

    private void loadChatWhenReady() {
        executor.execute(() -> {
            String url = "http://127.0.0.1:" + PORT + "/chat/";
            boolean up = false;
            for (int i = 0; i < 120; i++) {
                try {
                    java.net.URL u = new java.net.URL(url);
                    java.net.HttpURLConnection c = (java.net.HttpURLConnection) u.openConnection();
                    c.setConnectTimeout(1000);
                    c.setReadTimeout(1000);
                    int code = c.getResponseCode();
                    c.disconnect();
                    if (code == 200) { up = true; break; }
                } catch (Exception ignored) {}
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            }
            final boolean ok = up;
            runOnUiThread(() -> {
                if (ok) {
                    webView.loadUrl(url);
                } else {
                    showError("Server did not respond within 120s. Check logcat for [g4f-bootstrap].");
                }
            });
        });
    }

    private void showError(final String message) {
        runOnUiThread(() -> webView.loadData(
            "<html><body style='background:#18181b;color:#ef4444;"
            + "font-family:monospace;padding:24px;white-space:pre-wrap'>"
            + message.replace("<", "&lt;") + "</body></html>",
            "text/html", "utf-8"));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, android.content.Intent data) {
        if (requestCode == FILE_CHOOSER_REQUEST) {
            Uri[] results = null;
            if (resultCode == RESULT_OK && data != null && data.getData() != null) {
                results = new Uri[]{ data.getData() };
            }
            if (filePathCallback != null) {
                filePathCallback.onReceiveValue(results);
                filePathCallback = null;
            }
        } else {
            super.onActivityResult(requestCode, resultCode, data);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == WEB_PERMISSION_REQUEST && pendingWebPermissionRequest != null) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                pendingWebPermissionRequest.grant(pendingWebPermissionRequest.getResources());
            } else {
                pendingWebPermissionRequest.deny();
            }
            pendingWebPermissionRequest = null;
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    protected void onDestroy() {
        if (executor != null) executor.shutdown();
        super.onDestroy();
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && webView != null && webView.canGoBack()) {
            webView.goBack();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }
}
