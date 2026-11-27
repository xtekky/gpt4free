package dev.g4f.android.container;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.os.Looper;
import android.util.Base64;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Entry point for the Debian container version of the app. */
public final class ContainerActivity extends Activity {

    private static final String TAG = "g4f-container-ui";
    private static final int NOTIFICATION_REQUEST = 3301;

    private ContainerService service;
    private boolean bound;
    private WebView terminalView;
    private WebView contentView;
    private Button closeButton;
    private View fullscreenView;
    private View actionChrome;
    private WebChromeClient.CustomViewCallback fullscreenCallback;
    private boolean nativeFullscreen;
    private FrameLayout stage;
    private TextView statusText;
    private TextView progressText;
    private CheckBox desktopCheckBox;
    private volatile boolean terminalReady;
    private final List<String> pendingJs = new ArrayList<>();

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((ContainerService.LocalBinder) binder).getService();
            bound = true;
            service.registerListener(serviceListener);
            service.autoStartIfReady();
            runJs("if (window.__g4fSyncTabs) window.__g4fSyncTabs()");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            bound = false;
        }
    };

    private final ContainerService.Listener serviceListener = new ContainerService.Listener() {
        @Override
        public void onProgress(String message) {
            runOnUiThread(() -> progressText.setText(message));
        }

        @Override
        public void onState(String state) {
            runOnUiThread(() -> {
                statusText.setText(state);
                if (state != null && state.startsWith("Ready")) {
                    runJs("if (window.__g4fSyncTabs) window.__g4fSyncTabs()");
                }
            });
        }

        @Override
        public void onTerminalData(int sessionId, byte[] data) {
            String encoded = Base64.encodeToString(data, Base64.NO_WRAP);
            runJs("window.__g4fTermWrite(" + sessionId + ",'" + encoded + "')");
        }

        @Override
        public void onTerminalExit(int sessionId, int exitCode, boolean stopped) {
            runJs("window.__g4fTermExit(" + sessionId + "," + exitCode + "," + stopped + ")");
        }

        @Override
        public void onSessionsChanged() {
            runJs("if (window.__g4fSyncTabs) window.__g4fSyncTabs()");
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestNotificationsIfNeeded();
        buildUi();
        startAndBindService();
    }

    private void requestNotificationsIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] { Manifest.permission.POST_NOTIFICATIONS },
                    NOTIFICATION_REQUEST);
        }
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#101411"));

        statusText = new TextView(this);
        statusText.setTextColor(Color.parseColor("#c4f06c"));
        statusText.setTextSize(15f);
        statusText.setPadding(16, 14, 16, 4);
        root.addView(statusText);

        progressText = new TextView(this);
        progressText.setTextColor(Color.parseColor("#8b9a8c"));
        progressText.setTextSize(12f);
        progressText.setPadding(16, 0, 16, 10);
        root.addView(progressText);

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(10, 0, 10, 10);

        desktopCheckBox = new CheckBox(this);
        desktopCheckBox.setText("Desktop");
        desktopCheckBox.setTextColor(Color.parseColor("#d6dad3"));
        actions.addView(desktopCheckBox);

        actions.addView(button("Install", v -> {
            if (bound && service != null) service.install(desktopCheckBox.isChecked());
        }));
        actions.addView(button("Classic", v ->
                startActivity(new Intent(this, dev.g4f.android.MainActivity.class))));
        actions.addView(button("Stop", v -> {
            if (bound && service != null) service.closeAllSessions();
            runJs("if (window.closeAll) window.closeAll()");
        }));
        HorizontalScrollView actionScroller = new HorizontalScrollView(this);
        actionScroller.addView(actions);
        actionChrome = actionScroller;
        root.addView(actionScroller);

        stage = new FrameLayout(this);
        terminalView = new WebView(this);
        WebSettings settings = terminalView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        terminalView.setBackgroundColor(Color.parseColor("#101411"));
        terminalView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                terminalReady = true;
                List<String> scripts;
                synchronized (pendingJs) {
                    scripts = new ArrayList<>(pendingJs);
                    pendingJs.clear();
                }
                for (String script : scripts) terminalView.evaluateJavascript(script, null);
            }
        });
        terminalView.addJavascriptInterface(new TerminalBridge(), "AndroidTerminal");
        stage.addView(terminalView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        terminalView.loadUrl("file:///android_asset/container/terminal.html");

        root.addView(stage, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
    }

    private Button button(String label, android.view.View.OnClickListener listener) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(12f);
        button.setAllCaps(false);
        button.setOnClickListener(listener);
        return button;
    }

    private void startAndBindService() {
        Intent intent = new Intent(this, ContainerService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        bindService(intent, connection, Context.BIND_AUTO_CREATE);
    }

    private void openChat() {
        // The chat GUI is served by the container API, so make sure that
        // session is running before loading the bundled page.
        if (service == null) {
            Toast.makeText(this, "Container is starting", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            service.ensureSession("api");
        } catch (IllegalStateException error) {
            Toast.makeText(this, error.getMessage(), Toast.LENGTH_LONG).show();
            return;
        }
        openContent("http://127.0.0.1:1337/chat/");
        runJs("if (window.__g4fSetFixedTab) window.__g4fSetFixedTab('chat')");
    }

    private void openDesktop() {
        if (service == null || service.manager() == null
                || !service.manager().isDesktopInstalled()) {
            Toast.makeText(this, "Enable Desktop and reinstall to add XFCE + noVNC",
                    Toast.LENGTH_LONG).show();
            return;
        }
        // The noVNC client is served by the g4f API itself at /desktop/, so the
        // API session must be running before the WebView loads that URL.
        try {
            service.ensureSession("api");
            service.ensureSession("desktop");
        } catch (IllegalStateException error) {
            Toast.makeText(this, error.getMessage(), Toast.LENGTH_LONG).show();
        }
        openContent("http://127.0.0.1:1337/desktop/");
        runJs("if (window.__g4fSetFixedTab) window.__g4fSetFixedTab('desktop')");
    }

    private void openContent(String url) {
        if (contentView == null) {
            contentView = new WebView(this);
            WebSettings settings = contentView.getSettings();
            settings.setJavaScriptEnabled(true);
            settings.setDomStorageEnabled(true);
            settings.setAllowFileAccess(true);
            settings.setMediaPlaybackRequiresUserGesture(false);
            settings.setUserAgentString("Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 "
                    + "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36");
            contentView.setWebViewClient(new WebViewClient());
            contentView.setWebChromeClient(new WebChromeClient() {
                @Override
                public void onShowCustomView(View view, CustomViewCallback callback) {
                    showFullscreen(view, callback);
                }

                @Override
                public void onHideCustomView() {
                    hideFullscreen();
                }
            });
            contentView.addJavascriptInterface(new DesktopBridge(), "AndroidDesktop");
            FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT);
            stage.addView(contentView, params);

            closeButton = new Button(this);
            closeButton.setText("✕ Close");
            closeButton.setOnClickListener(v -> hideContent());
            FrameLayout.LayoutParams closeParams = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP | Gravity.END);
            closeParams.setMargins(0, 8, 8, 0);
            stage.addView(closeButton, closeParams);
            closeButton.setVisibility(android.view.View.GONE);
        }
        contentView.loadUrl(url);
        contentView.setVisibility(android.view.View.VISIBLE);
        contentView.bringToFront();
        if (closeButton != null) {
            closeButton.setVisibility(android.view.View.VISIBLE);
            closeButton.bringToFront();
        }
    }

    /** WebView fullscreen custom view (used by noVNC's Full button). */
    private void showFullscreen(View view, WebChromeClient.CustomViewCallback callback) {
        if (fullscreenView != null) {
            callback.onCustomViewHidden();
            return;
        }
        fullscreenView = view;
        fullscreenCallback = callback;
        stage.addView(view, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        contentView.setVisibility(android.view.View.GONE);
        if (closeButton != null) closeButton.setVisibility(android.view.View.GONE);
        setAppChromeVisible(false);
    }

    private void hideFullscreen() {
        if (fullscreenView == null) return;
        stage.removeView(fullscreenView);
        fullscreenView = null;
        if (fullscreenCallback != null) fullscreenCallback.onCustomViewHidden();
        fullscreenCallback = null;
        if (contentView != null) contentView.setVisibility(android.view.View.VISIBLE);
        if (closeButton != null) closeButton.setVisibility(android.view.View.VISIBLE);
        setAppChromeVisible(true);
    }

    private void setAppChromeVisible(boolean visible) {
        int flags = visible ? android.view.View.VISIBLE : android.view.View.GONE;
        statusText.setVisibility(flags);
        progressText.setVisibility(flags);
        if (actionChrome != null) actionChrome.setVisibility(flags);
    }

    /** Fallback for WebViews that block the HTML Fullscreen API in an iframe. */
    private void setNativeFullscreen(boolean enabled) {
        runOnUiThread(() -> {
            nativeFullscreen = enabled;
            int flags = enabled
                    ? View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    : 0;
            getWindow().getDecorView().setSystemUiVisibility(flags);
            setAppChromeVisible(!enabled);
            if (closeButton != null && contentView != null
                    && contentView.getVisibility() == android.view.View.VISIBLE) {
                closeButton.setVisibility(enabled ? android.view.View.GONE
                        : android.view.View.VISIBLE);
            }
        });
    }

    private final class DesktopBridge {
        @JavascriptInterface
        public void setFullscreen(String enabled) {
            setNativeFullscreen("true".equals(enabled));
        }
    }

    private void hideContent() {
        hideFullscreen();
        setNativeFullscreen(false);
        if (contentView != null) contentView.setVisibility(android.view.View.GONE);
        if (closeButton != null) closeButton.setVisibility(android.view.View.GONE);
        runJs("if (window.__g4fClearFixedTab) window.__g4fClearFixedTab()");
    }

    private void runJs(String script) {
        if (!terminalReady || terminalView == null) {
            synchronized (pendingJs) {
                pendingJs.add(script);
            }
            return;
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            terminalView.evaluateJavascript(script, null);
        } else {
            runOnUiThread(() -> terminalView.evaluateJavascript(script, null));
        }
    }

    @Override
    public void onBackPressed() {
        if (fullscreenView != null) {
            hideFullscreen();
            return;
        }
        if (nativeFullscreen) {
            setNativeFullscreen(false);
            return;
        }
        if (contentView != null && contentView.getVisibility() == android.view.View.VISIBLE) {
            hideContent();
            return;
        }
        super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (bound) {
            if (service != null) service.unregisterListener(serviceListener);
            unbindService(connection);
            bound = false;
        }
        if (terminalView != null) terminalView.destroy();
        if (contentView != null) contentView.destroy();
        super.onDestroy();
    }

    /** Bridge used by assets/container/terminal.html. */
    public final class TerminalBridge {
        @JavascriptInterface
        public String ensureSession(String kind) {
            if (service == null) return "";
            try {
                return Integer.toString(service.ensureSession(kind));
            } catch (Exception error) {
                Log.e(TAG, "ensureSession failed", error);
                toast(error.getMessage());
                return "";
            }
        }

        @JavascriptInterface
        public String newSession(String kind) {
            if (service == null) return "";
            try {
                return Integer.toString(service.newSession(kind));
            } catch (Exception error) {
                Log.e(TAG, "newSession failed", error);
                toast(error.getMessage());
                return "";
            }
        }

        @JavascriptInterface
        public void send(String sessionId, String base64) {
            if (service != null) service.write(Integer.parseInt(sessionId), base64);
        }

        @JavascriptInterface
        public void resize(String sessionId, int cols, int rows) {
            if (service != null) service.resize(Integer.parseInt(sessionId), cols, rows);
        }

        @JavascriptInterface
        public void close(String sessionId) {
            if (service != null) service.close(Integer.parseInt(sessionId));
        }

        @JavascriptInterface
        public void closeAll() {
            if (service != null) service.closeAllSessions();
        }

        @JavascriptInterface
        public void copy(String base64) {
            String text = new String(Base64.decode(base64, Base64.DEFAULT),
                    StandardCharsets.UTF_8);
            ClipboardManager clipboard =
                    (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null) {
                clipboard.setPrimaryClip(ClipData.newPlainText("g4f terminal", text));
            }
            toast("Copied");
        }

        @JavascriptInterface
        public void openChat() {
            runOnUiThread(ContainerActivity.this::openChat);
        }

        @JavascriptInterface
        public void openDesktop() {
            runOnUiThread(ContainerActivity.this::openDesktop);
        }

        @JavascriptInterface
        public String sessions() {
            return service == null ? "" : service.sessionList();
        }

        @JavascriptInterface
        public String status() {
            return service == null ? "Service not connected" : service.stateText();
        }

        private void toast(String message) {
            runOnUiThread(() -> Toast.makeText(ContainerActivity.this,
                    message, Toast.LENGTH_SHORT).show());
        }
    }
}
