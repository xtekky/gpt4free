package dev.g4f.android.container;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
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
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Entry point for the Debian container version of the app. */
public final class ContainerActivity extends Activity {

    private static final String TAG = "g4f-container-ui";
    private static final int NOTIFICATION_REQUEST = 3301;

    private ContainerService service;
    private boolean bound;
    private WebView contentView;
    private StoreView storeView;
    private Button closeButton;
    private Button fullscreenExitButton;
    private View fullscreenView;
    private View actionChrome;
    private HorizontalScrollView tabScroller;
    private View keyRow;
    private WebChromeClient.CustomViewCallback fullscreenCallback;
    private boolean nativeFullscreen;
    private FrameLayout stage;
    private LinearLayout tabBar;
    private TextView statusText;
    private TextView progressText;
    private CheckBox desktopCheckBox;

    private final Map<Integer, TerminalView> terminals = new LinkedHashMap<>();
    private final Map<Integer, String> sessionLabels = new LinkedHashMap<>();
    private final Map<Integer, Button> tabButtons = new LinkedHashMap<>();
    private final Map<String, Button> fixedButtons = new LinkedHashMap<>();
    private TerminalView activeTerminal;
    private String activeFixedTab;
    private Button ctrlButton;
    private Button altButton;
    private Button selectButton;
    private boolean ctrlPressed;
    private boolean altPressed;
    private boolean startingShell;

    // ── Automation WebViews (CDP targets) ─────────────────────────────
    // The app's WebView DevTools socket is the only browser available in the
    // container. The hidden content WebView exposes window.G4FAutomation so
    // the container's CDPSession can create dedicated automation WebViews.
    private final Map<String, WebView> automationWebViews = new LinkedHashMap<>();
    private int automationRefCount = 0;

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((ContainerService.LocalBinder) binder).getService();
            bound = true;
            service.registerListener(serviceListener);
            service.autoStartIfReady();
            syncSessions();
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
            runOnUiThread(() -> statusText.setText(state));
        }

        @Override
        public void onTerminalData(int sessionId, byte[] data) {
            TerminalView view = terminals.get(sessionId);
            if (view != null) view.write(data);
        }

        @Override
        public void onTerminalExit(int sessionId, int exitCode, boolean stopped) {
            TerminalView view = terminals.get(sessionId);
            if (view != null) view.showProcessExit(exitCode, stopped);
        }

        @Override
        public void onSessionsChanged() {
            runOnUiThread(ContainerActivity.this::syncSessions);
        }
    };

    private final TerminalView.Listener terminalListener = new TerminalView.Listener() {
        @Override
        public void onInput(int sessionId, byte[] data) {
            if (service != null) service.writeBytes(sessionId, data);
        }

        @Override
        public void onResize(int sessionId, int cols, int rows) {
            if (service != null) service.resize(sessionId, cols, rows);
        }

        @Override
        public void onCopy(String text) {
            copyToClipboard(text);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Expose @webview_devtools_remote_<pid> so CDPSession in the container
        // can attach to the app's WebView (no installable Chrome on Android).
        WebView.setWebContentsDebuggingEnabled(isWebViewDebugEnabled());
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

        actions.addView(button("Install", v ->
                confirmInstall(desktopCheckBox.isChecked())));
        actions.addView(button("Classic", v ->
                startActivity(new Intent(this, dev.g4f.android.MainActivity.class))));
        actions.addView(button("Stop", v -> closeAll()));
        HorizontalScrollView actionScroller = new HorizontalScrollView(this);
        actionScroller.addView(actions);
        actionChrome = actionScroller;
        root.addView(actionScroller);

        buildTabBar();
        root.addView(tabScroller);

        stage = new FrameLayout(this);
        stage.setBackgroundColor(Color.parseColor("#101411"));
        root.addView(stage, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        fullscreenExitButton = new Button(this);
        fullscreenExitButton.setText("✕ Exit");
        fullscreenExitButton.setTextSize(11f);
        fullscreenExitButton.setAllCaps(false);
        fullscreenExitButton.setOnClickListener(v -> setNativeFullscreen(false));
        FrameLayout.LayoutParams fullscreenParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP | Gravity.END);
        fullscreenParams.setMargins(0, 8, 8, 0);
        stage.addView(fullscreenExitButton, fullscreenParams);
        fullscreenExitButton.setVisibility(View.GONE);

        keyRow = buildKeyRow();
        root.addView(keyRow);

        setContentView(root);
        // A hidden 127.0.0.1 page keeps a control target with the automation
        // bridge available even before the user opens Home or Chat.
        prepareAutomationControlPage();
    }

    private void buildTabBar() {
        tabScroller = new HorizontalScrollView(this);
        tabScroller.setHorizontalScrollBarEnabled(false);
        tabScroller.setBackgroundColor(Color.parseColor("#181d19"));
        tabBar = new LinearLayout(this);
        tabBar.setOrientation(LinearLayout.HORIZONTAL);
        tabBar.setPadding(6, 4, 6, 4);
        tabScroller.addView(tabBar);

        addFixedButton("home", "Home", v -> openHome());
        addFixedButton("chat", "Chat", v -> openChat());
        addFixedButton("api", "API", v -> showApi());
        addFixedButton("desktop", "Desktop", v -> openDesktop());
        addFixedButton("store", "Store", v -> openStore());

        tabBar.addView(tabButton("Full", v -> toggleFullscreen()));
        tabBar.addView(tabButton("+ Shell", v -> newShell()));
        tabBar.addView(tabButton("Close", v -> closeActive()));
        tabBar.addView(tabButton("Close All", v -> closeAll()));
    }

    private void addFixedButton(String name, String label, View.OnClickListener listener) {
        Button button = tabButton(label, listener);
        fixedButtons.put(name, button);
        tabBar.addView(button);
    }

    private Button tabButton(String label, View.OnClickListener listener) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextSize(12f);
        button.setAllCaps(false);
        button.setMinWidth(0);
        button.setMinimumWidth(0);
        button.setPadding(18, 4, 18, 4);
        button.setTextColor(Color.parseColor("#c4f06c"));
        button.setBackgroundColor(Color.TRANSPARENT);
        button.setOnClickListener(listener);
        return button;
    }

    private View buildKeyRow() {
        HorizontalScrollView scroller = new HorizontalScrollView(this);
        scroller.setBackgroundColor(Color.parseColor("#181d19"));
        scroller.setHorizontalScrollBarEnabled(false);
        LinearLayout keys = new LinearLayout(this);
        keys.setOrientation(LinearLayout.HORIZONTAL);
        keys.setPadding(6, 4, 6, 4);
        scroller.addView(keys);
        keys.addView(keyButton("Esc", "esc"));
        keys.addView(keyButton("Tab", "tab"));
        ctrlButton = keyButton("Ctrl", "ctrl");
        keys.addView(ctrlButton);
        altButton = keyButton("Alt", "alt");
        keys.addView(altButton);
        keys.addView(keyButton("←", "left"));
        keys.addView(keyButton("↑", "up"));
        keys.addView(keyButton("↓", "down"));
        keys.addView(keyButton("→", "right"));
        keys.addView(keyButton("Home", "home"));
        keys.addView(keyButton("End", "end"));
        keys.addView(keyButton("^C", "ctrlc"));
        keys.addView(keyButton("^D", "ctrld"));
        selectButton = keyButton("Select", "select");
        keys.addView(selectButton);
        keys.addView(keyButton("Copy", "copy"));
        keys.addView(keyButton("All", "all"));
        keys.addView(keyButton("Paste", "paste"));
        return scroller;
    }

    private Button keyButton(String label, final String name) {
        Button button = tabButton(label, v -> handleKeyButton(name));
        button.setTextSize(11f);
        button.setPadding(14, 2, 14, 2);
        return button;
    }

    private void handleKeyButton(String name) {
        TerminalView view = requireActiveTerminal();
        if (view == null) return;
        if ("ctrl".equals(name)) {
            ctrlPressed = !ctrlPressed;
            if (ctrlPressed) altPressed = false;
            applyModifiers(view);
            return;
        }
        if ("alt".equals(name)) {
            altPressed = !altPressed;
            if (altPressed) ctrlPressed = false;
            applyModifiers(view);
            return;
        }
        if ("select".equals(name)) {
            view.toggleSelectionMode();
            styleModifierButtons();
            return;
        }
        if ("copy".equals(name)) {
            if (!view.copySelection()) toast("Nothing selected");
            return;
        }
        if ("all".equals(name)) {
            view.selectAll();
            return;
        }
        if ("paste".equals(name)) {
            pasteClipboard(view);
            return;
        }
        view.setCtrl(ctrlPressed);
        view.setAlt(altPressed);
        view.sendKey(name);
        ctrlPressed = false;
        altPressed = false;
        view.clearModifiers();
        styleModifierButtons();
        view.requestFocus();
    }

    private void applyModifiers(TerminalView view) {
        view.setCtrl(ctrlPressed);
        view.setAlt(altPressed);
        styleModifierButtons();
    }

    private void styleModifierButtons() {
        if (ctrlButton != null) {
            ctrlButton.setTextColor(Color.parseColor(ctrlPressed ? "#ffffff" : "#c4f06c"));
            ctrlButton.setBackgroundColor(Color.parseColor(ctrlPressed ? "#31402f" : "#00000000"));
        }
        if (altButton != null) {
            altButton.setTextColor(Color.parseColor(altPressed ? "#ffffff" : "#c4f06c"));
            altButton.setBackgroundColor(Color.parseColor(altPressed ? "#31402f" : "#00000000"));
        }
        if (selectButton != null) {
            boolean on = activeTerminal != null && activeTerminal.isSelectionMode();
            selectButton.setTextColor(Color.parseColor(on ? "#ffffff" : "#c4f06c"));
            selectButton.setBackgroundColor(Color.parseColor(on ? "#31402f" : "#00000000"));
        }
    }

    private TerminalView requireActiveTerminal() {
        if (contentVisible()) return null;
        if (activeTerminal != null) return activeTerminal;
        if (!terminals.isEmpty()) {
            TerminalView view = terminals.values().iterator().next();
            showTerminal(view, false);
            return view;
        }
        return null;
    }

    private void newShell() {
        if (service == null) {
            toast("Container is starting");
            return;
        }
        startingShell = true;
        try {
            int id = service.newSession("shell");
            TerminalView view = terminals.get(id);
            if (view != null) showTerminal(view, true);
        } catch (IllegalStateException error) {
            toast(error.getMessage());
        } finally {
            startingShell = false;
        }
    }

    private void showApi() {
        if (service == null) {
            toast("Container is starting");
            return;
        }
        try {
            int id = service.ensureSession("api");
            TerminalView view = terminals.get(id);
            if (view != null) showTerminal(view, true);
        } catch (IllegalStateException error) {
            toast(error.getMessage());
        }
    }

    private void closeActive() {
        if (activeTerminal == null || service == null) return;
        service.close(activeTerminal.sessionId());
    }

    private void closeAll() {
        if (service != null) service.closeAllSessions();
        hideContent();
        if (storeView != null) storeView.setVisibility(View.GONE);
        activeTerminal = null;
        activeFixedTab = null;
        styleTabs();
    }

    private void openHome() {
        // The site's Home page is mounted by the same API server as Chat.
        if (service == null) {
            toast("Container is starting");
            return;
        }
        try {
            service.ensureSession("api");
        } catch (IllegalStateException error) {
            toast(error.getMessage());
            return;
        }
        openContent("http://127.0.0.1:1337/");
        activeFixedTab = "home";
        styleTabs();
    }

    private void openChat() {
        // The chat GUI is served by the container API, so make sure that
        // session is running before loading the bundled page.
        if (service == null) {
            toast("Container is starting");
            return;
        }
        try {
            service.ensureSession("api");
        } catch (IllegalStateException error) {
            toast(error.getMessage());
            return;
        }
        openContent("http://127.0.0.1:1337/chat/");
        activeFixedTab = "chat";
        styleTabs();
    }

    private void openDesktop() {
        if (service == null || service.manager() == null
                || !service.manager().isDesktopInstalled()) {
            toast("Enable Desktop and reinstall to add XFCE + noVNC");
            return;
        }
        // The noVNC client is served by the g4f API itself at /desktop/, so the
        // API session must be running before the WebView loads that URL.
        try {
            service.ensureSession("api");
            service.ensureSession("desktop");
        } catch (IllegalStateException error) {
            toast(error.getMessage());
        }
        openContent("http://127.0.0.1:1337/desktop/");
        activeFixedTab = "desktop";
        styleTabs();
    }

    // ── Sandbox app store ──────────────────────────────────────────────

    private void openStore() {
        if (service == null || !service.isReady()) {
            toast("Install the container first");
            return;
        }
        ensureStoreView();
        hideKeyboard();
        if (contentView != null) contentView.setVisibility(View.GONE);
        if (closeButton != null) closeButton.setVisibility(View.GONE);
        for (TerminalView view : terminals.values()) view.setVisibility(View.GONE);
        activeTerminal = null;
        storeView.setVisibility(View.VISIBLE);
        storeView.bringToFront();
        activeFixedTab = "store";
        styleTabs();
        if (storeView.getToolsCount() == 0) reloadStore();
    }

    private void ensureStoreView() {
        if (storeView != null) return;
        storeView = new StoreView(this, new StoreView.Listener() {
            @Override
            public void onInstall(String tool) {
                installStoreTool(tool);
            }

            @Override
            public void onOpenCommand(String command) {
                openBinCommand(command);
            }
        });
        storeView.setVisibility(View.GONE);
        stage.addView(storeView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
    }

    /** Ask the container for the current `sandbox-cli list` catalog. */
    private void reloadStore() {
        if (service == null) return;
        storeView.showLoading();
        service.listStoreTools(new ContainerService.StoreListCallback() {
            @Override
            public void onResult(String output) {
                if (storeView != null) storeView.showTools(output);
            }

            @Override
            public void onError(String message) {
                if (storeView != null) storeView.showError(message);
            }
        });
    }

    /** Show the install in its own terminal so the user sees every log line. */
    private void installStoreTool(String tool) {
        if (service == null) {
            toast("Container is starting");
            return;
        }
        try {
            int id = service.installStoreTool(tool);
            TerminalView view = terminals.get(id);
            if (view != null) showTerminal(view, true);
            toast("Installing " + tool + "…");
        } catch (IllegalStateException | IllegalArgumentException error) {
            toast(error.getMessage());
        }
    }

    /** Show a bundled /g4f-bin command in a terminal, ready to be run. */
    private void openBinCommand(String command) {
        if (service == null) {
            toast("Container is starting");
            return;
        }
        try {
            int id = service.openBinCommand(command);
            TerminalView view = terminals.get(id);
            if (view != null) showTerminal(view, true);
            toast(command + " is ready — add arguments, then press Enter");
        } catch (IllegalStateException | IllegalArgumentException error) {
            toast(error.getMessage());
        }
    }

    private boolean storeVisible() {
        return storeView != null && storeView.getVisibility() == View.VISIBLE;
    }

    private void hideStore() {
        if (storeView != null) storeView.setVisibility(View.GONE);
        activeFixedTab = null;
        if (activeTerminal == null && !terminals.isEmpty()) {
            activeTerminal = terminals.values().iterator().next();
            activeTerminal.setVisibility(View.VISIBLE);
        }
        styleTabs();
    }

    // ── Session tabs ───────────────────────────────────────────────────

    private void syncSessions() {
        if (service == null) return;
        Map<Integer, String> current = parseSessions(service.sessionList());
        List<Integer> removed = new ArrayList<>();
        for (Integer id : terminals.keySet()) {
            if (!current.containsKey(id)) removed.add(id);
        }
        for (Integer id : removed) removeTerminal(id);

        for (Map.Entry<Integer, String> entry : current.entrySet()) {
            int id = entry.getKey();
            String label = entry.getValue();
            // The desktop session only runs Xvnc/startxfce4 in the background.
            // Rendering its PTY would show the display server console instead of
            // the /desktop/ VNC page, so keep it out of the tab bar.
            if ("desktop".equals(label)) continue;
            if (!terminals.containsKey(id)) {
                TerminalView view = createTerminal(id, label);
                if (activeTerminal == null && activeFixedTab == null) {
                    showTerminal(view, false);
                }
            }
        }

        if (terminals.isEmpty() && !startingShell && !contentVisible()
                && !storeVisible() && service.stateText().startsWith("Ready")) {
            startingShell = true;
            try {
                service.newSession("shell");
            } catch (IllegalStateException error) {
                Log.i(TAG, "Shell not started yet: " + error.getMessage());
            } finally {
                startingShell = false;
            }
        }
        styleTabs();
        if (nativeFullscreen) {
            if (activeTerminal == null && !contentVisible()) setNativeFullscreen(false);
            else updateChromeVisibility();
        }
    }

    private TerminalView createTerminal(int id, String label) {
        TerminalView view = new TerminalView(this);
        view.setSession(id, terminalListener);
        if ("shell".equals(label)) view.showBanner();
        terminals.put(id, view);
        sessionLabels.put(id, label);
        stage.addView(view, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        view.setVisibility(View.GONE);

        Button tab = tabButton(tabLabel(id, label), v -> selectSession(id));
        tabButtons.put(id, tab);
        int index = Math.max(0, tabBar.getChildCount() - 3);
        tabBar.addView(tab, index);
        return view;
    }

    private static String tabLabel(int id, String label) {
        if ("api".equals(label)) return "API";
        if ("setup".equals(label)) return "Setup";
        if ("desktop-setup".equals(label)) return "Desktop setup";
        if (label.startsWith("store-")) return label.substring(6);
        if (label.startsWith("bin-")) return label.substring(4);
        return label + " #" + id;
    }

    private void removeTerminal(int id) {
        TerminalView view = terminals.remove(id);
        sessionLabels.remove(id);
        if (view != null) {
            if (activeTerminal == view) activeTerminal = null;
            if (view.getParent() == stage) stage.removeView(view);
        }
        Button tab = tabButtons.remove(id);
        if (tab != null && tab.getParent() == tabBar) tabBar.removeView(tab);
    }

    private void selectSession(int id) {
        TerminalView view = terminals.get(id);
        if (view != null) showTerminal(view, true);
    }

    private void showTerminal(TerminalView view, boolean focus) {
        if (view == null) return;
        if (contentView != null) contentView.setVisibility(View.GONE);
        if (storeView != null) storeView.setVisibility(View.GONE);
        if (closeButton != null) closeButton.setVisibility(View.GONE);
        activeTerminal = view;
        activeFixedTab = null;
        for (TerminalView other : terminals.values()) {
            other.setVisibility(other == view ? View.VISIBLE : View.GONE);
        }
        styleTabs();
        applyModifiers(view);
        if (nativeFullscreen) updateChromeVisibility();
        if (focus) {
            view.requestFocus();
            InputMethodManager imm = (InputMethodManager) getSystemService(
                    Context.INPUT_METHOD_SERVICE);
            if (imm != null) imm.showSoftInput(view, InputMethodManager.SHOW_IMPLICIT);
        }
    }

    private void styleTabs() {
        for (Map.Entry<Integer, Button> entry : tabButtons.entrySet()) {
            boolean active = activeTerminal != null
                    && activeTerminal.sessionId() == entry.getKey();
            styleTab(entry.getValue(), active);
        }
        for (Map.Entry<String, Button> entry : fixedButtons.entrySet()) {
            boolean active = activeFixedTab != null && activeFixedTab.equals(entry.getKey());
            styleTab(entry.getValue(), active);
        }
        styleModifierButtons();
        scrollActiveTabIntoView();
    }

    private void styleTab(Button button, boolean active) {
        button.setTextColor(Color.parseColor(active ? "#ffffff" : "#c4f06c"));
        button.setBackgroundColor(Color.parseColor(active ? "#31402f" : "#00000000"));
    }

    private void scrollActiveTabIntoView() {
        if (tabScroller == null) return;
        View target = null;
        if (activeTerminal != null) target = tabButtons.get(activeTerminal.sessionId());
        if (target == null && activeFixedTab != null) target = fixedButtons.get(activeFixedTab);
        if (target == null) return;
        final View view = target;
        tabScroller.post(() -> tabScroller.smoothScrollTo(
                Math.max(0, view.getLeft() - 12), 0));
    }

    private static Map<Integer, String> parseSessions(String raw) {
        Map<Integer, String> result = new LinkedHashMap<>();
        if (raw == null || raw.isEmpty()) return result;
        for (String pair : raw.split(",")) {
            String[] parts = pair.split(":", 2);
            if (parts.length < 2) continue;
            try {
                result.put(Integer.parseInt(parts[0]), parts[1]);
            } catch (NumberFormatException ignored) {
                // Skip malformed entries instead of dropping the whole list.
            }
        }
        return result;
    }

    private boolean contentVisible() {
        return contentView != null && contentView.getVisibility() == View.VISIBLE;
    }

    // ── Clipboard ──────────────────────────────────────────────────────

    private void copyToClipboard(String text) {
        if (text == null || text.isEmpty()) return;
        ClipboardManager clipboard =
                (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("g4f terminal", text));
        }
        toast("Copied");
    }

    private void pasteClipboard(TerminalView view) {
        ClipboardManager clipboard =
                (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard == null || !clipboard.hasPrimaryClip()) return;
        ClipData.Item item = clipboard.getPrimaryClip().getItemAt(0);
        CharSequence text = item == null ? null : item.coerceToText(this);
        if (text == null) return;
        String value = text.toString().replace("\r\n", "\r").replace('\n', '\r');
        view.sendText(value);
    }

    private void toast(String message) {
        runOnUiThread(() -> Toast.makeText(this, message, Toast.LENGTH_SHORT).show());
    }

    /** Start (or, after a confirmation, redo) the container install.
     *
     *  A second tap while the files are installing or the setup script is
     *  still downloading CPython used to reset the rootfs and start over,
     *  which looks exactly like provisioning that never finishes. Block that
     *  window and confirm an intentional reinstall of a ready container. */
    private void confirmInstall(final boolean withDesktop) {
        if (!bound || service == null) {
            Toast.makeText(this, "Container service is not ready yet", Toast.LENGTH_SHORT).show();
            return;
        }
        if (service.isBusy()) {
            Toast.makeText(this,
                    "Setup is still running — wait until it finishes (the setup tab shows progress)",
                    Toast.LENGTH_LONG).show();
            return;
        }
        if (service.isReady()) {
            new AlertDialog.Builder(this)
                    .setTitle("Reinstall container?")
                    .setMessage("This deletes the current Debian container, the g4f-go Python "
                            + "runtime and all installed packages, then downloads and provisions "
                            + "everything again. Continue?")
                    .setNegativeButton("Cancel", null)
                    .setPositiveButton("Reinstall", (dialog, which) -> {
                        if (bound && service != null) service.install(withDesktop);
                    })
                    .show();
            return;
        }
        service.install(withDesktop);
    }

    /** WebView debugging master switch (manifest meta-data, same pattern as
     *  the Classic activity). Provider automation needs the DevTools socket. */
    private boolean isWebViewDebugEnabled() {
        try {
            android.content.pm.ApplicationInfo info = getPackageManager()
                    .getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
            return info.metaData == null || info.metaData.getBoolean("WebViewDebug", true);
        } catch (Exception e) {
            return true;
        }
    }

    /** Create the content WebView (hidden) once, with the desktop and
     *  automation JS bridges installed. */
    private void ensureContentView() {
        if (contentView != null) return;
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
        contentView.addJavascriptInterface(new AutomationBridge(), "G4FAutomation");
        contentView.setVisibility(View.GONE);
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
        closeButton.setVisibility(View.GONE);
    }

    /** Keep a hidden control target alive for the container's CDPSession.
     *
     *  The page is loaded with a 127.0.0.1 base URL so CDPSession recognises
     *  it as the control page (see g4f.requests.cdp._is_control_page) and can
     *  call window.G4FAutomation over the DevTools socket. */
    private void prepareAutomationControlPage() {
        ensureContentView();
        if (contentView.getUrl() == null) {
            contentView.loadDataWithBaseURL("http://127.0.0.1:1337/",
                    "<!doctype html><html><body></body></html>",
                    "text/html", "utf-8", null);
        }
    }

    private void openContent(String url) {
        ensureContentView();
        hideKeyboard();
        if (storeView != null) storeView.setVisibility(View.GONE);
        contentView.loadUrl(url);
        contentView.setVisibility(View.VISIBLE);
        contentView.bringToFront();
        if (closeButton != null) {
            closeButton.setVisibility(View.VISIBLE);
            closeButton.bringToFront();
        }
        activeTerminal = null;
        for (TerminalView view : terminals.values()) view.setVisibility(View.GONE);
        if (nativeFullscreen) updateChromeVisibility();
    }

    private void hideKeyboard() {
        View focus = getCurrentFocus();
        if (focus == null) return;
        InputMethodManager imm =
                (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null) imm.hideSoftInputFromWindow(focus.getWindowToken(), 0);
    }

    // ── Automation WebViews (created through the JS bridge) ────────────

    private void createAutomationWebView(final String targetId, final String url) {
        runOnUiThread(() -> {
            if (automationWebViews.containsKey(targetId)) return;
            WebView aw = new WebView(this);
            WebSettings s = aw.getSettings();
            s.setJavaScriptEnabled(true);
            s.setDomStorageEnabled(true);
            if (contentView != null) {
                s.setUserAgentString(contentView.getSettings().getUserAgentString());
            }
            aw.setWebViewClient(new WebViewClient());
            aw.setWebChromeClient(new WebChromeClient());

            FrameLayout container = new FrameLayout(this);
            container.addView(aw, new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
            Button closeBtn = new Button(this);
            closeBtn.setText("✕ Close");
            closeBtn.setOnClickListener(v -> destroyAutomationWebView(targetId));
            FrameLayout.LayoutParams btnParams = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP | Gravity.END);
            btnParams.setMargins(0, 48, 24, 0);
            container.addView(closeBtn, btnParams);

            automationWebViews.put(targetId, aw);
            if (automationRefCount == 0 && isWebViewDebugEnabled()) {
                WebView.setWebContentsDebuggingEnabled(true);
            }
            automationRefCount++;
            stage.addView(container, new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
            container.bringToFront();
            aw.loadUrl(url);
            Log.i("g4f-automation", "container created target " + targetId + " -> " + url);
        });
    }

    private void destroyAutomationWebView(final String targetId) {
        runOnUiThread(() -> {
            WebView aw = automationWebViews.remove(targetId);
            if (aw == null) return;
            automationRefCount = Math.max(0, automationRefCount - 1);
            if (automationRefCount == 0) {
                WebView.setWebContentsDebuggingEnabled(isWebViewDebugEnabled());
            }
            ViewGroup container = (ViewGroup) aw.getParent();
            if (container != null) stage.removeView(container);
            aw.destroy();
            Log.i("g4f-automation", "container destroyed target " + targetId
                    + " (" + automationRefCount + " remaining)");
        });
    }

    /** JS bridge used by the container's CDPSession via the DevTools socket. */
    public final class AutomationBridge {
        @JavascriptInterface
        public String createTarget(final String url) {
            final String targetId = "auto-" + System.nanoTime();
            createAutomationWebView(targetId,
                    url != null && url.length() > 0 ? url : "about:blank");
            return targetId;
        }

        @JavascriptInterface
        public void closeTarget(final String targetId) {
            if (targetId != null) destroyAutomationWebView(targetId);
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
        contentView.setVisibility(View.GONE);
        if (closeButton != null) closeButton.setVisibility(View.GONE);
        setAppChromeVisible(false);
    }

    private void hideFullscreen() {
        if (fullscreenView == null) return;
        stage.removeView(fullscreenView);
        fullscreenView = null;
        if (fullscreenCallback != null) fullscreenCallback.onCustomViewHidden();
        fullscreenCallback = null;
        if (contentView != null) contentView.setVisibility(View.VISIBLE);
        if (closeButton != null) {
            closeButton.setVisibility(nativeFullscreen ? View.GONE : View.VISIBLE);
        }
        updateChromeVisibility();
    }

    private void setAppChromeVisible(boolean visible) {
        int flags = visible ? View.VISIBLE : View.GONE;
        statusText.setVisibility(flags);
        progressText.setVisibility(flags);
        if (actionChrome != null) actionChrome.setVisibility(flags);
        if (tabScroller != null) tabScroller.setVisibility(flags);
        if (keyRow != null) keyRow.setVisibility(flags);
    }

    /** Toggle immersive mode for the current chat/desktop page or shell. */
    private void toggleFullscreen() {
        if (nativeFullscreen) {
            setNativeFullscreen(false);
            return;
        }
        if (!contentVisible() && activeTerminal == null) {
            toast("Open Chat or a shell first");
            return;
        }
        setNativeFullscreen(true);
    }

    /** Apply the app chrome/immersive state for the current mode.
     *
     *  Chat, Home and Desktop use the whole stage, so every extra row is
     *  hidden. A fullscreen shell keeps its Termux-style key row: without it
     *  there would be no way to send Ctrl/Alt, arrows or Esc while the tab
     *  bar is gone. */
    private void updateChromeVisibility() {
        if (!nativeFullscreen) {
            setAppChromeVisible(true);
            return;
        }
        setAppChromeVisible(false);
        if (activeTerminal != null && !contentVisible() && keyRow != null) {
            keyRow.setVisibility(View.VISIBLE);
        }
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
            updateChromeVisibility();
            if (closeButton != null) {
                closeButton.setVisibility(!enabled && contentVisible()
                        ? View.VISIBLE : View.GONE);
            }
            if (fullscreenExitButton != null) {
                fullscreenExitButton.setVisibility(enabled ? View.VISIBLE : View.GONE);
                if (enabled) fullscreenExitButton.bringToFront();
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
        if (contentView != null) contentView.setVisibility(View.GONE);
        if (storeView != null) storeView.setVisibility(View.GONE);
        if (closeButton != null) closeButton.setVisibility(View.GONE);
        activeFixedTab = null;
        if (activeTerminal == null && !terminals.isEmpty()) {
            activeTerminal = terminals.values().iterator().next();
        }
        for (TerminalView view : terminals.values()) {
            view.setVisibility(view == activeTerminal ? View.VISIBLE : View.GONE);
        }
        styleTabs();
        if (activeTerminal != null) activeTerminal.requestFocus();
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
        if (contentVisible()) {
            hideContent();
            return;
        }
        if (storeVisible()) {
            hideStore();
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
        for (WebView aw : automationWebViews.values()) {
            ViewGroup container = (ViewGroup) aw.getParent();
            if (container != null && container.getParent() instanceof ViewGroup) {
                ((ViewGroup) container.getParent()).removeView(container);
            }
            aw.destroy();
        }
        automationWebViews.clear();
        automationRefCount = 0;
        if (contentView != null) contentView.destroy();
        super.onDestroy();
    }
}
