package dev.g4f.android.container;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Base64;
import android.util.Log;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/** Foreground owner of the Debian container, its PTY sessions and wake lock. */
public final class ContainerService extends Service {

    public interface Listener {
        void onProgress(String message);

        void onState(String state);

        void onTerminalData(int sessionId, byte[] data);

        void onTerminalExit(int sessionId, int exitCode, boolean stopped);

        void onSessionsChanged();
    }

    public final class LocalBinder extends Binder {
        public ContainerService getService() {
            return ContainerService.this;
        }
    }

    private static final String TAG = "g4f-container";
    private static final String CHANNEL_ID = "g4f-container";
    private static final int NOTIFICATION_ID = 4701;

    private final LocalBinder binder = new LocalBinder();
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private final AtomicInteger nextSessionId = new AtomicInteger(1);
    private final Map<Integer, TerminalSession> sessions = new LinkedHashMap<>();
    private final Map<Integer, String> sessionLabels = new LinkedHashMap<>();
    private final List<Listener> listeners = new ArrayList<>();
    private final Set<Integer> stoppedSessions = new HashSet<>();

    private ContainerManager manager;
    private PowerManager.WakeLock wakeLock;
    private volatile boolean installing;
    private volatile boolean servicesStopped;
    private volatile String provisioningError;

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            manager = new ContainerManager(this);
        } catch (Exception error) {
            Log.e(TAG, "Cannot initialize container manager", error);
            notifyState("Error: " + error.getMessage());
        }
        PowerManager powerManager = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (powerManager != null) {
            wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "g4f:container");
            wakeLock.acquire();
        }
        startForeground(NOTIFICATION_ID, buildNotification("Container service starting"));
        notifyState(stateText());
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        closeAllSessions();
        if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        executor.shutdownNow();
        super.onDestroy();
    }

    public ContainerManager manager() {
        return manager;
    }

    public void registerListener(Listener listener) {
        synchronized (listeners) {
            if (!listeners.contains(listener)) listeners.add(listener);
        }
        notifyState(stateText());
        notifySessionsChanged();
    }

    public void unregisterListener(Listener listener) {
        synchronized (listeners) {
            listeners.remove(listener);
        }
    }

    /** Bring up the API and Desktop when an already-provisioned app reconnects. */
    public void autoStartIfReady() {
        if (manager == null || !manager.isInstalled() || !manager.isProvisioned()) return;
        if (installing || isProvisioningRunning() || servicesStopped) return;
        try {
            manager.writeLaunchers();
        } catch (Exception error) {
            Log.w(TAG, "Could not refresh launchers", error);
        }
        autoStartServices();
    }

    public String stateText() {
        if (manager == null) return "Container unavailable";
        if (installing) return "Installing container…";
        if (!manager.isInstalled()) return "Debian container not installed";
        if (provisioningError != null) return provisioningError;
        if (!manager.isProvisioned()) return "Provisioning gpt4free…";
        if (servicesStopped) return "Container stopped — tap API or Desktop to start";
        return manager.isDesktopInstalled()
                ? "Ready — desktop installed"
                : "Ready — desktop optional";
    }

    /** Download and extract the filesystem. Runs off the main thread. */
    public void install(final boolean withDesktop) {
        if (manager == null || installing) return;
        if (isProvisioningRunning()) {
            notifyProgress("Provisioning is still running; wait before reinstalling");
            return;
        }
        provisioningError = null;
        servicesStopped = false;
        installing = true;
        notifyState(stateText());
        executor.execute(() -> {
            try {
                manager.install(withDesktop, message -> {
                    for (Listener listener : snapshotListeners()) listener.onProgress(message);
                });
                notifyState(stateText());
                if (!manager.isProvisioned()) {
                    provision(withDesktop);
                } else if (withDesktop && !manager.isDesktopInstalled()) {
                    // A prebuilt overlay already contains the base environment.
                    provisionDesktopOnly();
                } else {
                    // Fully provisioned: bring up the API (and Desktop when
                    // requested) as soon as the container finishes loading.
                    manager.writeLaunchers();
                    autoStartServices();
                }
            } catch (Exception error) {
                Log.e(TAG, "Install failed", error);
                notifyState("Install failed: " + error.getMessage());
                notifyProgress("Install failed: " + error.getMessage());
            } finally {
                installing = false;
                notifyState(stateText());
            }
        });
    }

    /** Run apt/pip provisioning inside a visible terminal session. */
    public void provision(boolean withDesktop) {
        if (manager == null || !manager.isInstalled()) return;
        provisioningError = null;
        notifyState(stateText());
        String command = "bash /g4f-scripts/provision-base.sh";
        if (withDesktop) {
            command += " && bash /g4f-scripts/provision-desktop.sh";
        }
        startSession("setup", command);
    }

    /** Desktop-only provisioning for installs that used a prebuilt base. */
    private void provisionDesktopOnly() {
        if (manager == null || !manager.isInstalled()) return;
        if (manager.isDesktopInstalled()) return;
        provisioningError = null;
        startSession("desktop-setup", "bash /g4f-scripts/provision-desktop.sh");
    }

    /** Start API and, when installed, Desktop exactly once after provisioning. */
    private void autoStartServices() {
        if (manager == null || !manager.isInstalled() || !manager.isProvisioned()) return;
        if (isProvisioningRunning()) return;
        try {
            ensureSession("api");
            if (manager.isDesktopInstalled()) {
                ensureSession("desktop");
            }
        } catch (IllegalStateException error) {
            Log.i(TAG, "Auto-start deferred: " + error.getMessage());
        }
    }

    /** Return the existing session for a label, or start it if it is missing. */
    public int ensureSession(String kind) {
        String label = sessionLabel(kind);
        synchronized (sessions) {
            for (Map.Entry<Integer, String> entry : sessionLabels.entrySet()) {
                if (label.equals(entry.getValue())) return entry.getKey();
            }
        }
        return newSession(kind);
    }

    /** "id:label,id:label" so the terminal can restore singleton tabs. */
    public String sessionList() {
        StringBuilder result = new StringBuilder();
        synchronized (sessions) {
            for (Map.Entry<Integer, String> entry : sessionLabels.entrySet()) {
                if (result.length() > 0) result.append(',');
                result.append(entry.getKey()).append(':').append(entry.getValue());
            }
        }
        return result.toString();
    }

    private static String sessionLabel(String kind) {
        if ("api".equals(kind)) return "api";
        if ("desktop".equals(kind)) return "desktop";
        if ("setup".equals(kind)) return "setup";
        return "shell";
    }

    public int newSession(String kind) {
        if (manager == null || !manager.isInstalled()) {
            throw new IllegalStateException("Install the container first");
        }
        if (isProvisioningRunning()) {
            throw new IllegalStateException("Setup is still running; wait before starting a session");
        }
        servicesStopped = false;
        switch (kind == null ? "shell" : kind) {
            case "api":
                return startSession("api", "bash /g4f-scripts/start-api.sh");
            case "desktop":
                return startSession("desktop", "bash /g4f-scripts/start-desktop.sh");
            case "setup":
                return startSession("setup", "bash /g4f-scripts/provision-base.sh");
            case "shell":
            default:
                return startSession("shell", "bash -l");
        }
    }

    private int startSession(String label, String shellCommand) {
        final int id = nextSessionId.getAndIncrement();
        final String[] argv = manager.prootArgv(shellCommand);
        final String[] env = manager.prootEnv();
        TerminalSession session = TerminalSession.open(id, argv, env, 80, 24,
                new TerminalSession.Callback() {
                    @Override
                    public void onData(byte[] data) {
                        for (Listener listener : snapshotListeners()) {
                            listener.onTerminalData(id, data);
                        }
                    }

                    @Override
                    public void onExit(int exitCode) {
                        synchronized (sessions) {
                            sessions.remove(id);
                            sessionLabels.remove(id);
                        }
                        boolean stopped;
                        synchronized (stoppedSessions) {
                            stopped = stoppedSessions.remove(id);
                        }
                        notifySessionsChanged();
                        String message = stopped
                                ? label + " stopped"
                                : label + " exited with code " + exitCode;
                        Log.i(TAG, message);
                        notifyProgress(message);
                        // A failed setup must not leave the UI saying
                        // "Provisioning…" forever. Show a retry hint instead.
                        if ("setup".equals(label) && manager != null) {
                            if (manager.isProvisioned()) {
                                provisioningError = null;
                                try {
                                    manager.writeLaunchers();
                                } catch (Exception error) {
                                    Log.w(TAG, "Could not refresh launchers", error);
                                }
                                autoStartServices();
                            } else {
                                String reason = exitCode == 0
                                        ? "incomplete (exit 0)"
                                        : "failed (exit " + exitCode + ")";
                                provisioningError = "Provisioning " + reason + " — tap Install to retry";
                                notifyState(provisioningError);
                            }
                        } else if ("desktop-setup".equals(label) && manager != null) {
                            if (manager.isDesktopInstalled()) {
                                provisioningError = null;
                                try {
                                    manager.writeLaunchers();
                                    manager.installNovncAssets();
                                } catch (Exception error) {
                                    Log.w(TAG, "Could not refresh desktop", error);
                                }
                                autoStartServices();
                            } else {
                                provisioningError = "Desktop setup failed (exit " + exitCode +
                                        ") — tap Desktop to retry";
                            }
                            notifyState(stateText());
                        }
                        for (Listener listener : snapshotListeners()) {
                            listener.onTerminalExit(id, exitCode, stopped);
                        }
                    }
                });
        synchronized (sessions) {
            sessions.put(id, session);
            sessionLabels.put(id, label);
        }
        notifySessionsChanged();
        return id;
    }

    public void write(int sessionId, String base64) {
        TerminalSession session = findSession(sessionId);
        if (session != null) {
            session.write(Base64.decode(base64, Base64.DEFAULT));
        }
    }

    public void resize(int sessionId, int cols, int rows) {
        TerminalSession session = findSession(sessionId);
        if (session != null) session.resize(cols, rows);
    }

    public void close(int sessionId) {
        if ("setup".equals(sessionLabels.get(sessionId))) {
            notifyProgress("Setup is still running; it cannot be closed");
            return;
        }
        TerminalSession session;
        synchronized (sessions) {
            session = sessions.remove(sessionId);
            sessionLabels.remove(sessionId);
        }
        if (session != null) {
            synchronized (stoppedSessions) {
                stoppedSessions.add(sessionId);
            }
            session.close();
            synchronized (sessions) {
                servicesStopped = sessions.isEmpty();
            }
            notifySessionsChanged();
            notifyState(stateText());
        }
    }

    public void closeAllSessions() {
        synchronized (sessions) {
            for (Integer id : new ArrayList<>(sessions.keySet())) {
                if ("setup".equals(sessionLabels.get(id))) continue;
                TerminalSession session = sessions.remove(id);
                sessionLabels.remove(id);
                if (session != null) {
                    synchronized (stoppedSessions) {
                        stoppedSessions.add(id);
                    }
                    session.close();
                }
            }
        }
        synchronized (sessions) {
            servicesStopped = sessions.isEmpty();
        }
        notifySessionsChanged();
        notifyProgress(servicesStopped ? "Container stopped" : "Service sessions stopped");
        notifyState(stateText());
    }

    private boolean isSetupRunning() {
        synchronized (sessions) {
            return sessionLabels.containsValue("setup");
        }
    }

    private boolean isProvisioningRunning() {
        synchronized (sessions) {
            return sessionLabels.containsValue("setup")
                    || sessionLabels.containsValue("desktop-setup");
        }
    }

    @Nullable
    private TerminalSession findSession(int sessionId) {
        synchronized (sessions) {
            return sessions.get(sessionId);
        }
    }

    private List<Listener> snapshotListeners() {
        synchronized (listeners) {
            return new ArrayList<>(listeners);
        }
    }

    private void notifyProgress(String message) {
        for (Listener listener : snapshotListeners()) listener.onProgress(message);
    }

    private void notifyState(String state) {
        for (Listener listener : snapshotListeners()) listener.onState(state);
    }

    private void notifySessionsChanged() {
        for (Listener listener : snapshotListeners()) listener.onSessionsChanged();
    }

    private Notification buildNotification(String text) {
        Intent openIntent = new Intent(this, ContainerActivity.class);
        int flags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) flags |= PendingIntent.FLAG_IMMUTABLE;
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, openIntent, flags);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationManager managerService =
                    (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (managerService != null && managerService.getNotificationChannel(CHANNEL_ID) == null) {
                NotificationChannel channel = new NotificationChannel(
                        CHANNEL_ID, "G4F container", NotificationManager.IMPORTANCE_LOW);
                channel.setDescription("Debian container and terminal sessions");
                managerService.createNotificationChannel(channel);
            }
            return new Notification.Builder(this, CHANNEL_ID)
                    .setSmallIcon(android.R.drawable.ic_dialog_info)
                    .setContentTitle("G4F Container")
                    .setContentText(text)
                    .setContentIntent(pendingIntent)
                    .setOngoing(true)
                    .build();
        }
        return new Notification.Builder(this)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("G4F Container")
                .setContentText(text)
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();
    }
}
