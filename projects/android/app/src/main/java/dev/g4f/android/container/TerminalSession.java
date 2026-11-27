package dev.g4f.android.container;

import android.os.Handler;
import android.os.Looper;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** One PRoot process attached to a native pseudo-terminal. */
public final class TerminalSession {

    static {
        System.loadLibrary("g4f_pty");
    }

    public interface Callback {
        void onData(byte[] data);

        void onExit(int exitCode);
    }

    private final int id;
    private final long handle;
    private final Callback callback;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Object lock = new Object();
    private final Thread readerThread;
    private volatile boolean closed;
    private boolean disposed;

    private TerminalSession(int id, long handle, Callback callback) {
        this.id = id;
        this.handle = handle;
        this.callback = callback;
        this.readerThread = new Thread(this::readLoop, "g4f-pty-read-" + id);
        this.readerThread.start();
    }

    public static TerminalSession open(int id, String[] argv, String[] env,
                                       int cols, int rows, Callback callback) {
        long handle = nativeOpen(argv, env, cols, rows);
        if (handle == 0) {
            throw new IllegalStateException("Could not allocate a PTY for " + argv[0]);
        }
        return new TerminalSession(id, handle, callback);
    }

    public int id() {
        return id;
    }

    public void write(String text) {
        write(text.getBytes(StandardCharsets.UTF_8));
    }

    public void write(byte[] bytes) {
        if (bytes.length == 0) return;
        synchronized (lock) {
            if (!closed && !disposed) nativeWrite(handle, bytes);
        }
    }

    public void resize(int cols, int rows) {
        synchronized (lock) {
            if (!closed && !disposed) nativeResize(handle, cols, rows);
        }
    }

    public boolean isAlive() {
        synchronized (lock) {
            return !closed && !disposed && nativeAlive(handle);
        }
    }

    public void close() {
        synchronized (lock) {
            if (closed || disposed) return;
            closed = true;
            nativeClose(handle);
        }
    }

    private void readLoop() {
        byte[] buffer = new byte[8192];
        while (!closed) {
            int count = nativeRead(handle, buffer);
            if (count < 0) break;
            if (count == 0) continue;
            byte[] chunk = Arrays.copyOf(buffer, count);
            mainHandler.post(() -> {
                if (callback != null) callback.onData(chunk);
            });
        }
        int exitCode = -1;
        synchronized (lock) {
            if (!disposed) exitCode = nativeExitCode(handle);
        }
        if (callback != null) {
            final int code = exitCode;
            mainHandler.post(() -> callback.onExit(code));
        }
        synchronized (lock) {
            if (disposed) return;
            disposed = true;
            nativeDispose(handle);
        }
    }

    private static native long nativeOpen(String[] argv, String[] env, int cols, int rows);

    private static native int nativeRead(long handle, byte[] buffer);

    private static native boolean nativeWrite(long handle, byte[] data);

    private static native void nativeResize(long handle, int cols, int rows);

    private static native boolean nativeAlive(long handle);

    private static native int nativeExitCode(long handle);

    private static native void nativeClose(long handle);

    private static native void nativeDispose(long handle);
}
