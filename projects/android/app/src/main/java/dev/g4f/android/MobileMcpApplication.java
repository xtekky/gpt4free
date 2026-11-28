package dev.g4f.android;

import android.app.Application;
import android.util.Log;

import dev.g4f.android.mcp.MobileMcpServer;

/** Starts the loopback-only mobile MCP server for the whole app process. */
public final class MobileMcpApplication extends Application {
    private static final String TAG = "g4f-app";

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            MobileMcpServer.getInstance(this).start();
        } catch (Exception e) {
            // A port conflict must never prevent the app from opening.
            Log.w(TAG, "Could not start the mobile MCP server", e);
        }
    }
}
