package dev.g4f.android.mcp;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Model Context Protocol server for on-device actions.
 *
 * <p>Listens on loopback only (127.0.0.1:1338) and speaks JSON-RPC 2.0 over a
 * single HTTP endpoint, {@code POST /mcp}. It implements the MCP lifecycle
 * methods needed by clients ({@code initialize}, {@code tools/list},
 * {@code tools/call}, {@code ping}) and returns a standard CallToolResult.
 * Actions run on the main thread because several of them start Android
 * activities. The g4f API on port 1337 has its own, separate {@code /mcp}
 * route; a Python bridge forwards mobile tool calls to this server.
 */
public final class MobileMcpServer {
    public static final int PORT = 1338;
    public static final String ENDPOINT = "http://127.0.0.1:" + PORT + "/mcp";
    private static final String TAG = "g4f-mobile-mcp";
    private static final String DEFAULT_PROTOCOL_VERSION = "2024-11-05";
    private static final Set<String> SUPPORTED_VERSIONS = new HashSet<>(Arrays.asList(
            "2024-11-05", "2025-03-26", "2025-06-18"));
    private static final int MAX_BODY_BYTES = 256 * 1024;
    private static final long ACTION_TIMEOUT_S = 10L;

    private static volatile MobileMcpServer instance;

    private final Context appContext;
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private ServerSocket serverSocket;
    private Thread acceptThread;
    private volatile boolean running;

    private MobileMcpServer(Context context) {
        appContext = context.getApplicationContext();
    }

    public static MobileMcpServer getInstance(Context context) {
        MobileMcpServer result = instance;
        if (result == null) {
            synchronized (MobileMcpServer.class) {
                result = instance;
                if (result == null) {
                    result = new MobileMcpServer(context);
                    instance = result;
                }
            }
        }
        return result;
    }

    public synchronized void start() {
        if (running) return;
        try {
            ServerSocket socket = new ServerSocket();
            socket.setReuseAddress(true);
            socket.bind(new InetSocketAddress("127.0.0.1", PORT), 16);
            serverSocket = socket;
            running = true;
            acceptThread = new Thread(this::acceptLoop, "g4f-mobile-mcp");
            acceptThread.setDaemon(true);
            acceptThread.start();
            Log.i(TAG, "Mobile MCP server listening on " + ENDPOINT);
        } catch (IOException e) {
            Log.w(TAG, "Could not start Mobile MCP server on port " + PORT, e);
        }
    }

    public synchronized void stop() {
        running = false;
        ServerSocket socket = serverSocket;
        serverSocket = null;
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException ignored) {
                // Already closed.
            }
        }
        Log.i(TAG, "Mobile MCP server stopped");
    }

    public boolean isRunning() {
        return running;
    }

    private void acceptLoop() {
        while (running) {
            ServerSocket listener = serverSocket;
            if (listener == null) return;
            try {
                Socket client = listener.accept();
                client.setSoTimeout(10_000);
                executor.execute(() -> handleConnection(client));
            } catch (IOException e) {
                if (running) Log.w(TAG, "Accept failed", e);
            }
        }
    }

    private void handleConnection(Socket socket) {
        try (Socket client = socket;
             InputStream input = client.getInputStream();
             OutputStream output = client.getOutputStream()) {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(input, StandardCharsets.ISO_8859_1));
            String requestLine = reader.readLine();
            if (requestLine == null || requestLine.isEmpty()) return;
            String[] parts = requestLine.split(" ");
            if (parts.length < 2) {
                writeResponse(output, 400, "Bad Request", plain("Bad request"), null, null);
                return;
            }
            String method = parts[0].toUpperCase(Locale.US);
            String path = parts[1].split("\\?", 2)[0];

            Headers headers = new Headers(reader);
            String origin = headers.get("origin");
            boolean localOrigin = isLocalOrigin(origin);
            if (origin != null && !localOrigin) {
                respond(output, 403, errorResponse(null, -32000, "Forbidden origin"), null);
                return;
            }

            if ("OPTIONS".equals(method)) {
                writeResponse(output, 204, "No Content", new byte[0],
                        localOrigin ? origin : null, null);
                return;
            }
            if ("/health".equals(path) && "GET".equals(method)) {
                respond(output, 200, healthResponse(), localOrigin ? origin : null);
                return;
            }
            if ("/mcp".equals(path) && "GET".equals(method)) {
                respond(output, 200, mcpInfoResponse(), localOrigin ? origin : null);
                return;
            }
            if (!"/mcp".equals(path)) {
                respond(output, 404, errorResponse(null, -32600, "Not found"),
                        localOrigin ? origin : null);
                return;
            }
            if (!"POST".equals(method)) {
                writeResponse(output, 405, "Method Not Allowed", plain("Use POST"),
                        localOrigin ? origin : null, "POST, OPTIONS");
                return;
            }

            String lengthHeader = headers.get("content-length");
            int length = 0;
            if (lengthHeader != null) {
                try {
                    length = Integer.parseInt(lengthHeader.trim());
                } catch (NumberFormatException e) {
                    respond(output, 400, errorResponse(null, -32700, "Invalid Content-Length"),
                            localOrigin ? origin : null);
                    return;
                }
            }
            if (length < 0 || length > MAX_BODY_BYTES) {
                respond(output, 413, errorResponse(null, -32700, "Request body too large"),
                        localOrigin ? origin : null);
                return;
            }
            if ("chunked".equalsIgnoreCase(headers.get("transfer-encoding"))) {
                respond(output, 501, errorResponse(null, -32600, "Chunked body is not supported"),
                        localOrigin ? origin : null);
                return;
            }
            if ("100-continue".equalsIgnoreCase(headers.get("expect"))) {
                output.write("HTTP/1.1 100 Continue\r\n\r\n".getBytes(StandardCharsets.US_ASCII));
                output.flush();
            }
            byte[] body = readExactly(input, length);
            String version = headers.get("mcp-protocol-version");
            if (version != null && !SUPPORTED_VERSIONS.contains(version)) {
                respond(output, 400,
                        errorResponse(null, -32600, "Unsupported MCP-Protocol-Version"),
                        localOrigin ? origin : null);
                return;
            }
            JSONObject response = handleJsonRpc(body, version);
            if (response == null) {
                // Notification: 202 Accepted with no JSON-RPC body.
                writeResponse(output, 202, "Accepted", new byte[0],
                        localOrigin ? origin : null, null);
            } else {
                respond(output, 200, response, localOrigin ? origin : null);
            }
        } catch (SocketTimeoutException e) {
            Log.d(TAG, "Client timed out");
        } catch (Exception e) {
            Log.w(TAG, "Request failed", e);
        }
    }

    private JSONObject handleJsonRpc(byte[] body, String requestedVersion) {
        Object id = null;
        try {
            JSONObject request = new JSONObject(new String(body, StandardCharsets.UTF_8));
            boolean notification = !request.has("id");
            id = request.opt("id");
            String method = request.optString("method", "");
            JSONObject params = request.optJSONObject("params");
            if (params == null) params = new JSONObject();

            if (method.startsWith("notifications/")) return null;
            JSONObject result;
            switch (method) {
                case "initialize": {
                    String clientVersion = params.optString("protocolVersion", "");
                    String version = SUPPORTED_VERSIONS.contains(clientVersion)
                            ? clientVersion : (requestedVersion != null
                                    ? requestedVersion : DEFAULT_PROTOCOL_VERSION);
                    result = new JSONObject()
                            .put("protocolVersion", version)
                            .put("capabilities", new JSONObject()
                                    .put("tools", new JSONObject().put("listChanged", false)))
                            .put("serverInfo", new JSONObject()
                                    .put("name", "g4f-mobile-mcp-server")
                                    .put("version", "1.0.0"))
                            .put("instructions", "On-device actions. Intent-based actions open "
                                    + "a native app where the user can review or cancel.");
                    break;
                }
                case "ping":
                    result = new JSONObject();
                    break;
                case "tools/list": {
                    JSONArray tools = new JSONArray();
                    for (MobileActions.Tool tool : MobileActions.tools().values()) {
                        tools.put(new JSONObject()
                                .put("name", tool.name())
                                .put("description", tool.description())
                                .put("inputSchema", tool.inputSchema()));
                    }
                    result = new JSONObject().put("tools", tools);
                    break;
                }
                case "tools/call":
                    result = callTool(params);
                    break;
                case "resources/list":
                    result = new JSONObject().put("resources", new JSONArray());
                    break;
                case "resources/templates/list":
                    result = new JSONObject().put("resourceTemplates", new JSONArray());
                    break;
                case "prompts/list":
                    result = new JSONObject().put("prompts", new JSONArray());
                    break;
                case "logging/setLevel":
                    result = new JSONObject();
                    break;
                default:
                    return errorResponse(id, -32601, "Method not found: " + method);
            }
            if (notification) return null;
            return new JSONObject().put("jsonrpc", "2.0")
                    .put("id", id == null ? JSONObject.NULL : id)
                    .put("result", result);
        } catch (JSONException | IllegalArgumentException e) {
            return errorResponse(id, -32600, "Invalid request: " + e.getMessage());
        } catch (Exception e) {
            Log.w(TAG, "Tool dispatch failed", e);
            return errorResponse(id, -32603, "Internal error: " + e.getClass().getSimpleName());
        }
    }

    private JSONObject callTool(JSONObject params) {
        String name = params.optString("name", "");
        MobileActions.Tool tool = MobileActions.tools().get(name);
        if (tool == null) {
            return toolResult(true, errorPayload("Unknown tool: " + name));
        }
        JSONObject args = params.optJSONObject("arguments");
        if (args == null) {
            if (params.has("arguments") && !(params.opt("arguments") instanceof JSONObject)) {
                return toolResult(true, errorPayload("arguments must be an object"));
            }
            args = new JSONObject();
        }
        JSONObject payload;
        try {
            payload = executeOnMainThread(tool, args);
        } catch (IllegalArgumentException e) {
            return toolResult(true, errorPayload(e.getMessage()));
        } catch (Exception e) {
            Log.w(TAG, "Action failed: " + name, e);
            return toolResult(true, errorPayload(
                    "Action failed: " + e.getClass().getSimpleName()));
        }
        return toolResult(payload.has("error"), payload);
    }

    /** Most actions start activities or touch camera state; run them on the UI thread. */
    private JSONObject executeOnMainThread(final MobileActions.Tool tool, final JSONObject args)
            throws Exception {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return tool.execute(appContext, args);
        }
        final Object[] result = new Object[1];
        final Throwable[] failure = new Throwable[1];
        CountDownLatch done = new CountDownLatch(1);
        boolean posted = new Handler(Looper.getMainLooper()).post(() -> {
            try {
                result[0] = tool.execute(appContext, args);
            } catch (Throwable e) {
                failure[0] = e;
            } finally {
                done.countDown();
            }
        });
        if (!posted) throw new IllegalStateException("Main thread is not accepting work");
        if (!done.await(ACTION_TIMEOUT_S, TimeUnit.SECONDS)) {
            throw new IllegalStateException("Action timed out");
        }
        if (failure[0] instanceof Exception) throw (Exception) failure[0];
        if (failure[0] instanceof Error) throw (Error) failure[0];
        return (JSONObject) result[0];
    }

    private static JSONObject errorPayload(String message) {
        try {
            return new JSONObject().put("error", message);
        } catch (JSONException e) {
            throw new IllegalStateException(e);
        }
    }

    private static JSONObject toolResult(boolean isError, JSONObject payload) {
        String text = payload.toString();
        try {
            return new JSONObject()
                    .put("content", new JSONArray().put(new JSONObject()
                            .put("type", "text").put("text", text)))
                    .put("structuredContent", payload)
                    .put("isError", isError);
        } catch (JSONException e) {
            throw new IllegalStateException(e);
        }
    }

    private JSONObject mcpInfoResponse() {
        JSONArray tools = new JSONArray();
        for (MobileActions.Tool tool : MobileActions.tools().values()) {
            tools.put(tool.name());
        }
        try {
            return new JSONObject()
                    .put("protocol", "MCP")
                    .put("version", "2025-06-18")
                    .put("endpoint", ENDPOINT)
                    .put("transport", "json-rpc-2.0/http")
                    .put("serverInfo", new JSONObject()
                            .put("name", "g4f-mobile-mcp-server")
                            .put("version", "1.0.0"))
                    .put("capabilities", new JSONObject()
                            .put("tools", new JSONObject().put("listChanged", false)))
                    .put("tools", tools);
        } catch (JSONException e) {
            throw new IllegalStateException(e);
        }
    }

    private JSONObject healthResponse() {
        JSONArray names = new JSONArray();
        for (String name : MobileActions.tools().keySet()) names.put(name);
        try {
            return new JSONObject()
                    .put("status", "ok")
                    .put("endpoint", ENDPOINT)
                    .put("protocol", "json-rpc-2.0")
                    .put("tools", names);
        } catch (JSONException e) {
            throw new IllegalStateException(e);
        }
    }

    private static JSONObject errorResponse(Object id, int code, String message) {
        try {
            return new JSONObject()
                    .put("jsonrpc", "2.0")
                    .put("id", id == null ? JSONObject.NULL : id)
                    .put("error", new JSONObject().put("code", code).put("message", message));
        } catch (JSONException e) {
            throw new IllegalStateException(e);
        }
    }

    private static boolean isLocalOrigin(String origin) {
        if (origin == null || origin.isEmpty()) return false;
        try {
            URI uri = new URI(origin);
            String host = uri.getHost();
            return ("127.0.0.1".equals(host) || "localhost".equals(host) || "[::1]".equals(host))
                    && ("http".equalsIgnoreCase(uri.getScheme())
                        || "https".equalsIgnoreCase(uri.getScheme()));
        } catch (URISyntaxException e) {
            return false;
        }
    }

    private static byte[] readExactly(InputStream input, int length) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(length);
        byte[] buffer = new byte[4096];
        int remaining = length;
        while (remaining > 0) {
            int read = input.read(buffer, 0, Math.min(buffer.length, remaining));
            if (read < 0) break;
            out.write(buffer, 0, read);
            remaining -= read;
        }
        return out.toByteArray();
    }

    private static byte[] plain(String message) {
        return message.getBytes(StandardCharsets.UTF_8);
    }

    private static void respond(OutputStream output, int status, JSONObject body, String origin)
            throws IOException {
        writeResponse(output, status, reason(status), body.toString().getBytes(StandardCharsets.UTF_8),
                origin, null, "application/json; charset=utf-8");
    }

    private static void writeResponse(OutputStream output, int status, String reason, byte[] body,
                                      String origin, String allowMethods) throws IOException {
        writeResponse(output, status, reason, body, origin, allowMethods,
                "text/plain; charset=utf-8");
    }

    private static void writeResponse(OutputStream output, int status, String reason, byte[] body,
                                      String origin, String allowMethods, String contentType)
            throws IOException {
        StringBuilder headers = new StringBuilder()
                .append("HTTP/1.1 ").append(status).append(' ').append(reason).append("\r\n")
                .append("Content-Type: ").append(contentType).append("\r\n")
                .append("Content-Length: ").append(body.length).append("\r\n")
                .append("Cache-Control: no-store\r\n")
                .append("Connection: close\r\n");
        if (origin != null) {
            headers.append("Access-Control-Allow-Origin: ").append(origin).append("\r\n")
                    .append("Access-Control-Allow-Methods: ")
                    .append(allowMethods == null ? "POST, GET, OPTIONS" : allowMethods)
                    .append("\r\n")
                    .append("Access-Control-Allow-Headers: Content-Type, MCP-Protocol-Version\r\n");
        } else if (allowMethods != null) {
            headers.append("Allow: ").append(allowMethods).append("\r\n");
        }
        headers.append("\r\n");
        output.write(headers.toString().getBytes(StandardCharsets.US_ASCII));
        output.write(body);
        output.flush();
    }

    private static String reason(int status) {
        switch (status) {
            case 200: return "OK";
            case 202: return "Accepted";
            case 204: return "No Content";
            case 400: return "Bad Request";
            case 403: return "Forbidden";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 413: return "Payload Too Large";
            case 501: return "Not Implemented";
            default: return "Internal Server Error";
        }
    }

    /** Minimal case-insensitive HTTP header reader. */
    private static final class Headers {
        private final java.util.Map<String, String> values = new java.util.HashMap<>();

        Headers(BufferedReader reader) throws IOException {
            String line;
            while ((line = reader.readLine()) != null && !line.isEmpty()) {
                int colon = line.indexOf(':');
                if (colon > 0) {
                    values.put(line.substring(0, colon).trim().toLowerCase(Locale.US),
                            line.substring(colon + 1).trim());
                }
            }
        }

        String get(String name) {
            return values.get(name);
        }
    }
}
