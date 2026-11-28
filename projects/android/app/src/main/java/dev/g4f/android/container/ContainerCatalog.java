package dev.g4f.android.container;

import android.content.Context;
import android.os.Build;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Parsed view of assets/container/catalog.json. */
public final class ContainerCatalog {

    public static final class Asset {
        public final String url;
        public final String sha256;
        public final long bytes;

        Asset(String url, String sha256, long bytes) {
            this.url = url;
            this.sha256 = sha256;
            this.bytes = bytes;
        }
    }

    public final String distributionName;
    public final Asset rootfs;
    public final Asset proot;
    public final Asset libtalloc;
    public final Asset libandroidShmem;
    /** Optional overlay with pre-provisioned launchers already installed. */
    public final Asset prebuilt;
    /** Optional overlay with XFCE, TigerVNC and noVNC already installed. */
    public final Asset desktopPrebuilt;
    public final int apiPort;
    public final int vncPort;
    public final int novncPort;
    public final List<String> basePackages;
    public final List<String> desktopPackages;

    private ContainerCatalog(JSONObject json) throws JSONException {
        JSONObject distro = json.getJSONObject("distribution");
        distributionName = distro.getString("name");
        String abi = selectAbi(distro.getJSONObject("rootfs"));
        rootfs = parseAsset(distro.getJSONObject("rootfs").getJSONObject(abi));

        JSONObject runtime = json.getJSONObject("runtime");
        proot = parsePackage(runtime.getJSONObject("proot"), abi);
        libtalloc = parsePackage(runtime.getJSONObject("libtalloc"), abi);
        libandroidShmem = parsePackage(runtime.getJSONObject("libandroid-shmem"), abi);

        JSONObject g4f = json.getJSONObject("g4f");
        apiPort = g4f.getInt("api_port");
        vncPort = g4f.getInt("vnc_port");
        novncPort = g4f.getInt("novnc_port");
        prebuilt = g4f.isNull("prebuilt") ? null : parseAsset(g4f.getJSONObject("prebuilt"));
        desktopPrebuilt = g4f.isNull("desktop_prebuilt")
                ? null : parseAsset(g4f.getJSONObject("desktop_prebuilt"));
        basePackages = parseList(g4f.getJSONArray("base_packages"));
        desktopPackages = parseList(g4f.getJSONArray("desktop_packages"));
    }

    public static ContainerCatalog load(Context context) throws IOException, JSONException {
        byte[] raw;
        try (InputStream in = context.getAssets().open("container/catalog.json")) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int count;
            while ((count = in.read(buffer)) != -1) {
                out.write(buffer, 0, count);
            }
            raw = out.toByteArray();
        }
        return new ContainerCatalog(new JSONObject(new String(raw, StandardCharsets.UTF_8)));
    }

    private static Asset parseAsset(JSONObject json) throws JSONException {
        return new Asset(json.getString("url"), json.getString("sha256"), json.optLong("bytes", 0L));
    }

    private static Asset parsePackage(JSONObject json, String abi) throws JSONException {
        return parseAsset(json.getJSONObject("packages").getJSONObject(abi));
    }

    private static List<String> parseList(JSONArray array) throws JSONException {
        List<String> result = new ArrayList<>();
        for (int i = 0; i < array.length(); i++) {
            result.add(array.getString(i));
        }
        return result;
    }

    /** Pick the first ABI for which the catalog has every required asset. */
    private static String selectAbi(JSONObject rootfs) throws JSONException {
        List<String> preferred = new ArrayList<>();
        if (Build.SUPPORTED_ABIS != null) {
            preferred.addAll(Arrays.asList(Build.SUPPORTED_ABIS));
        }
        preferred.add("arm64-v8a");
        preferred.add("x86_64");
        for (String abi : preferred) {
            if (rootfs.has(abi)) {
                return abi;
            }
        }
        throw new JSONException("No supported ABI in rootfs catalog");
    }
}
