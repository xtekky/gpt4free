package dev.g4f.android.container;

import android.content.Context;
import android.content.pm.PackageManager;
import android.system.Os;
import android.util.Log;

import org.apache.commons.compress.archivers.ar.ArArchiveEntry;
import org.apache.commons.compress.archivers.ar.ArArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/** Downloads, verifies and extracts the Debian rootfs and PRoot runtime. */
public final class ContainerManager {

    public interface ProgressListener {
        void onProgress(String message);
    }

    private static final String ASSETS_VERSION = "container-v2";
    // normalizeEntryName() strips a leading "./" before applying this prefix.
    private static final String TERMUX_PREFIX = "data/data/com.termux/files/usr/";

    private final Context context;
    private final ContainerCatalog catalog;
    private final File base;
    private final File downloads;
    private final File rootfs;
    private final File runtime;
    private final File scripts;
    private final File binDir;
    private final File resolvConf;
    private final File temp;
    private final File nativeLibraryDir;
    private final File installedMarker;
    private final String versionName;

    public ContainerManager(Context context) throws IOException, org.json.JSONException {
        this.context = context.getApplicationContext();
        String version = "unknown";
        try {
            version = this.context.getPackageManager()
                    .getPackageInfo(this.context.getPackageName(), 0).versionName;
        } catch (PackageManager.NameNotFoundException ignored) {
            // Keep the fallback when the package manager has no version info.
        }
        this.versionName = version != null ? version : "unknown";
        this.catalog = ContainerCatalog.load(this.context);
        this.base = new File(this.context.getFilesDir(), "container");
        this.downloads = new File(base, "downloads");
        this.rootfs = new File(base, "rootfs");
        this.runtime = new File(base, "runtime");
        this.scripts = new File(base, "scripts");
        this.binDir = new File(base, "bin");
        this.resolvConf = new File(base, "resolv.conf");
        this.temp = new File(base, "tmp");
        String nativeDir = this.context.getApplicationInfo().nativeLibraryDir;
        this.nativeLibraryDir = new File(nativeDir != null
                ? nativeDir
                : new File(this.context.getApplicationInfo().dataDir, "lib").getAbsolutePath());
        this.installedMarker = new File(base, ".installed-" + ASSETS_VERSION);
        base.mkdirs();
        downloads.mkdirs();
        rootfs.mkdirs();
        runtime.mkdirs();
        scripts.mkdirs();
        binDir.mkdirs();
        temp.mkdirs();
        // Bind-mounted into every session: dpkg may replace /etc/resolv.conf
        // while packages are configured, which previously broke apt DNS.
        writeResolvConf();
        // Older installs used filesDir/app for the container source. Prepare
        // the dedicated tree here as well, so an existing install can recover
        // without a full reinstall.
        ensureG4fSource();
    }

    public ContainerCatalog catalog() {
        return catalog;
    }

    public File rootfsDir() {
        return rootfs;
    }

    public File runtimeDir() {
        return runtime;
    }

    public File scriptsDir() {
        return scripts;
    }

    /** PRoot must be exec'd from the read-only native library directory. */
    public File prootBinary() {
        return new File(nativeLibraryDir, "libproot.so");
    }

    /** The static loader must also be executable from nativeLibraryDir. */
    public File prootLoader() {
        return new File(nativeLibraryDir, "libproot_loader.so");
    }

    /** Dedicated source tree for the container, separate from Classic mode. */
    public File g4fSourceDir() {
        return new File(base, "g4f-src");
    }

    public boolean isInstalled() {
        return installedMarker.exists()
                && new File(rootfs, "bin/sh").exists()
                && prootBinary().exists()
                && prootLoader().exists();
    }

    public boolean isDesktopInstalled() {
        return new File(scripts, ".provisioned-desktop").exists();
    }

    public boolean isProvisioned() {
        return new File(scripts, ".provisioned-base").exists();
    }

    /** Download and unpack everything needed to start a container.
     *  When a prebuilt overlay is pinned in the catalog, apt/pip provisioning
     *  is skipped entirely; otherwise it runs in a visible terminal session. */
    public void install(boolean withDesktop, ProgressListener listener) throws Exception {
        synchronized (ContainerManager.class) {
            // Start from a clean filesystem. Re-extracting over a rootfs left by
            // an interrupted apt run can restore a half-configured dpkg database.
            reset();
            File rootfsArchive = new File(downloads, "rootfs.tar.gz");
            File tallocDeb = new File(downloads, "libtalloc.deb");
            File shmemDeb = new File(downloads, "libandroid-shmem.deb");
            File overlayArchive = new File(downloads, "g4f-prebuilt.tar.gz");
            File desktopArchive = new File(downloads, "g4f-desktop.tar.gz");
            boolean prebuilt = catalog.prebuilt != null;
            boolean desktopPrebuilt = withDesktop && catalog.desktopPrebuilt != null;

            listener.onProgress("Downloading Debian rootfs (" +
                    (catalog.rootfs.bytes / (1024 * 1024)) + " MB)");
            download(catalog.rootfs, rootfsArchive, listener);

            // PRoot and its loader are bundled as native libraries in the APK:
            // Android refuses to exec files from app-writable storage.
            listener.onProgress("Downloading PRoot libraries");
            download(catalog.libtalloc, tallocDeb, listener);
            download(catalog.libandroidShmem, shmemDeb, listener);

            if (prebuilt) {
                listener.onProgress("Downloading prebuilt g4f overlay (" +
                        (catalog.prebuilt.bytes / (1024 * 1024)) + " MB)");
                download(catalog.prebuilt, overlayArchive, listener);
            }
            if (desktopPrebuilt) {
                listener.onProgress("Downloading prebuilt desktop overlay (" +
                        (catalog.desktopPrebuilt.bytes / (1024 * 1024)) + " MB)");
                download(catalog.desktopPrebuilt, desktopArchive, listener);
            }

            listener.onProgress("Extracting Debian rootfs");
            extractTarGz(rootfsArchive, rootfs, "");

            listener.onProgress("Installing PRoot libraries");
            extractDeb(tallocDeb, runtime);
            extractDeb(shmemDeb, runtime);

            listener.onProgress("Preparing container configuration");
            writeResolvConf();
            writeShellProfile();
            copyAssetTree("container/scripts", scripts);
            deleteRecursively(binDir);
            binDir.mkdirs();
            copyAssetTree("container/bin", binDir);
            ensureG4fSource();

            // The prebuilt venv points at /usr/bin/python3; the pinned Debian
            // runtime is bundled in the APK so no apt/pip pass is needed for
            // the base install.
            listener.onProgress("Installing bundled Python runtime");
            copyAssetTree("container/python", rootfs);

            File baseMarker = new File(scripts, ".provisioned-base");
            if (prebuilt) {
                listener.onProgress("Extracting prebuilt g4f overlay");
                extractTarGz(overlayArchive, rootfs, "");
                writeFile(baseMarker, "base provisioning complete\n");
            } else {
                baseMarker.delete();
            }
            // The prebuilt overlay predates the corrected CLI syntax and used
            // a separate websockify server; always write our own launchers.
            writeLaunchers();

            File desktopMarker = new File(scripts, ".provisioned-desktop");
            if (desktopPrebuilt) {
                listener.onProgress("Extracting prebuilt desktop overlay");
                extractTarGz(desktopArchive, rootfs, "");
                installNovncAssets();
                writeFile(desktopMarker, "desktop provisioning complete\n");
            } else {
                desktopMarker.delete();
            }
            // The desktop overlay also ships the old websockify launcher, so
            // re-apply our /desktop relay launcher after extracting it.
            writeLaunchers();
            installedMarker.createNewFile();
            listener.onProgress(prebuilt
                    ? (desktopPrebuilt ? "Container and desktop ready"
                        : withDesktop ? "Container ready. Installing desktop…" : "Container ready")
                    : (withDesktop
                        ? "Container ready. Provisioning base system and desktop…"
                        : "Container ready. Provisioning base system…"));
        }
    }

    /** Clear the installed state so the next install re-downloads everything. */
    public void reset() {
        installedMarker.delete();
        new File(scripts, ".provisioned-base").delete();
        new File(scripts, ".provisioned-desktop").delete();
        deleteRecursively(rootfs);
        deleteRecursively(runtime);
        deleteRecursively(g4fSourceDir());
        rootfs.mkdirs();
        runtime.mkdirs();
    }

    /** Build the argv used to launch PRoot with a command inside Debian. */
    public String[] prootArgv(String shellCommand) {
        List<String> argv = new ArrayList<>();
        argv.add(prootBinary().getAbsolutePath());
        argv.add("--kill-on-exit");
        argv.add("-0");
        argv.add("-r");
        argv.add(rootfs.getAbsolutePath());
        argv.add("-b");
        argv.add("/dev");
        argv.add("-b");
        argv.add("/proc");
        argv.add("-b");
        argv.add("/sys");
        argv.add("-b");
        argv.add("/sdcard");
        argv.add("-b");
        argv.add(g4fSourceDir().getAbsolutePath() + ":/g4f-src");
        argv.add("-b");
        argv.add(scripts.getAbsolutePath() + ":/g4f-scripts");
        argv.add("-b");
        argv.add(nativeLibraryDir.getAbsolutePath() + ":/g4f-native");
        argv.add("-b");
        argv.add(binDir.getAbsolutePath() + ":/g4f-bin");
        argv.add("-b");
        argv.add(resolvConf.getAbsolutePath() + ":/etc/resolv.conf");
        argv.add("-w");
        argv.add("/root");
        argv.add("/bin/bash");
        argv.add("-lc");
        // PRoot itself starts before the container mount exists, so the shim
        // has to be enabled by the shell inside the container, not by the
        // host environment.
        argv.add("export LD_PRELOAD=/g4f-native/libg4f_linkfix.so; "
                + "export G4F_PYTHON=/opt/g4f-venv/bin/python; " + shellCommand);
        return argv.toArray(new String[0]);
    }

    /** Environment for the PRoot process (passed to the native PTY layer). */
    public String[] prootEnv() {
        return new String[] {
                "HOME=/root",
                "PATH=/g4f-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "PYTHONPATH=/g4f-src",
                // g4f-go must use the bundled interpreter; downloading a
                // platform CPython at runtime fails when HTTPS/CA setup is
                // unavailable inside PRoot.
                "G4F_PYTHON=/opt/g4f-venv/bin/python",
                "PYTHONHOME=",
                "TERM=xterm-256color",
                "LANG=C.UTF-8",
                "PROOT_LOADER=" + prootLoader().getAbsolutePath(),
                "PROOT_TMP_DIR=" + temp.getAbsolutePath(),
                "LD_LIBRARY_PATH=" + new File(runtime, "lib").getAbsolutePath(),
                "ANDROID_ROOT=/system",
                "TMPDIR=/tmp",
                "G4F_VERSION=" + versionName,
                "G4F_NO_VERSION_CHECK=1"
        };
    }

    private void writeResolvConf() throws IOException {
        String contents = "nameserver 1.1.1.1\nnameserver 8.8.8.8\n" +
                "options timeout:5 attempts:3\n";
        // The bind-mounted copy survives dpkg rewriting /etc/resolv.conf
        // inside the container during apt installs.
        writeFile(resolvConf, contents);
        File etc = new File(rootfs, "etc");
        etc.mkdirs();
        writeFile(new File(etc, "resolv.conf"), contents);
        writeFile(new File(etc, "hosts"),
                "127.0.0.1 localhost\n::1 localhost\n");
    }

    private void ensureG4fSource() throws IOException {
        File appDir = g4fSourceDir();
        File marker = new File(appDir, ".container-src-v2");
        if (marker.exists()) {
            return;
        }
        // The Gradle copyG4fAssets task also drops setup.py and README.md into
        // assets/g4f, so this becomes a pip-installable source tree. It lives
        // outside filesDir/app because Classic mode extracts its own tree there.
        deleteRecursively(appDir);
        appDir.mkdirs();
        // assets/g4f contains the *contents* of the g4f package plus the
        // pip-installable metadata. Wrap the package in g4f/ for "import g4f",
        // then lift setup.py and friends to the source root for pip.
        File packageDir = new File(appDir, "g4f");
        copyAssetTree("g4f", packageDir);
        for (String name : new String[] { "setup.py", "README.md", "requirements.txt" }) {
            copyFile(new File(packageDir, name), new File(appDir, name));
        }
        marker.createNewFile();
    }

    private void download(ContainerCatalog.Asset asset, File destination,
                          ProgressListener listener) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(asset.url).openConnection();
        connection.setConnectTimeout(20_000);
        connection.setReadTimeout(60_000);
        connection.setInstanceFollowRedirects(true);
        connection.setRequestProperty("User-Agent", "g4f-android-container/2.0");
        connection.connect();
        try {
            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                throw new IOException("HTTP " + status + " for " + asset.url);
            }
            File partial = new File(downloads, destination.getName() + ".part");
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream in = connection.getInputStream();
                 OutputStream out = new FileOutputStream(partial)) {
                byte[] buffer = new byte[64 * 1024];
                int count;
                long total = 0;
                while ((count = in.read(buffer)) != -1) {
                    out.write(buffer, 0, count);
                    digest.update(buffer, 0, count);
                    total += count;
                    if (asset.bytes > 0 && total % (4 * 1024 * 1024) < buffer.length) {
                        listener.onProgress("Downloaded " + (total / (1024 * 1024)) + "/" +
                                (asset.bytes / (1024 * 1024)) + " MB");
                    }
                }
            }
            String actual = hex(digest.digest());
            if (!actual.equalsIgnoreCase(asset.sha256)) {
                partial.delete();
                throw new IOException("Checksum mismatch for " + destination.getName() +
                        ": expected " + asset.sha256 + ", got " + actual);
            }
            if (destination.exists()) destination.delete();
            if (!partial.renameTo(destination)) {
                throw new IOException("Could not move " + partial + " to " + destination);
            }
        } finally {
            connection.disconnect();
        }
    }

    /** Install the branded noVNC client used by the API's /desktop/ route. */
    public void installNovncAssets() throws IOException {
        File novnc = new File(rootfs, "usr/share/novnc");
        deleteRecursively(novnc);
        novnc.mkdirs();
        // Gradle generates the full branded client under scripts/novnc so the
        // same asset tree also feeds G4F_DESKTOP_NOVNC_ASSETS during apt setup.
        copyAssetTree("container/scripts/novnc", novnc);
    }

    /** Re-apply our profile and launchers after install, overlay or reconnect. */
    public void writeLaunchers() throws IOException {
        // Older installs must also pick up new environment fixes (for example
        // clearing PYTHONHOME), so refresh the profile on every reconnect.
        writeShellProfile();
        writeApiLauncher();
        writeDesktopLauncher();
        // g4f-go itself carries part of that fix. Refreshing it from the APK
        // repairs existing containers after an app update without a full
        // reinstall; copyAssetFile keeps the executable bit.
        copyAssetFile("container/bin/g4f-go", new File(binDir, "g4f-go"));
    }

    /** g4f's --bind option takes one host:port value, not separate --port. */
    private void writeApiLauncher() throws IOException {
        File bin = new File(rootfs, "usr/local/bin");
        bin.mkdirs();
        File launcher = new File(bin, "g4f-api");
        // The API itself serves the included noVNC client at /desktop/ when
        // G4F_VNC_PORT points at the Xvnc display. This avoids a second HTTP
        // server (websockify) and keeps the Desktop tab on the API port.
        writeFile(launcher, "#!/bin/sh\n" +
                // A stale PYTHONHOME breaks CPython before its encodings package
                // loads; every launcher must clear it.
                "unset PYTHONHOME\n" +
                "export G4F_VNC_PORT=" + catalog.vncPort + "\n" +
                "export G4F_NOVNC_DIR=/usr/share/novnc\n" +
                "exec /opt/g4f-venv/bin/python -m g4f api --bind 127.0.0.1:" +
                catalog.apiPort + " --debug --no-reload\n");
        chmod(launcher, 0755);
    }

    /** Install the shared Xvnc/XFCE/D-Bus launcher from the APK asset. */
    private void writeDesktopLauncher() throws IOException {
        File bin = new File(rootfs, "usr/local/bin");
        bin.mkdirs();
        // Keep this in sync with provision-desktop.sh by shipping one script;
        // writeLaunchers() also runs after desktop overlays and on reconnect.
        copyAssetFile("container/scripts/g4f-desktop", new File(bin, "g4f-desktop"));
    }

    /** A login bash resets PATH from /etc/profile, so re-add /g4f-bin there. */
    private void writeShellProfile() throws IOException {
        File profileDir = new File(rootfs, "etc/profile.d");
        profileDir.mkdirs();
        writeFile(new File(profileDir, "g4f.sh"),
                "export G4F_CONTAINER=1\n" +
                // Shell sessions inherit this profile; clear any Python root
                // left by a host/tool launcher before entering the venv.
                "unset PYTHONHOME\n" +
                "export VIRTUAL_ENV=/opt/g4f-venv\n" +
                "export G4F_PYTHON=/opt/g4f-venv/bin/python\n" +
                "export PATH=/g4f-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n");
    }

    private static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            sb.append(String.format("%02x", value & 0xff));
        }
        return sb.toString();
    }

    private void extractDeb(File deb, File destination) throws IOException {
        try (InputStream fileIn = new FileInputStream(deb);
             ArArchiveInputStream ar = new ArArchiveInputStream(fileIn)) {
            ArArchiveEntry entry;
            while ((entry = ar.getNextArEntry()) != null) {
                String name = entry.getName();
                if (!name.startsWith("data.tar")) {
                    continue;
                }
                InputStream tarInput = ar;
                InputStream compressor = null;
                try {
                    if (name.endsWith(".xz")) {
                        compressor = new XZCompressorInputStream(ar);
                    } else if (name.endsWith(".gz")) {
                        compressor = new GzipCompressorInputStream(ar);
                    }
                    tarInput = compressor != null ? compressor : ar;
                    extractTar(tarInput, destination, TERMUX_PREFIX);
                } finally {
                    if (compressor != null) compressor.close();
                }
                return;
            }
        }
        throw new IOException("No data.tar member in " + deb.getName());
    }

    private void extractTarGz(File archive, File destination, String stripPrefix) throws IOException {
        try (InputStream fileIn = new FileInputStream(archive);
             GzipCompressorInputStream gzip = new GzipCompressorInputStream(fileIn)) {
            extractTar(gzip, destination, stripPrefix);
        }
    }

    private void extractTar(InputStream in, File destination, String stripPrefix) throws IOException {
        try (TarArchiveInputStream tar = new TarArchiveInputStream(in)) {
            TarArchiveEntry entry;
            while ((entry = tar.getNextTarEntry()) != null) {
                String name = normalizeEntryName(entry.getName(), stripPrefix);
                if (name == null || name.isEmpty()) {
                    // Root entries such as "./" are directory markers only.
                    continue;
                }
                if (!isSafeRelativePath(name)) {
                    // A single odd entry in a Debian package must not abort the
                    // whole install. Keep the container safe by skipping it and
                    // make the skip visible in the install progress/log.
                    Log.w("g4f-container", "Skipping unsafe tar entry: " + entry.getName());
                    continue;
                }
                File target = new File(destination, name);
                if (entry.isDirectory()) {
                    // A stale symlink here must not redirect mkdirs() outside the rootfs.
                    target.delete();
                    target.mkdirs();
                    chmod(target, entry.getMode());
                } else if (entry.isSymbolicLink()) {
                    // Debian images use absolute links like /bin/more. On Android
                    // those would escape the rootfs, so store them as equivalent
                    // relative links inside the extracted tree.
                    String link = relativeLink(name, entry.getLinkName());
                    if (link == null) continue;
                    target.getParentFile().mkdirs();
                    target.delete();
                    try {
                        Os.symlink(link, target.getAbsolutePath());
                    } catch (Exception ignored) {
                        // Some Android filesystems do not allow symlinks; skip.
                    }
                } else if (entry.isLink()) {
                    String linkName = normalizeEntryName(entry.getLinkName(), stripPrefix);
                    if (!isSafeRelativePath(linkName)) continue;
                    target.getParentFile().mkdirs();
                    target.delete();
                    File linkTarget = new File(destination, linkName);
                    try {
                        Os.link(linkTarget.getAbsolutePath(), target.getAbsolutePath());
                    } catch (Exception ignored) {
                        copyFile(linkTarget, target);
                    }
                } else if (entry.isFile()) {
                    target.getParentFile().mkdirs();
                    // Overwrite stale symlinks instead of writing through them.
                    target.delete();
                    try (OutputStream out = new FileOutputStream(target)) {
                        byte[] buffer = new byte[32 * 1024];
                        int count;
                        while ((count = tar.read(buffer)) != -1) {
                            out.write(buffer, 0, count);
                        }
                    }
                    chmod(target, entry.getMode());
                }
            }
        }
    }

    private static String normalizeEntryName(String raw, String stripPrefix) {
        if (raw == null) return null;
        String name = raw;
        while (name.startsWith("./")) name = name.substring(2);
        if (!stripPrefix.isEmpty() && name.startsWith(stripPrefix)) {
            name = name.substring(stripPrefix.length());
        }
        return name;
    }

    private static boolean isSafeRelativePath(String name) {
        // Android uses '/' as the only path separator. Debian packages contain
        // legitimate escaped names such as system-systemd\x2dcryptsetup.slice,
        // so a backslash is data here, not a traversal. Reject absolute paths
        // and '..' segments instead.
        if (name == null || name.isEmpty() || name.startsWith("/")) {
            return false;
        }
        for (String part : name.split("/")) {
            if (part.equals("..")) return false;
        }
        return true;
    }

    /** Split a path into safe segments, resolving "." and ".." lexically. */
    private static List<String> pathSegments(String path) {
        List<String> segments = new ArrayList<>();
        for (String part : path.split("/")) {
            if (part.isEmpty() || part.equals(".")) continue;
            if (part.equals("..")) {
                if (segments.isEmpty()) return null;
                segments.remove(segments.size() - 1);
            } else {
                segments.add(part);
            }
        }
        return segments;
    }

    /** Make a symlink target relative to the link's directory, inside the rootfs. */
    private static String relativeLink(String name, String linkTarget) {
        List<String> linkPath = pathSegments(name);
        if (linkPath == null || linkPath.isEmpty() || linkTarget == null) {
            return null;
        }
        List<String> linkDir = new ArrayList<>(linkPath.subList(0, linkPath.size() - 1));
        List<String> target;
        if (linkTarget.startsWith("/")) {
            target = pathSegments(linkTarget);
        } else {
            List<String> relative = pathSegments(linkTarget);
            if (relative == null) return null;
            target = new ArrayList<>(linkDir);
            target.addAll(relative);
        }
        if (target == null || target.isEmpty()) return null;
        int common = 0;
        while (common < linkDir.size() && common < target.size()
                && linkDir.get(common).equals(target.get(common))) {
            common++;
        }
        List<String> result = new ArrayList<>();
        for (int i = common; i < linkDir.size(); i++) result.add("..");
        result.addAll(target.subList(common, target.size()));
        return String.join("/", result);
    }

    private static void chmod(File file, int mode) {
        try {
            Os.chmod(file.getAbsolutePath(), mode & 0777);
        } catch (Exception ignored) {
            // Fall back to java.io permissions on filesystems without chmod.
            if (file.isFile()) file.setExecutable((mode & 0111) != 0, false);
        }
    }

    private static void copyFile(File from, File to) throws IOException {
        if (!from.exists()) return;
        try (InputStream in = new FileInputStream(from);
             OutputStream out = new FileOutputStream(to)) {
            byte[] buffer = new byte[32 * 1024];
            int count;
            while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        }
    }

    /** Copy one APK asset file into the rootfs and make it executable. */
    private void copyAssetFile(String assetPath, File target) throws IOException {
        target.getParentFile().mkdirs();
        try (InputStream in = context.getAssets().open(assetPath);
             OutputStream out = new FileOutputStream(target)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
        }
        chmod(target, 0755);
    }

    /** Copy the contents of an APK asset directory into a filesystem directory. */
    private void copyAssetTree(String assetPath, File targetDir) throws IOException {
        targetDir.mkdirs();
        String[] children = context.getAssets().list(assetPath);
        if (children == null) {
            return;
        }
        for (String child : children) {
            String childPath = assetPath + "/" + child;
            String[] grandchildren = context.getAssets().list(childPath);
            if (grandchildren == null || grandchildren.length == 0) {
                File target = new File(targetDir, child);
                target.getParentFile().mkdirs();
                try (InputStream in = context.getAssets().open(childPath);
                     OutputStream out = new FileOutputStream(target)) {
                    byte[] buffer = new byte[8192];
                    int count;
                    while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                }
                chmod(target, 0755);
            } else {
                copyAssetTree(childPath, new File(targetDir, child));
            }
        }
    }

    private static void writeFile(File file, String content) throws IOException {
        file.getParentFile().mkdirs();
        // Rootfs images may ship /etc/resolv.conf as a dangling symlink.
        file.delete();
        try (OutputStream out = new FileOutputStream(file)) {
            out.write(content.getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) deleteRecursively(child);
        }
        file.delete();
    }
}
