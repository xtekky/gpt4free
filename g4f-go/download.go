package main

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
)

// RuntimeManifest mirrors runtime.json: one download source per platform.
type RuntimeManifest struct {
	Version   int                    `json:"version"`
	Python    string                 `json:"python"`
	PBSTag    string                 `json:"pbs_tag"`
	Platforms map[string]RuntimeSpec `json:"platforms"`
}

// RuntimeSpec is a single platform's runtime download.
type RuntimeSpec struct {
	Kind   string `json:"kind"` // "pbs" (install_only tarball)
	Arch   string `json:"arch"`
	URL    string `json:"url"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// runtimeManifestKey returns the runtime.json entry used for this host:
// "linux-x64", "linux-arm64", "windows-amd64" or "darwin-*".
// Termux/Android uses the system python package and never reads this manifest.
func runtimeManifestKey() string {
	arch := goArchToken()
	switch runtime.GOOS {
	case "linux":
		if arch == "arm64" {
			return "linux-arm64"
		}
		return "linux-x64"
	case "windows":
		return "windows-amd64"
	case "darwin":
		if arch == "arm64" {
			return "darwin-arm64"
		}
		return "darwin-x64"
	}
	return runtime.GOOS + "-" + arch
}

// runtimeSpecForHost picks the manifest entry for this OS/arch.
func runtimeSpecForHost(m *RuntimeManifest) (*RuntimeSpec, error) {
	key := runtimeManifestKey()
	spec, ok := m.Platforms[key]
	if !ok {
		return nil, fmt.Errorf("no runtime in manifest for platform %q", key)
	}
	return &spec, nil
}

// partName is the temp file used while a download is in flight.
func partName(binDir string) string {
	return filepath.Join(binDir, ".g4f-runtime", "runtime.download")
}

// installedOkName marks a fully verified runtime extraction.
func installedOkName(binDir string) string {
	return filepath.Join(binDir, ".g4f-runtime", ".runtime-ok")
}

const (
	// downloadAttempts bounds retries of an interrupted runtime download.
	// Mobile networks drop long GitHub transfers often; each retry resumes
	// from the bytes already on disk instead of starting over.
	downloadAttempts = 3
	// downloadStallTimeout aborts a transfer that stops making progress.
	// Without it a half-open connection on a phone would hang provisioning
	// forever with no error to report.
	downloadStallTimeout = 90 * time.Second
)

// downloadRetryDelay is a variable so tests do not sleep.
var downloadRetryDelay = 2 * time.Second

// downloadHTTPClient uses bounded dial/TLS/header timeouts but no overall
// deadline: a 60 MB runtime transfer is expected to take minutes. Stalled
// reads are caught by stallGuard instead.
func downloadHTTPClient() *http.Client {
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   20 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
		ExpectContinueTimeout: 5 * time.Second,
	}
	return &http.Client{Transport: transport}
}

// stallGuard closes the response body when no bytes arrive for timeout.
type stallGuard struct {
	body    io.ReadCloser
	timeout time.Duration
	timer   *time.Timer
	stalled atomic.Bool
}

func newStallGuard(body io.ReadCloser, timeout time.Duration) *stallGuard {
	g := &stallGuard{body: body, timeout: timeout}
	g.timer = time.AfterFunc(timeout, func() {
		g.stalled.Store(true)
		body.Close()
	})
	return g
}

func (g *stallGuard) Read(p []byte) (int, error) {
	n, err := g.body.Read(p)
	if n > 0 {
		g.timer.Reset(g.timeout)
	}
	if err != nil && g.stalled.Load() {
		return n, fmt.Errorf("no data for %s", g.timeout)
	}
	return n, err
}

func (g *stallGuard) Stop() { g.timer.Stop() }

// downloadRuntime fetches the manifest URL for this host into cachePath
// (verifying size + sha256 when pinned) and reports live progress.
func downloadRuntime(binDir, cachePath string, spec *RuntimeSpec) error {
	if err := os.MkdirAll(filepath.Dir(partName(binDir)), 0o755); err != nil {
		return err
	}

	// Cache is valid when file exists with the pinned size (or any size when
	// the manifest has no pinned size/sha).
	if fi, err := os.Stat(cachePath); err == nil && fi.Size() > 0 {
		if spec.Size <= 0 || fi.Size() == spec.Size {
			fmt.Printf("runtime: using cached download (%s)\n", humanBytes(fi.Size()))
			return verifyRuntime(cachePath, spec)
		}
		fmt.Printf("runtime: cached download incomplete, re-downloading\n")
	}

	fmt.Printf("runtime: downloading CPython %s (%s)\n", PythonVer, humanBytes(spec.Size))
	fmt.Printf("  %s\n", spec.URL)
	start := time.Now()

	var lastErr error
	for attempt := 1; attempt <= downloadAttempts; attempt++ {
		if attempt > 1 {
			fmt.Printf("runtime: download interrupted (%v); retrying %d/%d\n",
				lastErr, attempt, downloadAttempts)
			time.Sleep(downloadRetryDelay)
		}
		lastErr = fetchRuntime(binDir, spec)
		if lastErr == nil {
			break
		}
	}
	if lastErr != nil {
		return fmt.Errorf("download failed after %d attempts: %w", downloadAttempts, lastErr)
	}

	if err := os.Rename(partName(binDir), cachePath); err != nil {
		return err
	}
	fi, err := os.Stat(cachePath)
	if err != nil {
		return err
	}
	if spec.Size > 0 && fi.Size() != spec.Size {
		return fmt.Errorf("size mismatch: got %d, expected %d (update runtime.json)", fi.Size(), spec.Size)
	}
	fmt.Printf("runtime: downloaded %s in %s\n", humanBytes(fi.Size()), time.Since(start).Round(time.Second))
	return verifyRuntime(cachePath, spec)
}

// fetchRuntime performs one transfer attempt and appends to the partial file
// so a retry (or a later process) resumes where the previous one stopped.
func fetchRuntime(binDir string, spec *RuntimeSpec) error {
	part := partName(binDir)
	offset := int64(0)
	if fi, err := os.Stat(part); err == nil {
		offset = fi.Size()
	}
	if spec.Size > 0 && offset > spec.Size {
		if err := os.Truncate(part, 0); err != nil {
			return err
		}
		offset = 0
	}
	if spec.Size > 0 && offset == spec.Size {
		return nil
	}
	if offset > 0 {
		fmt.Printf("runtime: resuming download at %s\n", humanBytes(offset))
	}

	req, err := http.NewRequest(http.MethodGet, spec.URL, nil)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := downloadHTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY | os.O_APPEND
	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored Range (or we started at 0): rewrite from scratch.
		offset = 0
		flags = os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	case http.StatusPartialContent:
	case http.StatusRequestedRangeNotSatisfiable:
		os.Remove(part)
		return fmt.Errorf("HTTP %s for range %d-", resp.Status, offset)
	default:
		return fmt.Errorf("HTTP %s", resp.Status)
	}

	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	total := spec.Size
	if total <= 0 {
		total = offset + resp.ContentLength
	}
	guard := newStallGuard(resp.Body, downloadStallTimeout)
	defer guard.Stop()
	_, copyErr := copyWithProgressFrom(out, guard, offset, total, time.Now())
	if cerr := out.Close(); copyErr == nil {
		copyErr = cerr
	}
	if copyErr != nil {
		// Keep the partial file: the next attempt resumes from here.
		return fmt.Errorf("download interrupted: %w", copyErr)
	}
	return nil
}

// verifyRuntime validates sha256 when pinned in the manifest.
func verifyRuntime(cachePath string, spec *RuntimeSpec) error {
	if spec.SHA256 == "" {
		return nil // unpinned; trust size/transport
	}
	f, err := os.Open(cachePath)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, spec.SHA256) {
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, spec.SHA256)
	}
	fmt.Println("runtime: sha256 verified")
	return nil
}

// copyWithProgress streams r into w while printing a \r-updated progress bar.
func copyWithProgress(w io.Writer, r io.Reader, total int64, start time.Time) (int64, error) {
	return copyWithProgressFrom(w, r, 0, total, start)
}

// copyWithProgressFrom is copyWithProgress for a resumed transfer: initial is
// the number of bytes already on disk, so the bar starts at the real offset.
func copyWithProgressFrom(w io.Writer, r io.Reader, initial, total int64, start time.Time) (int64, error) {
	buf := make([]byte, 256*1024)
	written := initial
	lastPrint := time.Time{}
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return written, werr
			}
			written += int64(n)
			// Throttle progress output to ~5 updates/sec.
			if time.Since(lastPrint) > 200*time.Millisecond {
				lastPrint = time.Now()
				progressLine(written, total, start)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return written, err
		}
	}
	// Final line: clear the \r-update with a newline.
	progressLine(written, total, start)
	return written, nil
}

// progressLine prints "pct | done/total | rate | eta" on one line (carriage
// return, no newline) so it reads like a live progress bar.
func progressLine(written, total int64, start time.Time) {
	if total > 0 {
		pct := float64(written) / float64(total) * 100
		eta := time.Duration(float64(time.Since(start)) / (float64(written) / float64(total)) * (1 - float64(written)/float64(total)))
		fmt.Printf("\r  %5.1f%%  %s / %s  %s/s  eta %s",
			pct, humanBytes(written), humanBytes(total), humanBytes(int64(float64(written)/time.Since(start).Seconds())), eta.Round(time.Second))
	} else {
		fmt.Printf("\r  %s downloaded", humanBytes(written))
	}
}

// humanBytes renders byte counts readably (KiB/MiB/GiB).
func humanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

// extractRuntime unpacks the downloaded archive into binDir/python-home/:
//   - *.tar.gz (pbs, android) via gzip+tar with a single top dir stripped
//   - *.zip (windows embed, legacy) via archive/zip
//
// The archive's single top-level directory (e.g. "python/", "prefix/") is
// stripped and the contents land in python-home/ (bin/..., lib/...),
// matching the launcher/lookup layout.
func extractRuntime(binDir, cachePath string) error {
	dest := filepath.Join(binDir, "python-home")
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	f, err := os.Open(cachePath)
	if err != nil {
		return err
	}
	defer f.Close()

	if strings.HasSuffix(strings.ToLower(cachePath), ".zip") {
		fi, err := f.Stat()
		if err != nil {
			return err
		}
		return extractZip(f, fi.Size(), dest)
	}

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)

	// Determine the single top-level directory.
	var top string
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		clean := path.Clean(hdr.Name)
		parts := strings.Split(clean, "/")
		if len(parts) > 0 && parts[0] != "." && parts[0] != "" && top == "" {
			top = parts[0]
		}
	}

	// Second pass: extract everything with zip-slip protection.
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	gz, err = gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr = tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if strings.Contains(hdr.Name, "..") {
			return fmt.Errorf("unsafe path in archive: %s", hdr.Name)
		}
		rel := strings.TrimPrefix(hdr.Name, "./")
		if rel == top {
			rel = ""
		} else if top != "" {
			rel = strings.TrimPrefix(rel, top+"/")
		}
		rel = strings.TrimPrefix(rel, "/")
		rel = strings.TrimPrefix(rel, "./")
		name := filepath.Clean(rel)
		if name == "." || name == "" {
			continue
		}
		if name == ".." || strings.HasPrefix(name, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe path in archive: %s", hdr.Name)
		}
		target := filepath.Join(dest, name)
		if !strings.HasPrefix(target, filepath.Clean(dest)+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe path in archive: %s", hdr.Name)
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(hdr.Mode)&0o777)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			if err := out.Close(); err != nil {
				return err
			}
			if err := os.Chmod(target, os.FileMode(hdr.Mode)&0o777); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// Ensure symlink and its target do not escape the destination directory
			if !isSafeSymlink(dest, target, hdr.Linkname) {
				continue
			}
			// Symlinks in pbs installs point within the tree; recreate them
			// (libpython3.so -> libpython3.15.so etc).
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			_ = os.Remove(target)
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				// Some filesystems disallow symlinks; fall back to a copy.
				_ = os.Remove(target)
				if copySymlinkTarget(target, hdr.Linkname) != nil {
					// Best effort: log-free skip keeps extraction robust.
					_ = err
				}
			}
		default:
			// Hard links, devices, etc: skip silently (not needed).
		}
	}
	return nil
}

// isSafeSymlink validates that candidate symlink target does not escape the destination directory
func isSafeSymlink(dest, target, linkname string) bool {
	if filepath.IsAbs(linkname) {
		return false
	}
	realDest, err := filepath.EvalSymlinks(filepath.Clean(dest))
	if err != nil {
		realDest = filepath.Clean(dest)
	}
	realTargetDir, err := filepath.EvalSymlinks(filepath.Dir(target))
	if err != nil {
		return false
	}
	targetRel, err := filepath.Rel(realDest, realTargetDir)
	if err != nil || strings.HasPrefix(filepath.Clean(targetRel), "..") {
		return false
	}
	resolvedLink := filepath.Join(realTargetDir, linkname)
	realLink, err := filepath.EvalSymlinks(resolvedLink)
	if err != nil {
		realLink = filepath.Clean(resolvedLink)
	}
	linkRel, err := filepath.Rel(realDest, realLink)
	if err != nil || strings.HasPrefix(filepath.Clean(linkRel), "..") {
		return false
	}
	return true
}

// copySymlinkTarget attempts to copy a symlink's target for filesystems that
// reject symlinks (best effort).
func copySymlinkTarget(target, linkname string) error {
	src := filepath.Join(filepath.Dir(target), linkname)
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(target, data, 0o755)
}

const defaultTermuxPrefix = "/data/data/com.termux/files/usr"

// termuxPrefix returns the Termux installation prefix. PREFIX is set by
// Termux; the fallback keeps Android builds usable when launched from a
// minimal shell wrapper.
func termuxPrefix() string {
	if prefix := strings.TrimSpace(os.Getenv("PREFIX")); prefix != "" {
		return prefix
	}
	return defaultTermuxPrefix
}

// IsTermuxSystem reports whether this launcher is running in Termux. Android
// builds are shipped for Termux, so GOOS alone is authoritative there; on
// other hosts we require explicit Termux environment markers so a normal
// desktop Python runtime is never mistaken for Termux.
func IsTermuxSystem() bool {
	if runtime.GOOS == "android" || os.Getenv("TERMUX_VERSION") != "" {
		return true
	}
	prefix := strings.TrimSpace(os.Getenv("PREFIX"))
	return prefix != "" && strings.Contains(prefix, "com.termux")
}

// findTermuxPython returns the executable installed by Termux's python
// package. Paths under $PREFIX are preferred over PATH lookups so a host
// python cannot be picked up by mistake.
func findTermuxPython(prefix string) (string, error) {
	for _, name := range []string{"python3", "python"} {
		exe := filepath.Join(prefix, "bin", name)
		info, err := os.Stat(exe)
		if err == nil && !info.IsDir() && info.Mode().Perm()&0o111 != 0 {
			return exe, nil
		}
	}
	return "", fmt.Errorf("python not found in Termux prefix %s (run 'pkg install python')", prefix)
}

// ensureTermuxRuntime uses the interpreter from Termux's python package. No
// CPython download, dlopen runner or build toolchain is needed.
func ensureTermuxRuntime() (string, error) {
	prefix := termuxPrefix()
	if py, err := findTermuxPython(prefix); err == nil {
		return py, nil
	}

	fmt.Println("Installing Termux packages: python")
	pkg := filepath.Join(prefix, "bin", "pkg")
	if _, err := os.Stat(pkg); err != nil {
		var lookErr error
		pkg, lookErr = exec.LookPath("pkg")
		if lookErr != nil {
			return "", fmt.Errorf("pkg not found in Termux prefix %s: %w", prefix, lookErr)
		}
	}
	cmd := exec.Command(pkg, "install", "-y", "python")
	cmd.Env = append(os.Environ(), "PREFIX="+prefix)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pkg install python failed: %w", err)
	}
	return findTermuxPython(prefix)
}

// pythonFromEnvironment lets a launcher point g4f-go at an interpreter that is
// already installed (Termux python or a host python). It must never trigger a
// download: that is the only way to keep
// g4f-go working on machines where HTTPS downloads fail (e.g. missing CA
// certificates under PRoot).
func pythonFromEnvironment() (string, error) {
	value := strings.TrimSpace(os.Getenv("G4F_PYTHON"))
	if value == "" {
		return "", nil
	}
	exe := value
	if !filepath.IsAbs(exe) {
		found, err := exec.LookPath(exe)
		if err != nil {
			return "", fmt.Errorf("G4F_PYTHON %q not found in PATH: %w", value, err)
		}
		exe = found
	}
	info, err := os.Stat(exe)
	if err != nil {
		return "", fmt.Errorf("G4F_PYTHON %q is not accessible: %w", value, err)
	}
	if info.IsDir() || info.Mode().Perm()&0o111 == 0 {
		return "", fmt.Errorf("G4F_PYTHON %q is not an executable file", value)
	}
	return exe, nil
}

// ensureRuntime is the top-level entry point. An explicit G4F_PYTHON always
// wins; Termux uses its system python package; every other platform downloads
// and extracts a pinned CPython runtime into binDir before returning the
// interpreter path.
func ensureRuntime() (string, error) {
	if py, err := pythonFromEnvironment(); err != nil || py != "" {
		return py, err
	}
	if IsTermuxSystem() {
		return ensureTermuxRuntime()
	}

	binDir := installDir()
	exe, err := pythonExecutable(binDir)
	if err == nil {
		if _, statErr := os.Stat(exe); statErr == nil {
			return exe, nil
		}
	}

	manifest, err := readRuntimeManifest()
	if err != nil {
		return "", err
	}
	spec, err := runtimeSpecForHost(manifest)
	if err != nil {
		return "", err
	}

	cachePath := filepath.Join(binDir, ".g4f-runtime", "runtime-"+filepath.Base(spec.URL))
	if err := downloadRuntime(binDir, cachePath, spec); err != nil {
		return "", err
	}

	return installDownloadedRuntime(binDir, cachePath)
}

func installDownloadedRuntime(binDir, cachePath string) (string, error) {
	// A missing interpreter invalidates older success stamps. Android may still
	// need to build its runner from an already extracted runtime.
	okStamp := installedOkName(binDir)
	_, stampErr := os.Stat(okStamp)
	extracted := stampErr != nil || runtime.GOOS != "android"
	if extracted {
		if err := os.Remove(okStamp); err != nil && !os.IsNotExist(err) {
			return "", err
		}
		fmt.Println("runtime: extracting (this can take a minute)...")
		start := time.Now()
		if err := extractRuntime(binDir, cachePath); err != nil {
			return "", fmt.Errorf("extract runtime: %w", err)
		}
		fmt.Printf("runtime: extracted in %s\n", time.Since(start).Round(time.Second))
	}

	exe, err := finalizeRuntime(binDir)
	if err != nil {
		return "", err
	}
	if extracted {
		if err := os.WriteFile(okStamp, []byte("ok"), 0o644); err != nil {
			return "", err
		}
	}
	return exe, nil
}

// Check if a command/binary exists in PATH
func commandExists(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

// finalizeRuntime does platform-specific finishing (launcher setup) and
// returns the interpreter path.
func finalizeRuntime(binDir string) (string, error) {
	// Android builds the dlopen C runner instead of a shell launcher.
	if runtime.GOOS == "android" {
		return finalizeAndroidRuntime(binDir)
	}
	// Write the unix launcher (windows uses python.exe from the archive).
	if err := writeLauncher(binDir); err != nil {
		return "", err
	}
	exe, err := pythonExecutable(binDir)
	if err != nil {
		if runtime.GOOS == "windows" {
			return "", fmt.Errorf("python runtime extracted but %s is missing; check whether antivirus quarantined it, then retry: %w", filepath.Join(binDir, "python-home", "python.exe"), err)
		}
		return "", err
	}
	if _, err := os.Stat(exe); err != nil {
		return "", fmt.Errorf("python runtime extracted but %s is missing", exe)
	}
	return exe, nil
}

// parseRuntimeManifest decodes a RuntimeManifest from bytes.
func parseRuntimeManifest(data []byte) (*RuntimeManifest, error) {
	var m RuntimeManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if len(m.Platforms) == 0 {
		return nil, fmt.Errorf("runtime.json: no platforms defined")
	}
	return &m, nil
}
