package main

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
	Kind   string `json:"kind"` // "pbs" (install_only tarball) or "android"
	Arch   string `json:"arch"`
	URL    string `json:"url"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// runtimeManifestKey returns the runtime.json entry used for this host:
// "linux-x64", "linux-arm64", "windows-amd64", "darwin-*" or "android".
func runtimeManifestKey() string {
	if runtime.GOOS == "android" {
		return "android"
	}
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

// downloadRuntime fetches the manifest URL for this host into cachePath
// (verifying size + sha256 when pinned) and reports live progress to stderr.
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

	fmt.Printf("runtime: downloading CPython %s (%s)\n", "3.14.7", humanBytes(spec.Size))
	fmt.Printf("  %s\n", spec.URL)
	start := time.Now()

	out, err := os.Create(partName(binDir))
	if err != nil {
		return err
	}
	// Network http.Client with no default timeout: progress is what keeps the
	// user informed, not a hard cutoff.
	client := &http.Client{}
	resp, err := client.Get(spec.URL)
	if err != nil {
		out.Close()
		os.Remove(partName(binDir))
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		out.Close()
		os.Remove(partName(binDir))
		return fmt.Errorf("download failed: HTTP %s", resp.Status)
	}

	// Prefer Content-Length; fall back to manifest size.
	total := resp.ContentLength
	if total <= 0 {
		total = spec.Size
	}
	_, copyErr := copyWithProgress(out, resp.Body, total, start)
	if cerr := out.Close(); copyErr == nil {
		copyErr = cerr
	}
	if copyErr != nil {
		os.Remove(partName(binDir))
		return fmt.Errorf("download interrupted: %w", copyErr)
	}
	if err := os.Rename(partName(binDir), cachePath); err != nil {
		os.Remove(partName(binDir))
		return err
	}
	fmt.Printf("runtime: downloaded %s in %s\n", humanBytes(total), time.Since(start).Round(time.Second))

	if spec.Size > 0 {
		fi, err := os.Stat(cachePath)
		if err != nil {
			return err
		}
		if fi.Size() != spec.Size {
			return fmt.Errorf("size mismatch: got %d, expected %d (update runtime.json)", fi.Size(), spec.Size)
		}
	}
	return verifyRuntime(cachePath, spec)
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
	buf := make([]byte, 256*1024)
	var written int64
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
		clean := filepath.Clean(hdr.Name)
		parts := strings.Split(clean, string(os.PathSeparator))
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
		rel := hdr.Name
		if top != "" {
			rel = strings.TrimPrefix(rel, top+"/")
			rel = strings.TrimPrefix(rel, top)
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
			// Symlinks in pbs installs point within the tree; recreate them
			// (libpython3.so -> libpython3.14.so etc).
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

func IsTermuxSystem() bool {
	// Check for Termux specific environment variable or directory
	if os.Getenv("TERMUX_VERSION") != "" {
		return true
	}
	_, err := os.Stat("/data/data/com.termux/files/usr")
	return !os.IsNotExist(err)
}

// ensureTermuxRuntime installs python and build dependencies via Termux's
// package manager and returns the system python path. No CPython download is
// needed — Termux provides a native python package.
func ensureTermuxRuntime() (string, error) {
	// Packages required for g4f and native extensions.
	required := []string{"python", "clang", "make", "libxml2", "libxslt", "libjpeg-turbo", "libpng"}
	var missing []string
	for _, pkg := range required {
		if !termuxPackageInstalled(pkg) {
			missing = append(missing, pkg)
		}
	}
	if len(missing) > 0 {
		fmt.Printf("Installing Termux packages: %s\n", strings.Join(missing, ", "))
		args := append([]string{"install", "-y"}, missing...)
		cmd := exec.Command("pkg", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("pkg install failed: %w", err)
		}
	}

	py, err := exec.LookPath("python3")
	if err != nil {
		py, err = exec.LookPath("python")
	}
	if err != nil {
		return "", fmt.Errorf("python not found in Termux PATH (run 'pkg install python')")
	}
	return py, nil
}

// termuxPackageInstalled checks if a Termux package is already installed
// using dpkg.
func termuxPackageInstalled(pkg string) bool {
	cmd := exec.Command("dpkg", "-s", pkg)
	return cmd.Run() == nil
}

// ensureRuntime is the top-level entry point. It downloads (if needed) and
// extracts the platform runtime into binDir, then returns the python
// executable/launcher path.
func ensureRuntime() (string, error) {
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

	// Extract unless already done (stamp written after successful extract).
	okStamp := installedOkName(binDir)
	if _, err := os.Stat(okStamp); err != nil {
		fmt.Println("runtime: extracting (this can take a minute)...")
		start := time.Now()
		if err := extractRuntime(binDir, cachePath); err != nil {
			return "", fmt.Errorf("extract runtime: %w", err)
		}
		if err := os.WriteFile(okStamp, []byte("ok"), 0o644); err != nil {
			return "", err
		}
		fmt.Printf("runtime: extracted in %s\n", time.Since(start).Round(time.Second))
	}

	return finalizeRuntime(binDir)
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
