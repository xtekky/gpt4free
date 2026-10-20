package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestBrowserSpecForHost(t *testing.T) {
	spec, err := browserSpecForHost()
	if err != nil {
		t.Skipf("no browser build for %s/%s: %v", runtime.GOOS, runtime.GOARCH, err)
	}
	if spec.URL == "" || spec.SHA256 == "" || spec.Binary == "" {
		t.Fatalf("incomplete spec: %+v", spec)
	}
	if spec.Size <= 0 {
		t.Fatalf("spec has no size: %+v", spec)
	}
	if len(spec.SHA256) != 64 {
		t.Fatalf("sha256 is not 64 hex chars: %q", spec.SHA256)
	}
}

func TestHeadlessActive(t *testing.T) {
	t.Setenv("G4F_BROWSER_HEADLESS", "")
	os.Unsetenv("G4F_BROWSER_HEADLESS")

	// g4f defaults to headless.
	if !headlessActive([]string{"client", "hi"}) {
		t.Error("expected headless by default")
	}
	if headlessActive([]string{"client", "--no-headless"}) {
		t.Error("--no-headless should disable headless")
	}

	t.Setenv("G4F_BROWSER_HEADLESS", "false")
	if headlessActive([]string{"client"}) {
		t.Error("G4F_BROWSER_HEADLESS=false should disable headless")
	}
	t.Setenv("G4F_BROWSER_HEADLESS", "true")
	if !headlessActive([]string{"client", "--no-headless"}) {
		t.Error("explicit env should win over --no-headless")
	}
}

func TestIsTruthy(t *testing.T) {
	for _, v := range []string{"1", "true", "TRUE", " yes ", "on"} {
		if !isTruthy(v) {
			t.Errorf("isTruthy(%q) = false, want true", v)
		}
	}
	for _, v := range []string{"", "0", "false", "no", "off", "maybe"} {
		if isTruthy(v) {
			t.Errorf("isTruthy(%q) = true, want false", v)
		}
	}
}

func TestParseBrowserServeArgs(t *testing.T) {
	opts, err := parseBrowserServeArgs(nil)
	if err != nil {
		t.Fatalf("parseBrowserServeArgs(nil): %v", err)
	}
	if opts.Host != browserDefaultHost || opts.Port != browserDefaultPort {
		t.Fatalf("unexpected defaults: %+v", opts)
	}

	opts, err = parseBrowserServeArgs([]string{"--host", "0.0.0.0", "--port", "1234", "--log-level", "info"})
	if err != nil {
		t.Fatalf("parseBrowserServeArgs: %v", err)
	}
	if opts.Host != "0.0.0.0" || opts.Port != 1234 {
		t.Fatalf("unexpected parse: %+v", opts)
	}
	if len(opts.Args) != 2 || opts.Args[0] != "--log-level" || opts.Args[1] != "info" {
		t.Fatalf("unexpected forwarded args: %v", opts.Args)
	}

	opts, err = parseBrowserServeArgs([]string{"--host=127.0.0.1", "--port=0"})
	if err != nil {
		t.Fatalf("parseBrowserServeArgs(=): %v", err)
	}
	if opts.Host != "127.0.0.1" || opts.Port != 0 {
		t.Fatalf("unexpected parse: %+v", opts)
	}

	if _, err := parseBrowserServeArgs([]string{"--port", "abc"}); err == nil {
		t.Error("expected error for non-numeric port")
	}
	if _, err := parseBrowserServeArgs([]string{"--host"}); err == nil {
		t.Error("expected error for missing value")
	}
}

func TestBrowserServeArgsUsesForkFingerprint(t *testing.T) {
	args := browserServeArgs("127.0.0.1", 9222)
	for _, a := range args {
		// The fork already sends a full Chrome UA; adding the suffix would
		// duplicate it, and --user-agent conflicts with the suffix flag.
		if a == "--user-agent" || a == "--user-agent-suffix" {
			t.Fatalf("browserServeArgs must not shape the UA (the fork already sends Chrome): %v", args)
		}
	}
	for _, want := range []string{"serve", "--block-private-networks", "--load-resources", "iframe", "--http-cache-dir"} {
		found := false
		for _, a := range args {
			if a == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("browserServeArgs is missing %q: %v", want, args)
		}
	}
}

func TestFreePort(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort(): %v", err)
	}
	if port <= 0 || port > 65535 {
		t.Fatalf("freePort() = %d, out of range", port)
	}
}

// TestExtractBrowserZip covers the Windows portable layout: a single top-level
// directory that must be stripped, and no unix permission bits in the archive.
func TestExtractBrowserZip(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "portable.zip")

	f, err := os.Create(archive)
	if err != nil {
		t.Fatalf("create archive: %v", err)
	}
	zw := zip.NewWriter(f)
	entries := map[string]string{
		"lightpanda-portable-windows-x64/lightpanda.exe":     "MZ fake exe",
		"lightpanda-portable-windows-x64/VCRUNTIME140.dll":   "dll",
		"lightpanda-portable-windows-x64/README-WINDOWS.txt": "readme",
	}
	for name, body := range entries {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("create entry %s: %v", name, err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatalf("write entry %s: %v", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close file: %v", err)
	}

	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatalf("mkdir out: %v", err)
	}
	if err := extractBrowserZip(archive, out, "lightpanda.exe"); err != nil {
		t.Fatalf("extractBrowserZip: %v", err)
	}

	// The top-level directory must be stripped.
	for name, want := range entries {
		rel := filepath.Base(name)
		got, err := os.ReadFile(filepath.Join(out, rel))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		if string(got) != want {
			t.Errorf("%s = %q, want %q", rel, got, want)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "lightpanda-portable-windows-x64")); !os.IsNotExist(err) {
		t.Error("top-level directory was not stripped")
	}

	// The binary must be executable even though the zip carries no modes.
	if runtime.GOOS != "windows" {
		fi, err := os.Stat(filepath.Join(out, "lightpanda.exe"))
		if err != nil {
			t.Fatalf("stat binary: %v", err)
		}
		if fi.Mode().Perm()&0o111 == 0 {
			t.Errorf("binary mode = %v, want executable", fi.Mode().Perm())
		}
	}
}

func TestExtractBrowserZipRejectsTraversal(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "evil.zip")

	f, err := os.Create(archive)
	if err != nil {
		t.Fatalf("create archive: %v", err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("top/../../escape.txt")
	if err != nil {
		t.Fatalf("create entry: %v", err)
	}
	if _, err := w.Write([]byte("nope")); err != nil {
		t.Fatalf("write entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	f.Close()

	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatalf("mkdir out: %v", err)
	}
	if err := extractBrowserZip(archive, out, "lightpanda.exe"); err == nil {
		t.Error("expected traversal to be rejected")
	}
}

func TestExtractBrowserZipFlatLayout(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "lightpanda.zip")

	f, err := os.Create(archive)
	if err != nil {
		t.Fatalf("create archive: %v", err)
	}
	zw := zip.NewWriter(f)
	w, err := zw.Create("lightpanda")
	if err != nil {
		t.Fatalf("create entry: %v", err)
	}
	if _, err := w.Write([]byte("ELF fake")); err != nil {
		t.Fatalf("write entry: %v", err)
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("close zip: %v", err)
	}
	f.Close()

	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatalf("mkdir out: %v", err)
	}
	if err := extractBrowserZip(archive, out, "lightpanda"); err != nil {
		t.Fatalf("extractBrowserZip: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(out, "lightpanda"))
	if err != nil {
		t.Fatalf("read binary: %v", err)
	}
	if string(got) != "ELF fake" {
		t.Errorf("binary = %q, want %q", got, "ELF fake")
	}
	if runtime.GOOS != "windows" {
		fi, err := os.Stat(filepath.Join(out, "lightpanda"))
		if err != nil {
			t.Fatalf("stat binary: %v", err)
		}
		if fi.Mode().Perm()&0o111 == 0 {
			t.Errorf("binary mode = %v, want executable", fi.Mode().Perm())
		}
	}
}

// TestPatchBrowserRunpath builds a minimal ELF64 with a DT_RUNPATH pointing at
// a build-machine path and checks that it is rewritten to $ORIGIN in place.
func TestPatchBrowserRunpath(t *testing.T) {
	const oldPath = "/home/builder/.lp-cache/prebuilt-v8/v0.5.9"
	image := buildTestELF(oldPath)

	dir := t.TempDir()
	path := filepath.Join(dir, "lightpanda")
	if err := os.WriteFile(path, image, 0o755); err != nil {
		t.Fatalf("write elf: %v", err)
	}

	off, size, err := elfRunpath(image)
	if err != nil {
		t.Fatalf("elfRunpath: %v", err)
	}
	if off < 0 {
		t.Fatal("elfRunpath found no RUNPATH")
	}
	if got := string(image[off : off+size-1]); got != oldPath {
		t.Fatalf("RUNPATH = %q, want %q", got, oldPath)
	}

	if err := patchBrowserRunpath(path); err != nil {
		t.Fatalf("patchBrowserRunpath: %v", err)
	}
	patched, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read patched: %v", err)
	}
	off, size, err = elfRunpath(patched)
	if err != nil {
		t.Fatalf("elfRunpath(patched): %v", err)
	}
	if got := string(patched[off : off+size-1]); got != "$ORIGIN" {
		t.Errorf("patched RUNPATH = %q, want %q", got, "$ORIGIN")
	}
	if len(patched) != len(image) {
		t.Errorf("patch changed file size: %d -> %d", len(image), len(patched))
	}

	// Patching twice must be a no-op.
	if err := patchBrowserRunpath(path); err != nil {
		t.Fatalf("second patchBrowserRunpath: %v", err)
	}
	again, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read twice-patched: %v", err)
	}
	if !bytes.Equal(again, patched) {
		t.Error("second patch modified the file")
	}
}

func TestElfRunpathRejectsNonELF(t *testing.T) {
	if _, _, err := elfRunpath([]byte("not an elf at all, but long enough to pass the length check....")); err == nil {
		t.Error("expected an error for a non-ELF file")
	}
}

// buildTestELF returns a little-endian ELF64 image with a .dynamic section
// holding a single DT_RUNPATH entry and a .dynstr section holding its value.
func buildTestELF(runpath string) []byte {
	const (
		shoff     = 0x100
		shnum     = 3
		shentsize = 64
		dynOff    = 0x200
		dynSize   = 32 // DT_RUNPATH + DT_NULL
		strOff    = 0x300
	)
	strSize := len(runpath) + 1
	img := make([]byte, strOff+strSize)

	copy(img, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0})
	le := binary.LittleEndian
	le.PutUint64(img[0x28:], shoff)
	le.PutUint16(img[0x3a:], shentsize)
	le.PutUint16(img[0x3c:], shnum)

	// Section 1: .dynamic (SHT_DYNAMIC), linked to section 2.
	le.PutUint32(img[shoff+1*shentsize+4:], 6)
	le.PutUint64(img[shoff+1*shentsize+0x18:], dynOff)
	le.PutUint64(img[shoff+1*shentsize+0x20:], dynSize)
	le.PutUint32(img[shoff+1*shentsize+0x28:], 2)

	// Section 2: .dynstr (SHT_STRTAB).
	le.PutUint32(img[shoff+2*shentsize+4:], 3)
	le.PutUint64(img[shoff+2*shentsize+0x18:], strOff)
	le.PutUint64(img[shoff+2*shentsize+0x20:], uint64(strSize))

	le.PutUint64(img[dynOff:], 29) // DT_RUNPATH
	le.PutUint64(img[dynOff+8:], 0)
	le.PutUint64(img[dynOff+16:], 0) // DT_NULL

	copy(img[strOff:], runpath)
	return img
}

func TestBrowserDirUnderConfigDir(t *testing.T) {
	if got, want := browserDir(), filepath.Join(configDir(), "browser"); got != want {
		t.Errorf("browserDir() = %q, want %q", got, want)
	}
	if got, want := browserStampPath(), filepath.Join(browserDir(), ".version"); got != want {
		t.Errorf("browserStampPath() = %q, want %q", got, want)
	}
}

func TestInstalledBrowserBinaryNotInstalled(t *testing.T) {
	spec, err := browserSpecForHost()
	if err != nil {
		t.Skipf("no browser build for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// A missing binary must produce an actionable error, not a panic.
	if _, err := installedBrowserBinary(); err != nil {
		if !strings.Contains(err.Error(), "browser install") {
			t.Errorf("error should mention the install command, got: %v", err)
		}
	}
	_ = spec
}
