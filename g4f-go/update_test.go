package main

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCompareVersions(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"0.1.0", "0.1.0", 0},
		{"v0.1.0", "0.1.0", 0},
		{"0.1.0", "0.2.0", -1},
		{"0.2.0", "0.1.9", 1},
		{"1.0", "1.0.0", 0},
		{"1.0.0", "1.0.1", -1},
		{"0.2.0-rc1", "0.2.0", 0},
	}
	for _, tc := range cases {
		if got := compareVersions(tc.a, tc.b); got != tc.want {
			t.Errorf("compareVersions(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestReleaseAssetName(t *testing.T) {
	got := releaseAssetName("v0.2.0", "linux", "amd64")
	if want := "g4f-go-0.2.0-linux-amd64.zip"; got != want {
		t.Fatalf("releaseAssetName() = %q, want %q", got, want)
	}
}

func TestFindReleaseAsset(t *testing.T) {
	assets := []releaseAsset{
		{Name: "g4f-go-0.2.0-windows-amd64.zip", BrowserDownloadURL: "windows"},
		{Name: "g4f-go-0.2.0-" + runtime.GOOS + "-" + runtime.GOARCH + ".zip", BrowserDownloadURL: "host"},
	}
	got, err := findReleaseAsset(assets, "v0.2.0")
	if err != nil {
		t.Fatalf("findReleaseAsset() error: %v", err)
	}
	if got.BrowserDownloadURL != "host" {
		t.Fatalf("findReleaseAsset() = %q, want host asset", got.BrowserDownloadURL)
	}

	if _, err := findReleaseAsset(assets[:1], "v0.2.0"); err == nil && runtime.GOOS != "windows" {
		t.Fatal("findReleaseAsset() expected an error for an unsupported platform")
	}
}

func TestExtractReleaseBinary(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "release.zip")
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	// Release zips contain a platform directory with the binary inside.
	w, err := zw.Create("linux-amd64/g4f-go")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("#!/bin/sh\necho hi\n")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(archivePath, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	bin, err := extractReleaseBinary(archivePath, "g4f-go", dir)
	if err != nil {
		t.Fatalf("extractReleaseBinary() error: %v", err)
	}
	defer os.Remove(bin)
	data, err := os.ReadFile(bin)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "#!/bin/sh\necho hi\n" {
		t.Fatalf("extracted contents = %q", string(data))
	}

	if _, err := extractReleaseBinary(archivePath, "missing", dir); err == nil {
		t.Fatal("extractReleaseBinary() expected an error for a missing binary")
	}
}

func TestUninstallTargetsDedup(t *testing.T) {
	oldConfig := os.Getenv("G4F_CONFIG_DIR")
	t.Setenv("G4F_CONFIG_DIR", t.TempDir())
	defer os.Setenv("G4F_CONFIG_DIR", oldConfig)

	dirs := uninstallTargets(installDir())
	if len(dirs) != 3 {
		t.Fatalf("uninstallTargets() returned %d dirs, want 3: %v", len(dirs), dirs)
	}
	seen := map[string]bool{}
	for _, dir := range dirs {
		if dir == "" {
			t.Fatal("uninstallTargets() returned an empty path")
		}
		if seen[dir] {
			t.Fatalf("uninstallTargets() returned duplicate path %q", dir)
		}
		seen[dir] = true
	}
}
