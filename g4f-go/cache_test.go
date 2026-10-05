package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestClearCache(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("G4F_CACHE_DIR", dir)

	// Populate the cache with nested content.
	sub := filepath.Join(dir, ".models", "openai")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	file := filepath.Join(sub, "models.json")
	if err := os.WriteFile(file, []byte("[]"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	if code := clearCache(); code != 0 {
		t.Fatalf("clearCache() = %d, want 0", code)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("cache not empty: %d entries left", len(entries))
	}
}

func TestClearCacheMissingDir(t *testing.T) {
	t.Setenv("G4F_CACHE_DIR", filepath.Join(t.TempDir(), "does-not-exist"))

	if code := clearCache(); code != 0 {
		t.Fatalf("clearCache() = %d, want 0 for missing dir", code)
	}
}

func TestPythonCacheDirEnvOverride(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("G4F_CACHE_DIR", dir)

	if got := pythonCacheDir(); got != dir {
		t.Fatalf("pythonCacheDir() = %q, want %q", got, dir)
	}
}
