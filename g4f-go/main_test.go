package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureG4FPackage_NoLookupNeededForNonUpgradeWhenInstalled(t *testing.T) {
	if IsTermuxSystem() {
		t.Skip("termux runtime path uses system python flow")
	}

	binDir := t.TempDir()
	stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		t.Fatalf("mkdir stamp dir: %v", err)
	}
	if err := os.WriteFile(stamp, []byte("ok"), 0o644); err != nil {
		t.Fatalf("write stamp: %v", err)
	}

	if err := ensureG4FPackage(binDir, "/missing/python", []string{"client"}, ""); err != nil {
		t.Fatalf("ensureG4FPackage() returned unexpected error: %v", err)
	}
}

func TestEnsureG4FPackage_NoLookupNeededForMatchingPin(t *testing.T) {
	if IsTermuxSystem() {
		t.Skip("termux runtime path uses system python flow")
	}

	binDir := t.TempDir()
	stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		t.Fatalf("mkdir stamp dir: %v", err)
	}
	if err := os.WriteFile(stamp, []byte("ok"), 0o644); err != nil {
		t.Fatalf("write stamp: %v", err)
	}
	if err := writePinnedG4FVersion(binDir, "1.5.0"); err != nil {
		t.Fatalf("write pin stamp: %v", err)
	}

	// Installed and pinned to the same version: no pip lookup happens (the
	// python executable does not exist, so any install attempt would error).
	if err := ensureG4FPackage(binDir, "/missing/python", []string{"client"}, "1.5.0"); err != nil {
		t.Fatalf("ensureG4FPackage() returned unexpected error: %v", err)
	}

	// A different pin must trigger a (failing here, no interpreter) reinstall.
	if err := ensureG4FPackage(binDir, "/missing/python", []string{"client"}, "1.6.0"); err == nil {
		t.Fatal("ensureG4FPackage() with a changed pin expected an error (reinstall attempted), got nil")
	}
}
