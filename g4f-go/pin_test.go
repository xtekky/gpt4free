package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseVersionFlag(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		wantPin      string
		wantExplicit bool
		wantRest     []string
	}{
		{"pinned command", []string{"-v", "1.5.0", "client", "hi"}, "1.5.0", true, []string{"client", "hi"}},
		{"pin only", []string{"-v", "1.5.0"}, "1.5.0", true, []string{}},
		{"prerelease version", []string{"-v", "1.5.0rc1", "gui"}, "1.5.0rc1", true, []string{"gui"}},
		{"latest unpins", []string{"-v", "latest", "api"}, "", true, []string{"api"}},
		{"bare -v stays a version query", []string{"-v"}, "", false, []string{"-v"}},
		{"non-version is not a pin", []string{"-v", "client"}, "", false, []string{"-v", "client"}},
		{"no flag", []string{"client"}, "", false, []string{"client"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pin, explicit, rest := parseVersionFlag(tt.args)
			if pin != tt.wantPin || explicit != tt.wantExplicit {
				t.Fatalf("parseVersionFlag(%q) = (%q, %v), want (%q, %v)", tt.args, pin, explicit, tt.wantPin, tt.wantExplicit)
			}
			if len(rest) != len(tt.wantRest) {
				t.Fatalf("parseVersionFlag(%q) rest = %q, want %q", tt.args, rest, tt.wantRest)
			}
			for i := range rest {
				if rest[i] != tt.wantRest[i] {
					t.Fatalf("parseVersionFlag(%q) rest = %q, want %q", tt.args, rest, tt.wantRest)
				}
			}
		})
	}
}

func TestG4fPin_PersistsInstalledPinAndClearsOnLatest(t *testing.T) {
	binDir := t.TempDir()

	// No flag and no stored pin: unpinned.
	pin, rest := g4fPin(binDir, []string{"client", "hi"})
	if pin != "" || len(rest) != 2 {
		t.Fatalf("g4fPin() = (%q, %q), want (\"\", [client hi])", pin, rest)
	}

	// A flag selects the pin but is not persisted before the matching version
	// is installed — the stamp records what is installed, not what was asked.
	pin, rest = g4fPin(binDir, []string{"-v", "1.5.0", "client"})
	if pin != "1.5.0" || len(rest) != 1 || rest[0] != "client" {
		t.Fatalf("g4fPin() = (%q, %q), want (1.5.0, [client])", pin, rest)
	}
	if got := readPinnedG4FVersion(binDir); got != "" {
		t.Fatalf("persisted pin before install = %q, want empty", got)
	}

	// installG4F records the pin after a successful install.
	if err := writePinnedG4FVersion(binDir, pin); err != nil {
		t.Fatalf("write pin stamp: %v", err)
	}

	// Later runs without the flag keep the pin.
	pin, _ = g4fPin(binDir, []string{"client"})
	if pin != "1.5.0" {
		t.Fatalf("g4fPin() without flag = %q, want stored pin 1.5.0", pin)
	}

	// `-v latest` drops it again.
	pin, rest = g4fPin(binDir, []string{"-v", "latest", "api"})
	if pin != "" || len(rest) != 1 || rest[0] != "api" {
		t.Fatalf("g4fPin(-v latest) = (%q, %q), want (\"\", [api])", pin, rest)
	}
	if got := readPinnedG4FVersion(binDir); got != "" {
		t.Fatalf("persisted pin after -v latest = %q, want empty", got)
	}
}

func TestG4fPinChangeTriggersReinstall(t *testing.T) {
	binDir := t.TempDir()
	stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		t.Fatalf("mkdir stamp dir: %v", err)
	}
	if err := os.WriteFile(stamp, []byte("g4f installed"), 0o644); err != nil {
		t.Fatalf("write stamp: %v", err)
	}
	if err := writePinnedG4FVersion(binDir, "1.5.0"); err != nil {
		t.Fatalf("write pin stamp: %v", err)
	}

	// Pinning to a different version than the installed one must reinstall,
	// even though the flag itself is not persisted up front.
	pin, _ := g4fPin(binDir, []string{"-v", "1.6.0", "client"})
	if pin != "1.6.0" {
		t.Fatalf("g4fPin() = %q, want 1.6.0", pin)
	}
	if !needsInstall(binDir, pin) {
		t.Fatal("needsInstall() with changed pin = false, want true")
	}

	// Re-pin to the installed version: nothing to do.
	pin, _ = g4fPin(binDir, []string{"-v", "1.5.0", "client"})
	if needsInstall(binDir, pin) {
		t.Fatal("needsInstall() with matching pin = true, want false")
	}
}

func TestG4fSpec(t *testing.T) {
	if got := g4fSpec(""); got != "g4f[slim]" {
		t.Fatalf("g4fSpec(\"\") = %q, want g4f[slim]", got)
	}
	if got := g4fSpec("1.5.0"); got != "g4f[slim]==1.5.0" {
		t.Fatalf("g4fSpec(1.5.0) = %q, want g4f[slim]==1.5.0", got)
	}
}

func TestNeedsInstall(t *testing.T) {
	binDir := t.TempDir()
	stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")

	// Not installed yet: always needs an install, pinned or not.
	if !needsInstall(binDir, "") {
		t.Fatal("needsInstall() unpinned on fresh dir = false, want true")
	}
	if !needsInstall(binDir, "1.5.0") {
		t.Fatal("needsInstall() pinned on fresh dir = false, want true")
	}

	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		t.Fatalf("mkdir stamp dir: %v", err)
	}
	if err := os.WriteFile(stamp, []byte("g4f installed"), 0o644); err != nil {
		t.Fatalf("write stamp: %v", err)
	}

	// Installed, unpinned: nothing to do.
	if needsInstall(binDir, "") {
		t.Fatal("needsInstall() unpinned on installed dir = true, want false")
	}

	// Installed and pinned to the same version: nothing to do.
	if err := writePinnedG4FVersion(binDir, "1.5.0"); err != nil {
		t.Fatalf("write pin stamp: %v", err)
	}
	if needsInstall(binDir, "1.5.0") {
		t.Fatal("needsInstall() with matching pin = true, want false")
	}

	// Installed at another version than the pin: reinstall.
	if !needsInstall(binDir, "1.4.0") {
		t.Fatal("needsInstall() with changed pin = false, want true")
	}
}

func TestSyncPinWithInstallSpec(t *testing.T) {
	binDir := t.TempDir()

	// Bare "g4f" unpins.
	syncPinWithInstallSpec(binDir, []string{"install", "g4f"})
	if got := readPinnedG4FVersion(binDir); got != "" {
		t.Fatalf("pin after 'install g4f' = %q, want empty", got)
	}

	// "g4f==X.Y.Z" pins.
	syncPinWithInstallSpec(binDir, []string{"install", "g4f==1.5.0"})
	if got := readPinnedG4FVersion(binDir); got != "1.5.0" {
		t.Fatalf("pin after 'install g4f==1.5.0' = %q, want 1.5.0", got)
	}

	// Extras are handled too.
	syncPinWithInstallSpec(binDir, []string{"install", "g4f[slim]==1.6.1"})
	if got := readPinnedG4FVersion(binDir); got != "1.6.1" {
		t.Fatalf("pin after 'install g4f[slim]==1.6.1' = %q, want 1.6.1", got)
	}

	// Other packages and uninstall leave the pin untouched.
	syncPinWithInstallSpec(binDir, []string{"install", "requests==2.0"})
	if got := readPinnedG4FVersion(binDir); got != "1.6.1" {
		t.Fatalf("pin after unrelated install = %q, want 1.6.1", got)
	}
	syncPinWithInstallSpec(binDir, []string{"uninstall", "g4f"})
	if got := readPinnedG4FVersion(binDir); got != "1.6.1" {
		t.Fatalf("pin after uninstall = %q, want 1.6.1", got)
	}
}
