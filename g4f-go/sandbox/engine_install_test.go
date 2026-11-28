package sandbox

import (
	"strings"
	"testing"
)

func TestPodmanInstallCommands(t *testing.T) {
	for _, goos := range []string{"linux", "darwin", "windows"} {
		cmds := podmanInstallCommands(goos)
		if len(cmds) == 0 {
			t.Fatalf("podmanInstallCommands(%q) is empty", goos)
		}
		for _, c := range cmds {
			if c.name == "" || len(c.args) == 0 {
				t.Fatalf("podmanInstallCommands(%q) has an incomplete command: %+v", goos, c)
			}
		}
	}
	if got := podmanInstallCommands("plan9"); got != nil {
		t.Fatalf("podmanInstallCommands(plan9) = %v, want nil", got)
	}
}

func TestPodmanAutoInstallCanBeDisabled(t *testing.T) {
	for _, value := range []string{"0", "false", "no", "off", "FALSE"} {
		t.Setenv("G4F_SANDBOX_INSTALL_PODMAN", value)
		if podmanAutoInstall() {
			t.Errorf("podmanAutoInstall() = true with G4F_SANDBOX_INSTALL_PODMAN=%q", value)
		}
	}
	for _, value := range []string{"", "1", "true", "yes", "on"} {
		t.Setenv("G4F_SANDBOX_INSTALL_PODMAN", value)
		if !podmanAutoInstall() {
			t.Errorf("podmanAutoInstall() = false with G4F_SANDBOX_INSTALL_PODMAN=%q", value)
		}
	}
}

func TestPodmanFallbackPathsIncludeManagerLocations(t *testing.T) {
	paths := strings.Join(podmanFallbackPaths("darwin"), " ")
	if !strings.Contains(paths, "/opt/homebrew/bin/podman") {
		t.Fatalf("darwin fallback paths missing Homebrew: %s", paths)
	}
	t.Setenv("ProgramFiles", `C:\Program Files`)
	t.Setenv("LOCALAPPDATA", `C:\Users\test\AppData\Local`)
	paths = strings.Join(podmanFallbackPaths("windows"), " ")
	if !strings.Contains(paths, "podman.exe") {
		t.Fatalf("windows fallback paths missing podman.exe: %s", paths)
	}
}

func TestKiriInstallCommands(t *testing.T) {
	darwin := kiriInstallCommands("darwin")
	if len(darwin) == 0 {
		t.Fatal("kiriInstallCommands(darwin) is empty")
	}
	foundBrew := false
	for _, c := range darwin {
		if c.name == "brew" && len(c.args) > 0 {
			foundBrew = true
		}
	}
	if !foundBrew {
		t.Fatalf("darwin kiri install missing brew cask: %+v", darwin)
	}

	windows := kiriInstallCommands("windows")
	if len(windows) == 0 {
		t.Fatal("kiriInstallCommands(windows) is empty")
	}

	linux := kiriInstallCommands("linux")
	if linux != nil {
		t.Fatalf("kiriInstallCommands(linux) = %v, want nil (Kiri has no distro package)", linux)
	}
	if got := kiriInstallCommands("plan9"); got != nil {
		t.Fatalf("kiriInstallCommands(plan9) = %v, want nil", got)
	}
}

func TestKiriAutoInstallCanBeDisabled(t *testing.T) {
	for _, value := range []string{"0", "false", "no", "off", "FALSE"} {
		t.Setenv("G4F_SANDBOX_INSTALL_KIRI", value)
		if kiriAutoInstall() {
			t.Errorf("kiriAutoInstall() = true with G4F_SANDBOX_INSTALL_KIRI=%q", value)
		}
	}
	for _, value := range []string{"", "1", "true", "yes", "on"} {
		t.Setenv("G4F_SANDBOX_INSTALL_KIRI", value)
		if !kiriAutoInstall() {
			t.Errorf("kiriAutoInstall() = false with G4F_SANDBOX_INSTALL_KIRI=%q", value)
		}
	}
}

func TestKiriFallbackPathsIncludeManagerLocations(t *testing.T) {
	paths := strings.Join(kiriFallbackPaths("darwin"), " ")
	if !strings.Contains(paths, "/opt/homebrew/bin/kiri") {
		t.Fatalf("darwin fallback paths missing Homebrew: %s", paths)
	}
	t.Setenv("ProgramFiles", `C:\Program Files`)
	t.Setenv("LOCALAPPDATA", `C:\Users\test\AppData\Local`)
	paths = strings.Join(kiriFallbackPaths("windows"), " ")
	if !strings.Contains(paths, "kiri.exe") {
		t.Fatalf("windows fallback paths missing kiri.exe: %s", paths)
	}
}

func TestConfirmInstallKiriDeclinesWithoutTerminal(t *testing.T) {
	clearEnv(t)
	if confirmInstallKiri() {
		t.Fatal("confirmInstallKiri() should decline when stdin is not a terminal")
	}
}
