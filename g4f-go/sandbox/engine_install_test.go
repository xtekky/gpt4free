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
