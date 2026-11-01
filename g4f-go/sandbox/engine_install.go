package sandbox

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// installCommand describes one non-interactive package-manager invocation that
// can install the Podman CLI. root marks commands that need elevated rights.
type installCommand struct {
	name string
	args []string
	root bool
}

// podmanInstallCommands lists the package managers we know how to drive, in
// the order they are tried. The first command that exists wins; when it fails,
// the next one is attempted.
func podmanInstallCommands(goos string) []installCommand {
	switch goos {
	case "linux":
		return []installCommand{
			{"apt-get", []string{"update"}, true},
			{"apt-get", []string{"install", "-y", "podman"}, true},
			{"dnf", []string{"install", "-y", "podman"}, true},
			{"yum", []string{"install", "-y", "podman"}, true},
			{"pacman", []string{"-S", "--noconfirm", "podman"}, true},
			{"zypper", []string{"--non-interactive", "install", "podman"}, true},
			{"apk", []string{"add", "--no-cache", "podman"}, true},
		}
	case "darwin":
		return []installCommand{
			{"brew", []string{"install", "podman"}, false},
		}
	case "windows":
		return []installCommand{
			{"winget", []string{"install", "--id", "RedHat.Podman", "-e", "--accept-source-agreements", "--accept-package-agreements"}, false},
			{"choco", []string{"install", "podman-cli", "-y"}, true},
			{"scoop", []string{"install", "podman"}, false},
		}
	default:
		return nil
	}
}

// podmanAutoInstall reports whether the launcher may try to install Podman
// when neither Docker nor Podman is present. Set
// G4F_SANDBOX_INSTALL_PODMAN=0 to opt out.
func podmanAutoInstall() bool {
	switch strings.ToLower(os.Getenv("G4F_SANDBOX_INSTALL_PODMAN")) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// installPodman tries the host's package manager to install the Podman CLI.
// It is only called after Docker and Podman were both missing.
func installPodman() error {
	if !podmanAutoInstall() {
		return fmt.Errorf("automatic Podman install is disabled (G4F_SANDBOX_INSTALL_PODMAN=0)")
	}

	cmds := podmanInstallCommands(runtime.GOOS)
	if len(cmds) == 0 {
		return fmt.Errorf("no supported package manager for %s", runtime.GOOS)
	}

	var tried []string
	var lastErr error
	for _, c := range cmds {
		if _, err := exec.LookPath(c.name); err != nil {
			continue
		}
		name, args := c.name, c.args
		if c.root && os.Geteuid() != 0 {
			if _, err := exec.LookPath("sudo"); err == nil {
				args = append([]string{name}, args...)
				name = "sudo"
			}
		}

		fmt.Fprintf(os.Stderr, "g4f-sandbox: no container engine found; installing Podman with %s\n", c.name)
		if err := runInstallCommand(name, args...); err != nil {
			lastErr = fmt.Errorf("%s %s: %w", c.name, strings.Join(c.args, " "), err)
			tried = append(tried, c.name)
			continue
		}
		if _, err := findPodmanBinary(); err == nil {
			if runtime.GOOS == "darwin" || runtime.GOOS == "windows" {
				preparePodmanMachine()
			}
			return nil
		}
		lastErr = fmt.Errorf("%s finished but podman is not on PATH", c.name)
		tried = append(tried, c.name)
	}

	if lastErr == nil {
		return fmt.Errorf("no supported package manager found (tried %s)", strings.Join(tried, ", "))
	}
	return fmt.Errorf("%s (tried %s)", lastErr, strings.Join(tried, ", "))
}

func runInstallCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// findPodmanBinary looks for Podman on PATH and in the usual install
// locations used on Windows and macOS, then returns its absolute path.
func findPodmanBinary() (string, error) {
	if path, err := exec.LookPath("podman"); err == nil {
		return path, nil
	}
	for _, path := range podmanFallbackPaths(runtime.GOOS) {
		if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
			return path, nil
		}
	}
	return "", exec.ErrNotFound
}

func podmanFallbackPaths(goos string) []string {
	switch goos {
	case "windows":
		var out []string
		for _, env := range []string{"ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"} {
			if dir := os.Getenv(env); dir != "" {
				out = append(out,
					filepath.Join(dir, "RedHat", "Podman", "podman.exe"),
					filepath.Join(dir, "Programs", "Podman", "podman.exe"),
				)
			}
		}
		return out
	case "darwin":
		return []string{
			"/opt/homebrew/bin/podman",
			"/usr/local/bin/podman",
		}
	default:
		return nil
	}
}

// preparePodmanMachine initialises and starts the Podman machine that Docker
// Desktop would otherwise provide on macOS and Windows. Failures are ignored:
// checkReady reports the actionable error a moment later.
func preparePodmanMachine() {
	bin, err := findPodmanBinary()
	if err != nil {
		return
	}
	_ = runInstallCommand(bin, "machine", "init")
	_ = runInstallCommand(bin, "machine", "start")
}
