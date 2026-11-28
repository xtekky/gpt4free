package sandbox

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

// kiriInstallCommands lists package-manager invocations that can install Kiri.
// Kiri is not in distro package repositories on Linux, so it is installed from
// a GitHub release .deb/.rpm (see installKiriFromRelease) there.
func kiriInstallCommands(goos string) []installCommand {
	switch goos {
	case "darwin":
		return []installCommand{
			{"brew", []string{"install", "--cask", "libracoder/kiri/kiri"}, false},
		}
	case "windows":
		return []installCommand{
			{"winget", []string{"install", "--id", "LibraCoder.Kiri", "-e", "--accept-source-agreements", "--accept-package-agreements"}, false},
			{"choco", []string{"install", "kiri", "-y"}, true},
			{"scoop", []string{"install", "kiri"}, false},
		}
	case "linux":
		return nil
	default:
		return nil
	}
}

// kiriAutoInstall reports whether the launcher may try to install Kiri when no
// container engine is present. Set G4F_SANDBOX_INSTALL_KIRI=0 to opt out.
// Unlike Podman, Kiri is offered interactively first; this only gates whether
// the offer is made at all.
func kiriAutoInstall() bool {
	switch strings.ToLower(os.Getenv("G4F_SANDBOX_INSTALL_KIRI")) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// findKiriBinary looks for the Kiri CLI on PATH and in the usual install
// locations, then returns its absolute path.
func findKiriBinary() (string, error) {
	if path, err := exec.LookPath("kiri"); err == nil {
		return path, nil
	}
	for _, path := range kiriFallbackPaths(runtime.GOOS) {
		if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
			return path, nil
		}
	}
	return "", exec.ErrNotFound
}

func kiriFallbackPaths(goos string) []string {
	switch goos {
	case "darwin":
		return []string{
			"/Applications/Kiri.app/Contents/MacOS/Kiri",
			"/opt/homebrew/bin/kiri",
			"/usr/local/bin/kiri",
		}
	case "windows":
		var out []string
		for _, env := range []string{"ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"} {
			if dir := os.Getenv(env); dir != "" {
				out = append(out,
					filepath.Join(dir, "Kiri", "kiri.exe"),
					filepath.Join(dir, "Programs", "Kiri", "kiri.exe"),
				)
			}
		}
		return out
	default:
		return nil
	}
}

// kiriRelease is a minimal decoding of the GitHub "latest release" response.
type kiriRelease struct {
	TagName string `json:"tag_name"`
}

// installKiri tries the host's package manager to install the Kiri CLI. On
// Linux, where Kiri ships no distro package, it downloads the lite .deb/.rpm
// from the latest GitHub release and installs it. It is only called after
// Docker, Podman and Kiri were all missing.
func installKiri() error {
	if !kiriAutoInstall() {
		return fmt.Errorf("automatic Kiri install is disabled (G4F_SANDBOX_INSTALL_KIRI=0)")
	}

	cmds := kiriInstallCommands(runtime.GOOS)
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

		fmt.Fprintf(os.Stderr, "g4f-sandbox: no container engine found; installing Kiri with %s\n", c.name)
		if err := runInstallCommand(name, args...); err != nil {
			lastErr = fmt.Errorf("%s %s: %w", c.name, strings.Join(c.args, " "), err)
			tried = append(tried, c.name)
			continue
		}
		if _, err := findKiriBinary(); err == nil {
			return nil
		}
		lastErr = fmt.Errorf("%s finished but kiri is not on PATH", c.name)
		tried = append(tried, c.name)
	}

	if runtime.GOOS == "linux" {
		if err := installKiriFromRelease(); err == nil {
			if _, err := findKiriBinary(); err == nil {
				return nil
			}
		} else {
			lastErr = err
		}
	}

	if lastErr == nil {
		return fmt.Errorf("no supported package manager found for Kiri (tried %s)", strings.Join(tried, ", "))
	}
	return fmt.Errorf("%s; install Kiri manually from https://runkiri.com/releases/ or pass --engine", lastErr)
}

// installKiriFromRelease downloads the Kiri lite package for the host
// architecture from the latest GitHub release and installs it with the
// system package manager (dpkg or rpm/dnf).
func installKiriFromRelease() error {
	arch := runtime.GOARCH
	var debArch, rpmArch string
	switch arch {
	case "amd64":
		debArch, rpmArch = "amd64", "x86_64"
	case "arm64":
		debArch, rpmArch = "arm64", "aarch64"
	default:
		return fmt.Errorf("no Kiri package for %s", arch)
	}

	resp, err := http.Get("https://api.github.com/repos/libracoder/kiri/releases/latest")
	if err != nil {
		return fmt.Errorf("fetch latest Kiri release: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch latest Kiri release: HTTP %s", resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read latest Kiri release: %w", err)
	}
	var release kiriRelease
	if err := json.Unmarshal(body, &release); err != nil {
		return fmt.Errorf("parse latest Kiri release: %w", err)
	}
	tag := strings.TrimPrefix(release.TagName, "v")
	if tag == "" {
		return fmt.Errorf("latest Kiri release has no version tag")
	}

	root, err := os.MkdirTemp("", "g4f-sandbox-kiri-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(root)

	debURL := fmt.Sprintf("https://github.com/libracoder/kiri/releases/download/%s/kiri-lite_%s_%s.deb", tag, tag, debArch)
	rpmURL := fmt.Sprintf("https://github.com/libracoder/kiri/releases/download/%s/kiri-%s-1.%s.rpm", tag, tag, rpmArch)

	if _, err := exec.LookPath("dpkg"); err == nil {
		pkg := filepath.Join(root, "kiri.deb")
		if err := downloadFile(debURL, pkg); err != nil {
			return err
		}
		return installDeb(pkg)
	}
	haveRPM := false
	for _, cmd := range []string{"rpm", "dnf"} {
		if _, err := exec.LookPath(cmd); err == nil {
			haveRPM = true
			break
		}
	}
	if haveRPM {
		pkg := filepath.Join(root, "kiri.rpm")
		if err := downloadFile(rpmURL, pkg); err != nil {
			return err
		}
		return installRPM(pkg)
	}
	return fmt.Errorf("no dpkg/rpm/dnf found to install the Kiri package")
}

func downloadFile(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %s", url, resp.Status)
	}
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		return err
	}
	return nil
}

func installDeb(pkg string) error {
	args := []string{"-i", pkg}
	name := "dpkg"
	if os.Geteuid() != 0 {
		if _, err := exec.LookPath("sudo"); err != nil {
			return fmt.Errorf("installing %s needs root; re-run with sudo", pkg)
		}
		name = "sudo"
		args = append([]string{"dpkg"}, args...)
	}
	fmt.Fprintf(os.Stderr, "g4f-sandbox: installing Kiri package %s\n", pkg)
	return runInstallCommand(name, args...)
}

func installRPM(pkg string) error {
	if _, err := exec.LookPath("dnf"); err == nil {
		args := []string{"localinstall", "-y", pkg}
		name := "dnf"
		if os.Geteuid() != 0 {
			if _, err := exec.LookPath("sudo"); err != nil {
				return fmt.Errorf("installing %s needs root; re-run with sudo", pkg)
			}
			name = "sudo"
			args = append([]string{"dnf"}, args...)
		}
		fmt.Fprintf(os.Stderr, "g4f-sandbox: installing Kiri package %s\n", pkg)
		return runInstallCommand(name, args...)
	}
	args := []string{"-i", pkg}
	name := "rpm"
	if os.Geteuid() != 0 {
		if _, err := exec.LookPath("sudo"); err != nil {
			return fmt.Errorf("installing %s needs root; re-run with sudo", pkg)
		}
		name = "sudo"
		args = append([]string{"rpm"}, args...)
	}
	fmt.Fprintf(os.Stderr, "g4f-sandbox: installing Kiri package %s\n", pkg)
	return runInstallCommand(name, args...)
}

// confirmInstallKiri asks the user whether to install Kiri when no container
// engine was found. It only prompts on a terminal; elsewhere it declines so
// CI and piped runs fall back to Podman instead of hanging.
func confirmInstallKiri() bool {
	if !isatty(os.Stdin) {
		return false
	}
	fmt.Fprint(os.Stderr, "g4f-sandbox: no container engine found. Install Kiri now? [y/N] ")
	var answer string
	if _, err := fmt.Scanln(&answer); err != nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(answer)) {
	case "y", "yes":
		return true
	default:
		return false
	}
}

// askInstallKiri offers to install Kiri and, when accepted, installs it.
func askInstallKiri() error {
	if !kiriAutoInstall() {
		return fmt.Errorf("automatic Kiri install is disabled (G4F_SANDBOX_INSTALL_KIRI=0)")
	}
	if !confirmInstallKiri() {
		return fmt.Errorf("Kiri install declined")
	}
	return installKiri()
}

// isatty reports whether f is a terminal.
func isatty(f *os.File) bool {
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return (info.Mode() & os.ModeCharDevice) != 0
}
