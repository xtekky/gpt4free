package polli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const polliPackage = "@pollinations/cli"

// pkgStep is one package-manager invocation. optional steps (apt-get update)
// may fail without aborting the rest of the group.
type pkgStep struct {
	name     string
	args     []string
	root     bool
	optional bool
}

type pkgGroup []pkgStep

func runCommand(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func runInteractive(name string, args ...string) int {
	cmd := exec.Command(name, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	return 0
}

func runPackageGroup(label string, group pkgGroup) error {
	if len(group) == 0 {
		return fmt.Errorf("no install commands for %s", runtime.GOOS)
	}
	if _, err := exec.LookPath(group[0].name); err != nil {
		return err
	}

	for _, step := range group {
		name, args := step.name, step.args
		if step.root && os.Geteuid() != 0 {
			if sudo, err := exec.LookPath("sudo"); err == nil {
				args = append([]string{name}, args...)
				name = sudo
			}
		}
		fmt.Fprintf(os.Stderr, "polli-sandbox: %s with %s\n", label, step.name)
		if err := runCommand(name, args...); err != nil {
			if step.optional {
				continue
			}
			return err
		}
	}
	return nil
}

func runFirstPackageGroup(label string, groups []pkgGroup) error {
	if len(groups) == 0 {
		return fmt.Errorf("no supported package manager for %s", runtime.GOOS)
	}
	var lastErr error
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		if _, err := exec.LookPath(group[0].name); err != nil {
			continue
		}
		if err := runPackageGroup(label, group); err == nil {
			return nil
		} else {
			lastErr = err
		}
	}
	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("no supported package manager found for %s", runtime.GOOS)
}

// nodeInstallGroups lists how to install Node.js and npm on the host.
func nodeInstallGroups(goos string) []pkgGroup {
	switch goos {
	case "linux":
		return []pkgGroup{
			{
				{name: "apt-get", args: []string{"update"}, root: true, optional: true},
				{name: "apt-get", args: []string{"install", "-y", "nodejs", "npm"}, root: true},
			},
			{
				{name: "dnf", args: []string{"install", "-y", "nodejs", "npm"}, root: true},
			},
			{
				{name: "yum", args: []string{"install", "-y", "nodejs", "npm"}, root: true},
			},
			{
				{name: "pacman", args: []string{"-S", "--noconfirm", "nodejs", "npm"}, root: true},
			},
			{
				{name: "zypper", args: []string{"--non-interactive", "install", "nodejs", "npm"}, root: true},
			},
			{
				{name: "apk", args: []string{"add", "--no-cache", "nodejs", "npm"}, root: true},
			},
		}
	case "darwin":
		return []pkgGroup{
			{
				{name: "brew", args: []string{"install", "node"}},
			},
		}
	case "windows":
		return []pkgGroup{
			{
				{name: "winget", args: []string{"install", "--id", "OpenJS.NodeJS.LTS", "-e", "--accept-source-agreements", "--accept-package-agreements"}},
			},
			{
				{name: "choco", args: []string{"install", "nodejs-lts", "-y"}, root: true},
			},
			{
				{name: "scoop", args: []string{"install", "nodejs"}},
			},
		}
	default:
		return nil
	}
}

func sshInstallGroups(goos string) []pkgGroup {
	switch goos {
	case "linux":
		return []pkgGroup{
			{
				{name: "apt-get", args: []string{"update"}, root: true, optional: true},
				{name: "apt-get", args: []string{"install", "-y", "openssh-client"}, root: true},
			},
			{
				{name: "dnf", args: []string{"install", "-y", "openssh-clients"}, root: true},
			},
			{
				{name: "yum", args: []string{"install", "-y", "openssh-clients"}, root: true},
			},
			{
				{name: "pacman", args: []string{"-S", "--noconfirm", "openssh"}, root: true},
			},
			{
				{name: "zypper", args: []string{"--non-interactive", "install", "openssh"}, root: true},
			},
			{
				{name: "apk", args: []string{"add", "--no-cache", "openssh-client"}, root: true},
			},
		}
	case "darwin":
		return []pkgGroup{
			{
				{name: "brew", args: []string{"install", "openssh"}},
			},
		}
	case "windows":
		return []pkgGroup{
			{
				{name: "winget", args: []string{"install", "--id", "Microsoft.OpenSSH.Beta", "-e", "--accept-source-agreements", "--accept-package-agreements"}},
			},
			{
				{name: "choco", args: []string{"install", "openssh", "-y"}, root: true},
			},
		}
	default:
		return nil
	}
}

func extraNodePaths(goos string) []string {
	switch goos {
	case "windows":
		var out []string
		for _, env := range []string{"ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"} {
			if dir := os.Getenv(env); dir != "" {
				out = append(out,
					filepath.Join(dir, "nodejs"),
					filepath.Join(dir, "Programs", "nodejs"),
				)
			}
		}
		return out
	case "darwin":
		return []string{"/opt/homebrew/bin", "/usr/local/bin"}
	default:
		return []string{"/usr/local/bin", "/usr/bin"}
	}
}

func extraPolliPaths(goos string) []string {
	home, _ := os.UserHomeDir()
	var out []string
	out = append(out, extraNodePaths(goos)...)
	if home != "" {
		out = append(out,
			filepath.Join(home, ".local", "bin"),
			filepath.Join(home, ".npm-global", "bin"),
		)
		if goos == "windows" {
			if appData := os.Getenv("APPDATA"); appData != "" {
				out = append(out, filepath.Join(appData, "npm"))
			}
		}
	}
	return out
}

func prependPath(dirs ...string) {
	if len(dirs) == 0 {
		return
	}
	path := os.Getenv("PATH")
	parts := strings.Split(path, string(os.PathListSeparator))
	seen := make(map[string]bool, len(parts))
	for _, part := range parts {
		seen[part] = true
	}
	var add []string
	for _, dir := range dirs {
		if dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		add = append(add, dir)
	}
	if len(add) == 0 {
		return
	}
	_ = os.Setenv("PATH", strings.Join(append(add, parts...), string(os.PathListSeparator)))
}

func npmGlobalBin() string {
	out, err := exec.Command("npm", "prefix", "-g").Output()
	if err != nil {
		return ""
	}
	prefix := strings.TrimSpace(string(out))
	if prefix == "" {
		return ""
	}
	if runtime.GOOS == "windows" {
		return prefix
	}
	return filepath.Join(prefix, "bin")
}

func findPolli() (string, error) {
	prependPath(npmGlobalBin())
	prependPath(extraPolliPaths(runtime.GOOS)...)
	if path, err := exec.LookPath("polli"); err == nil {
		return path, nil
	}
	for _, dir := range extraPolliPaths(runtime.GOOS) {
		for _, name := range []string{"polli", "polli.cmd", "polli.exe"} {
			path := filepath.Join(dir, name)
			if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
				return path, nil
			}
		}
	}
	return "", exec.ErrNotFound
}

func findSSHTools() (string, string, error) {
	prependPath(extraNodePaths(runtime.GOOS)...)
	if runtime.GOOS == "windows" {
		prependPath(`C:\Windows\System32\OpenSSH`)
	}
	ssh, err := exec.LookPath("ssh")
	if err != nil {
		return "", "", err
	}
	keygen, err := exec.LookPath("ssh-keygen")
	if err != nil {
		return "", "", err
	}
	return ssh, keygen, nil
}

// ensurePolli installs the Pollinations CLI (`polli`) when it is missing.
func ensurePolli() error {
	if _, err := findPolli(); err == nil {
		return nil
	}
	if _, err := exec.LookPath("npm"); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox: npm not found; installing Node.js")
		if err := runFirstPackageGroup("installing Node.js", nodeInstallGroups(runtime.GOOS)); err != nil {
			return fmt.Errorf("could not install Node.js/npm: %w", err)
		}
		prependPath(extraNodePaths(runtime.GOOS)...)
		if _, err := exec.LookPath("npm"); err != nil {
			return fmt.Errorf("npm is still not available after installing Node.js")
		}
	}

	args := []string{"install", "-g", polliPackage, "--no-fund", "--no-audit"}
	fmt.Fprintf(os.Stderr, "polli-sandbox: installing %s with npm\n", polliPackage)
	err := runCommand("npm", args...)
	if err != nil && os.Geteuid() != 0 {
		if sudo, serr := exec.LookPath("sudo"); serr == nil {
			fmt.Fprintln(os.Stderr, "polli-sandbox: retrying the global npm install with sudo")
			err = runCommand(sudo, append([]string{"npm"}, args...)...)
		}
	}
	if err != nil {
		return fmt.Errorf("npm install -g %s failed: %w", polliPackage, err)
	}
	if _, err := findPolli(); err != nil {
		return fmt.Errorf("%s was installed but the polli command was not found; add %s to PATH", polliPackage, npmGlobalBin())
	}
	return nil
}

// ensureSSH installs the OpenSSH client, which `polli sandbox` uses for its
// ProxyCommand and its <id>.polli host alias.
func ensureSSH() error {
	if _, _, err := findSSHTools(); err == nil {
		return nil
	}
	fmt.Fprintln(os.Stderr, "polli-sandbox: installing the OpenSSH client")
	if err := runFirstPackageGroup("installing OpenSSH", sshInstallGroups(runtime.GOOS)); err != nil {
		return fmt.Errorf("could not install OpenSSH: %w", err)
	}
	if _, _, err := findSSHTools(); err != nil {
		return fmt.Errorf("ssh/ssh-keygen are still not available after installing OpenSSH")
	}
	return nil
}
