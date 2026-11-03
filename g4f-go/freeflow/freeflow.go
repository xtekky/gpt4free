// Package freeflow runs the pi-freeflow proxy as a standalone local service.
//
// pi-freeflow is normally loaded as an extension by Pi or Oh My Pi (OMP).
// This command installs the same npm package into a private directory and
// starts its daemon directly, so `g4f-go freeflow` works without either host
// (and without the Python runtime).
package freeflow

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	// Plugin is the npm package that provides the free-model catalog and proxy.
	Plugin = "pi-freeflow"
	// DefaultPort is the proxy's default loopback port.
	DefaultPort = 28180
	// DefaultInstallSpec tracks the latest published package.
	DefaultInstallSpec = "pi-freeflow@latest"
	// DaemonGrace keeps a standalone daemon alive after its last client exits.
	// The extension normally retires after ten idle seconds; a user-started
	// proxy should keep serving until Ctrl-C, like `sandbox-cli start-freeflow-api`.
	DaemonGrace = "31536000000"
)

// Options are the parsed command line flags.
type Options struct {
	Action  string
	Port    int
	PortSet bool
	Force   bool
}

// Run executes `g4f-go freeflow [command]`.
func Run(args []string) int {
	opts, err := Parse(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 2
	}
	switch opts.Action {
	case "help":
		printHelp()
		return 0
	case "install":
		src, err := ensurePackage(opts.Force)
		if err != nil {
			fmt.Fprintln(os.Stderr, "freeflow:", err)
			return 1
		}
		fmt.Printf("%s is installed at %s\n", Plugin, src)
		return 0
	case "status":
		return status(opts)
	case "serve":
		return serve(opts)
	default:
		fmt.Fprintf(os.Stderr, "freeflow: unknown command %q (run: g4f-go freeflow help)\n", opts.Action)
		return 2
	}
}

// Parse accepts the command and a small set of flags in any position.
func Parse(args []string) (Options, error) {
	opts := Options{Action: "serve", Port: DefaultPort}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "help" || arg == "--help" || arg == "-h":
			opts.Action = "help"
		case arg == "serve" || arg == "start" || arg == "run":
			opts.Action = "serve"
		case arg == "install":
			opts.Action = "install"
		case arg == "status":
			opts.Action = "status"
		case arg == "--force" || arg == "-f":
			opts.Force = true
		case arg == "--port" || arg == "-p":
			if i+1 >= len(args) {
				return opts, fmt.Errorf("%s requires a port", arg)
			}
			i++
			port, err := parsePort(args[i])
			if err != nil {
				return opts, err
			}
			opts.Port = port
			opts.PortSet = true
		case strings.HasPrefix(arg, "--port="):
			port, err := parsePort(strings.TrimPrefix(arg, "--port="))
			if err != nil {
				return opts, err
			}
			opts.Port = port
			opts.PortSet = true
		case strings.HasPrefix(arg, "-") && arg != "-":
			return opts, fmt.Errorf("unknown option %q (run: g4f-go freeflow help)", arg)
		default:
			return opts, fmt.Errorf("unknown command %q (run: g4f-go freeflow help)", arg)
		}
	}
	return opts, nil
}

func parsePort(value string) (int, error) {
	n, err := strconv.Atoi(value)
	if err != nil || n < 1 || n > 65535 {
		return 0, fmt.Errorf("invalid port %q (expected 1-65535)", value)
	}
	return n, nil
}

func printHelp() {
	fmt.Print(`g4f-go freeflow - run the pi-freeflow proxy without Pi/OMP

Usage:
  g4f-go freeflow [serve] [--port N]   install (if needed) and run the proxy
  g4f-go freeflow install [--force]    install/update the proxy package only
  g4f-go freeflow status [--port N]    show the package, endpoint and API key
  g4f-go freeflow help                 show this help

The proxy listens on http://127.0.0.1:28180/v1 by default. Clients use the
bearer token from ~/.pi/agent/token (or PI_FREEFLOW_DATA_DIR/token). The
daemon keeps running until Ctrl-C; its log is written next to the token.
Node.js >=22.19 is preferred; Bun >=1.0 is used when Node.js is unavailable
or too old. npm or Bun is needed to install the package.

Environment:
  PI_FREEFLOW_PORT       proxy port (default 28180)
  PI_FREEFLOW_DATA_DIR   data directory for token, log and caches
  PI_CODING_AGENT_DIR    Pi agent directory (default ~/.pi/agent)
  G4F_FREEFLOW_HOME      where g4f-go keeps the standalone npm install
`)
}

func serve(opts Options) int {
	jsRuntime, err := findRuntime()
	if err != nil {
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 1
	}
	src, err := ensurePackage(opts.Force)
	if err != nil {
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 1
	}
	daemonDir, daemon, err := stageDaemon(src)
	if err != nil {
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 1
	}

	port := opts.Port
	if !opts.PortSet {
		port = envPort()
	}
	env := os.Environ()
	env = append(env, "PI_FREEFLOW_PORT="+strconv.Itoa(port))
	if os.Getenv("PI_FREEFLOW_DAEMON_GRACE_MS") == "" {
		env = append(env, "PI_FREEFLOW_DAEMON_GRACE_MS="+DaemonGrace)
	}

	fmt.Printf("Starting %s on http://127.0.0.1:%d/v1 (Ctrl-C to stop)\n", Plugin, port)

	cmd := exec.Command(jsRuntime.path, jsRuntime.args(daemon)...)
	cmd.Dir = daemonDir
	cmd.Env = env
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 1
	}
	if token := waitForToken(3 * time.Second); token != "" {
		fmt.Printf("API key:  %s\n", token)
	} else {
		fmt.Printf("API key:  not written yet (%s)\n", tokenPath())
	}
	if err := cmd.Wait(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			if code := exitErr.ExitCode(); code > 0 {
				return code
			}
		}
		fmt.Fprintln(os.Stderr, "freeflow:", err)
		return 1
	}
	return 0
}

// waitForToken gives a freshly started daemon a moment to create its token
// file. Existing tokens return immediately.
func waitForToken(timeout time.Duration) string {
	deadline := time.Now().Add(timeout)
	for {
		if token := readToken(); token != "" {
			return token
		}
		if time.Now().After(deadline) {
			return ""
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func envPort() int {
	raw := os.Getenv("PI_FREEFLOW_PORT")
	if raw == "" {
		return DefaultPort
	}
	port, err := parsePort(raw)
	if err != nil {
		return DefaultPort
	}
	return port
}

func status(opts Options) int {
	src, err := findPackage()
	if err != nil {
		fmt.Printf("%s is not installed (run: g4f-go freeflow install)\n", Plugin)
		return 1
	}
	port := opts.Port
	if !opts.PortSet {
		port = envPort()
	}
	fmt.Printf("package:  %s\n", src)
	fmt.Printf("base URL: http://127.0.0.1:%d/v1\n", port)
	fmt.Printf("log:      %s\n", filepath.Join(dataDir(), "pi-freeflow.log"))
	showAccess(port)

	health, err := probeHealth(port)
	if err != nil {
		fmt.Printf("daemon:   not running (%v)\n", err)
		return 0
	}
	fmt.Printf("daemon:   %s\n", health)
	return 0
}

func showAccess(port int) {
	token := readToken()
	if token == "" {
		fmt.Printf("API key:  not written yet (%s)\n", tokenPath())
		return
	}
	fmt.Printf("API key:  %s\n", token)
}

// probeHealth asks the running daemon for its version. The endpoint is
// authenticated with the same token the OpenAI client uses.
func probeHealth(port int) (string, error) {
	req, err := http.NewRequest("GET", fmt.Sprintf("http://127.0.0.1:%d/_health", port), nil)
	if err != nil {
		return "", err
	}
	if token := readToken(); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	client := &http.Client{Timeout: 2 * time.Second}
	res, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(res.Body, 4096))
	if res.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", res.StatusCode)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err == nil {
		if version, ok := payload["version"].(string); ok && version != "" {
			return "running v" + version, nil
		}
	}
	scanner := bufio.NewScanner(strings.NewReader(string(body)))
	if scanner.Scan() {
		return scanner.Text(), nil
	}
	return "running", nil
}

type jsRuntime struct {
	name string
	path string
}

type packageInstaller struct {
	name string
	path string
}

func (r jsRuntime) args(daemon string) []string {
	if r.name == "bun" {
		return []string{"run", daemon}
	}
	return []string{"--experimental-strip-types", daemon}
}

func findRuntime() (jsRuntime, error) {
	return selectRuntime(exec.LookPath, func(path string) (string, error) {
		out, err := exec.Command(path, "--version").Output()
		return strings.TrimSpace(string(out)), err
	})
}

func selectRuntime(lookPath func(string) (string, error), version func(string) (string, error)) (jsRuntime, error) {
	node, nodeErr := lookPath("node")
	if nodeErr == nil {
		nodeVersion, err := version(node)
		if err == nil && nodeVersionSupported(nodeVersion) {
			return jsRuntime{name: "node", path: node}, nil
		}
		if err != nil {
			nodeErr = fmt.Errorf("cannot run Node.js: %w", err)
		} else {
			nodeErr = fmt.Errorf("Node.js %s is too old (requires >=22.19)", nodeVersion)
		}
	}

	bun, bunErr := lookPath("bun")
	if bunErr == nil {
		bunVersion, err := version(bun)
		if err == nil && bunVersionSupported(bunVersion) {
			return jsRuntime{name: "bun", path: bun}, nil
		}
		if err != nil {
			bunErr = fmt.Errorf("cannot run Bun: %w", err)
		} else {
			bunErr = fmt.Errorf("Bun %s is too old or invalid (requires >=1.0)", bunVersion)
		}
	}

	if nodeErr != nil && bunErr != nil {
		return jsRuntime{}, fmt.Errorf("Node.js >=22.19 or Bun >=1.0 is required (%v; %v); install one or use sandbox-cli start-freeflow-api", nodeErr, bunErr)
	}
	if nodeErr != nil {
		return jsRuntime{}, fmt.Errorf("Node.js >=22.19 or Bun >=1.0 is required (%v); install one or use sandbox-cli start-freeflow-api", nodeErr)
	}
	return jsRuntime{}, fmt.Errorf("Node.js >=22.19 or Bun >=1.0 is required (%v); install one or use sandbox-cli start-freeflow-api", bunErr)
}

func nodeVersionSupported(version string) bool {
	version = strings.TrimPrefix(version, "v")
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return false
	}
	if major > 22 {
		return true
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return false
	}
	return major == 22 && minor >= 19
}

func selectPackageInstaller(lookPath func(string) (string, error), goos string) (packageInstaller, error) {
	npm, npmErr := lookPath("npm")
	if npmErr != nil && goos == "windows" {
		npm, npmErr = lookPath("npm.cmd")
	}
	if npmErr == nil {
		return packageInstaller{name: "npm", path: npm}, nil
	}
	bun, bunErr := lookPath("bun")
	if bunErr == nil {
		return packageInstaller{name: "Bun", path: bun}, nil
	}
	return packageInstaller{}, fmt.Errorf("npm or Bun is required to install %s (npm: %v; Bun: %v)", Plugin, npmErr, bunErr)
}

func bunVersionSupported(version string) bool {
	version = strings.TrimPrefix(strings.TrimSpace(version), "v")
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil || major < 1 {
		return false
	}
	_, err = strconv.Atoi(parts[1])
	return err == nil
}

// findPackage returns the installed pi-freeflow package directory. It prefers
// g4f-go's own install and falls back to an OMP plugin or Pi package, so a
// user who already has freeflow does not download it twice.
func findPackage() (string, error) {
	for _, dir := range packageCandidates() {
		if isPackageDir(dir) {
			return dir, nil
		}
	}
	return "", fs.ErrNotExist
}

func packageCandidates() []string {
	var dirs []string
	if home := os.Getenv("G4F_FREEFLOW_HOME"); home != "" {
		dirs = append(dirs, filepath.Join(expandHome(home), "node_modules", Plugin))
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs,
			filepath.Join(home, ".g4f", "freeflow", "node_modules", Plugin),
			filepath.Join(home, ".omp", "plugins", "node_modules", Plugin),
			filepath.Join(home, ".pi", "agent", "npm", "node_modules", Plugin),
		)
	}
	return dirs
}

func isPackageDir(dir string) bool {
	if info, err := os.Stat(filepath.Join(dir, "src", "daemon.ts")); err == nil && !info.IsDir() {
		return true
	}
	return false
}

// ensurePackage installs pi-freeflow with npm or Bun when it is not present.
// Force reinstalls it into g4f-go's own directory even if another host has a copy.
func ensurePackage(force bool) (string, error) {
	if !force {
		if src, err := findPackage(); err == nil {
			return src, nil
		}
	}
	installer, err := selectPackageInstaller(exec.LookPath, runtime.GOOS)
	if err != nil {
		return "", err
	}
	root := installRoot()
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	fmt.Fprintf(os.Stderr, "Installing %s into %s\n", DefaultInstallSpec, root)
	var cmd *exec.Cmd
	if installer.name == "npm" {
		cmd = exec.Command(installer.path, "install", "--prefix", root, "--no-audit", "--no-fund", DefaultInstallSpec)
	} else {
		cmd = exec.Command(installer.path, "add", DefaultInstallSpec)
		cmd.Dir = root
	}
	cmd.Env = os.Environ()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%s install %s failed: %w", installer.name, DefaultInstallSpec, err)
	}
	src := filepath.Join(root, "node_modules", Plugin)
	if !isPackageDir(src) {
		return "", fmt.Errorf("%s install %s did not create %s", installer.name, DefaultInstallSpec, src)
	}
	return src, nil
}

// stageDaemon copies the package out of node_modules. Node refuses to strip
// TypeScript types for files below a node_modules path, so the daemon always
// runs from a normal directory (the same workaround sandbox-cli uses).
func stageDaemon(src string) (string, string, error) {
	dest := filepath.Join(installRoot(), "daemon")
	stamp := filepath.Join(dest, ".version")
	version := packageVersion(src)
	if data, err := os.ReadFile(stamp); err == nil && strings.TrimSpace(string(data)) == version &&
		isPackageDir(dest) {
		if err := ensureUndici(src, dest); err != nil {
			return "", "", err
		}
		return dest, filepath.Join(dest, "src", "daemon.ts"), nil
	}
	if err := os.RemoveAll(dest); err != nil {
		return "", "", err
	}
	if err := copyTree(src, dest); err != nil {
		return "", "", err
	}
	if err := ensureUndici(src, dest); err != nil {
		return "", "", err
	}
	if err := os.WriteFile(stamp, []byte(version+"\n"), 0o644); err != nil {
		return "", "", err
	}
	return dest, filepath.Join(dest, "src", "daemon.ts"), nil
}

func packageVersion(src string) string {
	data, err := os.ReadFile(filepath.Join(src, "package.json"))
	if err != nil {
		return "0"
	}
	var pkg struct {
		Version string `json:"version"`
	}
	if json.Unmarshal(data, &pkg) != nil || pkg.Version == "" {
		return "0"
	}
	return pkg.Version
}

func ensureUndici(src, dest string) error {
	target := filepath.Join(dest, "node_modules", "undici")
	if info, err := os.Stat(target); err == nil && info.IsDir() {
		return nil
	}
	undici := filepath.Join(filepath.Dir(src), "undici")
	if info, err := os.Stat(undici); err != nil || !info.IsDir() {
		return fmt.Errorf("undici is missing next to %s; reinstall %s", src, Plugin)
	}
	return copyTree(undici, target)
}

func copyTree(src, dest string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dest, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}

func installRoot() string {
	if home := os.Getenv("G4F_FREEFLOW_HOME"); home != "" {
		return expandHome(home)
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".g4f", "freeflow")
	}
	return filepath.Join(os.TempDir(), "g4f-freeflow")
}

// dataDir mirrors pi-freeflow's own resolution order so the token, log and
// API key we print are the ones the daemon writes.
func dataDir() string {
	if dir := os.Getenv("PI_FREEFLOW_DATA_DIR"); strings.TrimSpace(dir) != "" {
		return expandHome(dir)
	}
	if dir := os.Getenv("PI_CODING_AGENT_DIR"); strings.TrimSpace(dir) != "" {
		return expandHome(dir)
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".pi", "agent")
	}
	return "."
}

func tokenPath() string {
	return filepath.Join(dataDir(), "token")
}

func readToken() string {
	data, err := os.ReadFile(tokenPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "~\\") {
		if home, err := os.UserHomeDir(); err == nil {
			if path == "~" {
				return home
			}
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
