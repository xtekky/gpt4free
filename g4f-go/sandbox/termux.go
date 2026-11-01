package sandbox

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The native Termux backend runs the g4f API directly with Termux's Python
// instead of a container. It is used when --backend termux is requested
// explicitly; Termux otherwise auto-selects udocker. It installs Python/g4f on
// demand and keeps the process in the selected data directory.

const termuxPythonHint = "pkg install python"

func isTermux() bool {
	if os.Getenv("G4F_SANDBOX_TERMUX") == "1" {
		return true
	}
	if os.Getenv("TERMUX_VERSION") != "" {
		return true
	}
	return strings.Contains(os.Getenv("PREFIX"), "com.termux")
}

// useTermux returns true only when the native Termux Python backend was
// requested explicitly. Termux now auto-selects udocker (see useUdocker), so
// the native path stays available as an API-only fallback via --backend termux.
func useTermux(o options) bool {
	return o.backend == "termux"
}

func termuxPaths(o options) (pidPath, logPath string) {
	return filepath.Join(o.dataDir, "g4f.pid"), filepath.Join(o.dataDir, "g4f.log")
}

func termuxStatePath(o options) string {
	return filepath.Join(o.dataDir, "g4f.state")
}

type termuxState struct {
	PID  int
	Port int
	Bind string
	Spec string
}

// termuxApplyState restores the port/bind used by a running server, so
// `status` and `open` work without repeating the original flags.
func termuxApplyState(o options) options {
	if state, ok := readTermuxState(o); ok {
		if state.Port > 0 {
			o.port = state.Port
		}
		if state.Bind != "" {
			o.bind = state.Bind
		}
	}
	return o
}

func readTermuxState(o options) (termuxState, bool) {
	data, err := os.ReadFile(termuxStatePath(o))
	if err != nil {
		return termuxState{}, false
	}
	var state termuxState
	for _, line := range strings.Split(string(data), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		switch key {
		case "pid":
			state.PID, _ = strconv.Atoi(value)
		case "port":
			state.Port, _ = strconv.Atoi(value)
		case "bind":
			state.Bind = value
		case "spec":
			state.Spec = value
		}
	}
	return state, state.PID > 0
}

func writeTermuxState(o options, pid int) error {
	state := fmt.Sprintf("pid=%d\nport=%d\nbind=%s\nspec=%s\n", pid, o.port, o.bind, termuxSpec(o))
	return os.WriteFile(termuxStatePath(o), []byte(state), 0o644)
}

func termuxSpec(o options) string {
	sum := sha256.Sum256([]byte(strings.Join(termuxServerArgs(o), "\x00") + "\x00" + o.apiKey))
	return hex.EncodeToString(sum[:8])
}

func runTermux(o options) error {
	switch o.action {
	case "up":
		return termuxUp(o)
	case "down":
		return termuxDown(o)
	case "restart":
		if err := termuxDown(o); err != nil {
			return err
		}
		return termuxUp(o)
	case "status":
		return termuxStatus(o)
	case "logs":
		return termuxLogs(o)
	case "shell":
		shell := os.Getenv("SHELL")
		if shell == "" {
			shell = "sh"
		}
		return termuxExec(o, []string{shell})
	case "exec":
		return termuxExec(o, o.forwarded)
	case "install":
		if len(o.forwarded) == 0 {
			return fmt.Errorf("usage: g4f-sandbox install <python-package...>")
		}
		py, err := ensureTermuxPython()
		if err != nil {
			return err
		}
		return pipInstall(py, o.forwarded...)
	case "cli":
		return fmt.Errorf("sandbox-cli is only available in the Docker desktop image; on Termux use 'g4f-sandbox install <package>' or 'g4f-sandbox exec g4f ...'")
	case "pull":
		py, err := ensureTermuxPython()
		if err != nil {
			return err
		}
		if err := ensureTermuxG4F(py); err != nil {
			return err
		}
		fmt.Printf("Termux runtime ready: %s\n", py)
		return nil
	case "build":
		return fmt.Errorf("building the desktop image requires Docker or Podman, not Termux")
	default:
		return fmt.Errorf("%s is not available in the Termux backend", o.action)
	}
}

func termuxUp(o options) error {
	if o.mode == modeDesktop {
		return fmt.Errorf("the native Termux backend runs the API only; use --backend udocker for desktop mode")
	}
	if o.buildFirst {
		return fmt.Errorf("--build needs Docker or Podman; the native Termux backend runs the API only")
	}

	py, err := ensureTermuxPython()
	if err != nil {
		return err
	}
	if err := ensureTermuxG4F(py); err != nil {
		return err
	}
	if err := os.MkdirAll(o.dataDir, 0o755); err != nil {
		return fmt.Errorf("create data directory %s: %w", o.dataDir, err)
	}

	pidPath, logPath := termuxPaths(o)
	if pid, ok := readPID(pidPath); ok && processAlive(pid) {
		state, hasState := readTermuxState(o)
		if !hasState || state.Spec == termuxSpec(o) {
			if hasState {
				o = termuxApplyState(o)
			}
			fmt.Printf("g4f is already running (pid %d)\n", pid)
			printTermuxBanner(o, py)
			if o.open {
				_ = openBrowser(modeURL(o))
			}
			return nil
		}
		if err := termuxDown(o); err != nil {
			return err
		}
	}
	_ = os.Remove(pidPath)
	_ = os.Remove(termuxStatePath(o))

	args := termuxServerArgs(o)
	if !o.detach {
		printTermuxBanner(o, py)
		return termuxForeground(o, py, args)
	}

	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open log %s: %w", logPath, err)
	}
	defer logFile.Close()

	cmd := exec.Command(py, args...)
	cmd.Dir = o.dataDir
	cmd.Env = termuxEnv(o)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	configureDetached(cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start g4f: %w", err)
	}
	pid := cmd.Process.Pid
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(pid)+"\n"), 0o644); err != nil {
		_ = terminateProcess(pid)
		return fmt.Errorf("write pid file %s: %w", pidPath, err)
	}
	if err := writeTermuxState(o, pid); err != nil {
		_ = terminateProcess(pid)
		_ = os.Remove(pidPath)
		return fmt.Errorf("write state file: %w", err)
	}
	_ = cmd.Process.Release()

	if err := waitForTCP(fmt.Sprintf("%s:%d", displayHost(o.bind), o.port), 60*time.Second); err != nil {
		tail := tailFile(logPath, 30)
		_ = terminateProcess(pid)
		_ = os.Remove(pidPath)
		_ = os.Remove(termuxStatePath(o))
		if tail != "" {
			return fmt.Errorf("g4f did not start: %w\n%s", err, tail)
		}
		return fmt.Errorf("g4f did not start: %w (see %s)", err, logPath)
	}

	printTermuxBanner(o, py)
	if o.open {
		if err := openBrowser(modeURL(o)); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox: could not open a browser:", err)
			fmt.Fprintln(os.Stderr, "g4f-sandbox: open this URL manually:", modeURL(o))
		}
	}
	return nil
}

func termuxForeground(o options, py string, args []string) error {
	cmd := exec.Command(py, args...)
	cmd.Dir = o.dataDir
	cmd.Env = termuxEnv(o)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func termuxDown(o options) error {
	pidPath, _ := termuxPaths(o)
	pid, ok := readPID(pidPath)
	if !ok {
		fmt.Println("g4f is not running")
		return nil
	}
	if !processAlive(pid) {
		_ = os.Remove(pidPath)
		_ = os.Remove(termuxStatePath(o))
		fmt.Println("g4f is not running (removed stale pid file)")
		return nil
	}
	if err := terminateProcess(pid); err != nil {
		return fmt.Errorf("stop pid %d: %w", pid, err)
	}
	for i := 0; i < 50; i++ {
		if !processAlive(pid) {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if processAlive(pid) {
		_ = forceKillProcess(pid)
	}
	_ = os.Remove(pidPath)
	_ = os.Remove(termuxStatePath(o))
	fmt.Printf("stopped g4f (pid %d)\n", pid)
	return nil
}

func termuxStatus(o options) error {
	o = termuxApplyState(o)
	py, _ := findTermuxPython()
	fmt.Printf("backend:   termux (native API only)\n")
	fmt.Printf("data:      %s\n", o.dataDir)
	if py != "" {
		fmt.Printf("python:    %s\n", py)
	} else {
		fmt.Printf("python:    not installed (%s)\n", termuxPythonHint)
	}
	pidPath, logPath := termuxPaths(o)
	pid, ok := readPID(pidPath)
	if ok && processAlive(pid) {
		fmt.Printf("g4f:       running (pid %d)\n", pid)
		fmt.Printf("api:       %s\n", modeURL(o))
	} else {
		if ok {
			_ = os.Remove(pidPath)
			_ = os.Remove(termuxStatePath(o))
		}
		fmt.Printf("g4f:       not running\n")
	}
	fmt.Printf("log:       %s\n", logPath)
	return nil
}

func termuxLogs(o options) error {
	_, logPath := termuxPaths(o)
	if _, err := os.Stat(logPath); err != nil {
		return fmt.Errorf("no log yet (%s); start it first with g4f-sandbox up", logPath)
	}
	tail, err := exec.LookPath("tail")
	if err != nil {
		data, rerr := os.ReadFile(logPath)
		if rerr != nil {
			return rerr
		}
		fmt.Print(string(data))
		return nil
	}
	cmd := exec.Command(tail, "-n", "200", "-f", logPath)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func termuxExec(o options, command []string) error {
	if len(command) == 0 {
		return fmt.Errorf("no command given; usage: g4f-sandbox exec <command> [args...]")
	}
	if err := os.MkdirAll(o.dataDir, 0o755); err != nil {
		return err
	}
	cmd := exec.Command(command[0], command[1:]...)
	cmd.Dir = o.dataDir
	cmd.Env = termuxEnv(o)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func termuxServerArgs(o options) []string {
	// g4f's API binds 0.0.0.0 by default. Mirror the launcher's safe default
	// (127.0.0.1) and let --bind 0.0.0.0 expose it on the LAN.
	bind := fmt.Sprintf("%s:%d", o.bind, o.port)
	return []string{"-m", "g4f", "--port", strconv.Itoa(o.port), "--bind", bind}
}

func termuxEnv(o options) []string {
	env := os.Environ()
	env = setEnv(env, "PYTHONUNBUFFERED", "1")
	if o.apiKey != "" {
		env = setEnv(env, "G4F_API_KEY", o.apiKey)
	}
	return env
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, prefix+value)
}

func printTermuxBanner(o options, py string) {
	fmt.Println()
	fmt.Println("g4f sandbox (Termux, API only)")
	fmt.Printf("  api:     %s\n", modeURL(o))
	fmt.Printf("  python:  %s\n", py)
	fmt.Printf("  data:    %s\n", o.dataDir)
	fmt.Printf("  logs:    g4f-sandbox logs\n")
	fmt.Printf("  stop:    g4f-sandbox down\n")
	if o.apiKey != "" {
		fmt.Println("  password: G4F_API_KEY is set (any username works in the browser)")
	}
	fmt.Println()
}

func findTermuxPython() (string, error) {
	for _, name := range []string{"python", "python3"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("Python not found; install it with: %s", termuxPythonHint)
}

func ensureTermuxPython() (string, error) {
	if py, err := findTermuxPython(); err == nil {
		return py, nil
	}
	pkg, err := exec.LookPath("pkg")
	if err != nil {
		return "", fmt.Errorf("Python not found; install it with: %s", termuxPythonHint)
	}
	fmt.Fprintln(os.Stderr, "Installing Python with pkg ...")
	cmd := exec.Command(pkg, "install", "-y", "python")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("pkg install python failed: %w", err)
	}
	return findTermuxPython()
}

func ensureTermuxG4F(py string) error {
	check := exec.Command(py, "-c", "import g4f")
	if check.Run() == nil {
		return nil
	}
	fmt.Fprintln(os.Stderr, "Installing g4f with pip (first run only) ...")
	return pipInstall(py, "g4f[slim]")
}

func pipInstall(py string, packages ...string) error {
	args := append([]string{"-m", "pip", "install", "--upgrade"}, packages...)
	cmd := exec.Command(py, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err == nil {
		return nil
	}

	// A minimal Termux Python may not ship pip; bootstrap it and retry once.
	bootstrap := exec.Command(py, "-m", "ensurepip", "--upgrade")
	bootstrap.Stdout = os.Stdout
	bootstrap.Stderr = os.Stderr
	if err := bootstrap.Run(); err != nil {
		return fmt.Errorf("pip is not available in %s; run: %s -m ensurepip", py, py)
	}
	cmd = exec.Command(py, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("pip install %s failed: %w", strings.Join(packages, " "), err)
	}
	return nil
}

func readPID(path string) (int, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

func waitForTCP(address string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		conn, err := net.DialTimeout("tcp", address, 2*time.Second)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		lastErr = err
		if time.Now().After(deadline) {
			return lastErr
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func tailFile(path string, maxLines int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	const maxBytes = 128 * 1024
	if info, err := f.Stat(); err == nil && info.Size() > maxBytes {
		_, _ = f.Seek(info.Size()-maxBytes, io.SeekStart)
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	return strings.Join(lines, "\n")
}
