package sandbox

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// udocker is the user-space container runner used on Termux. It can pull and
// execute the same g4f images as Docker without a daemon or root, but exposes a
// much smaller CLI: containers are created with `create`, every execution is a
// `run`, and there is no start/stop/exec/logs lifecycle. This backend keeps a
// detached `udocker run` process (like the native Termux backend) and uses the
// container name for run/rm/inspect.

const udockerHint = "pip install udocker"

// useUdocker returns true when the udocker backend should run. Termux selects
// it automatically; elsewhere it must be requested with --backend udocker or
// --engine udocker.
func useUdocker(o options) bool {
	if o.backend == "udocker" {
		return true
	}
	if o.engine != "" || o.backend != "" {
		return false
	}
	return isTermux()
}

func udockerAutoInstall() bool {
	switch strings.ToLower(os.Getenv("G4F_SANDBOX_INSTALL_UDOCKER")) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// findUdockerBinary looks on PATH and in the locations pip uses on Termux
// (PREFIX/bin) and on ordinary Linux hosts (~/.local/bin, ~/bin).
func findUdockerBinary() (string, error) {
	if path, err := exec.LookPath("udocker"); err == nil {
		return path, nil
	}
	var candidates []string
	if prefix := os.Getenv("PREFIX"); prefix != "" {
		candidates = append(candidates, filepath.Join(prefix, "bin", "udocker"))
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		candidates = append(candidates,
			filepath.Join(home, ".local", "bin", "udocker"),
			filepath.Join(home, "bin", "udocker"),
		)
	}
	for _, path := range candidates {
		if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
			return path, nil
		}
	}
	return "", exec.ErrNotFound
}

// ensureUdocker returns the udocker binary with its tools installed, installing
// the pip package and downloading the tools on first use. udocker is a Python
// package on Termux (pkg install python comes first).
func ensureUdocker() (string, error) {
	path, err := findUdockerBinary()
	if err != nil {
		if !udockerAutoInstall() {
			return "", fmt.Errorf("udocker not found; install it with: %s", udockerHint)
		}
		py, perr := ensureTermuxPython()
		if perr != nil {
			return "", fmt.Errorf("udocker not found (%v); install it with: %s", perr, udockerHint)
		}
		fmt.Fprintln(os.Stderr, "Installing udocker with pip (first run only) ...")
		if err := pipInstall(py, "udocker"); err != nil {
			return "", fmt.Errorf("install udocker: %w", err)
		}
		path, err = findUdockerBinary()
		if err != nil {
			return "", fmt.Errorf("udocker was installed but is not on PATH; add %s/bin or ~/.local/bin to PATH", os.Getenv("PREFIX"))
		}
	}
	if err := ensureUdockerTools(path); err != nil {
		return "", err
	}
	return path, nil
}

// udockerDir is the udocker root (default ~/.udocker). It holds the tools,
// images and containers, so it should live on a filesystem with enough space.
func udockerDir() string {
	return strings.TrimSpace(os.Getenv("G4F_SANDBOX_UDOCKER_DIR"))
}

func udockerToolsDir() string {
	if dir := udockerDir(); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".udocker")
}

// udockerToolsInstalled reports whether `udocker install` has already unpacked
// the tools (proot and friends) into <UDOCKER_DIR>/bin.
func udockerToolsInstalled() bool {
	dir := udockerToolsDir()
	if dir == "" {
		return false
	}
	entries, err := os.ReadDir(filepath.Join(dir, "bin"))
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			return true
		}
	}
	return false
}

// ensureUdockerTools downloads the udocker tools on first use. udocker can
// fetch them lazily on the first pull/run, but doing it up front keeps the
// first sandbox start from failing halfway through.
func ensureUdockerTools(bin string) error {
	if udockerToolsInstalled() {
		return nil
	}
	if !udockerAutoInstall() {
		return fmt.Errorf("udocker tools are not installed; run: %s install (automatic install is disabled by G4F_SANDBOX_INSTALL_UDOCKER=0)", bin)
	}
	fmt.Fprintln(os.Stderr, "Installing udocker tools (first run only) ...")
	cmd := exec.Command(bin, "install")
	cmd.Env = udockerEnv()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("udocker install: %w", err)
	}
	if !udockerToolsInstalled() {
		return fmt.Errorf("udocker install finished but %s/bin is empty; set G4F_SANDBOX_UDOCKER_DIR to a writable directory", udockerToolsDir())
	}
	return nil
}

// udockerEnv applies the launcher's udocker configuration to a command. Only
// UDOCKER_DIR is needed here; the execution mode is applied with `udocker
// setup` because UDOCKER_DEFAULT_EXECUTION_MODE accepts fewer modes.
func udockerEnv() []string {
	env := os.Environ()
	if dir := udockerDir(); dir != "" {
		env = envWith(env, "UDOCKER_DIR", dir)
	}
	return env
}

func envWith(env []string, key, value string) []string {
	prefix := key + "="
	out := env[:0]
	for _, entry := range env {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return append(out, prefix+value)
}

// udockerSetupArgs returns the `udocker setup` invocation for the configured
// execution mode, or nil when no mode is configured. Valid modes are the ones
// documented by udocker (P1/P2, F1-F4, R1-R3, S1).
func udockerSetupArgs(o options) ([]string, error) {
	mode := strings.ToUpper(strings.TrimSpace(os.Getenv("G4F_SANDBOX_UDOCKER_EXECMODE")))
	switch mode {
	case "":
		return nil, nil
	case "P1", "P2", "F1", "F2", "F3", "F4", "R1", "R2", "R3", "S1":
		return []string{"setup", "--execmode=" + mode, o.name}, nil
	default:
		return nil, fmt.Errorf("invalid G4F_SANDBOX_UDOCKER_EXECMODE %q (use P1, P2, F1-F4, R1-R3 or S1)", mode)
	}
}

func udockerPaths(o options) (pidPath, logPath string) {
	return filepath.Join(o.dataDir, "udocker.pid"), filepath.Join(o.dataDir, "udocker.log")
}

func udockerStatePath(o options) string {
	return filepath.Join(o.dataDir, "udocker.state")
}

type udockerState struct {
	PID  int
	Port int
	Bind string
	Spec string
}

// udockerApplyState restores the port/bind of a detached run so `status` and
// `open` work without repeating the original flags.
func udockerApplyState(o options) options {
	if state, ok := readUdockerState(o); ok {
		if state.Port > 0 {
			o.port = state.Port
		}
		if state.Bind != "" {
			o.bind = state.Bind
		}
	}
	return o
}

func readUdockerState(o options) (udockerState, bool) {
	data, err := os.ReadFile(udockerStatePath(o))
	if err != nil {
		return udockerState{}, false
	}
	var state udockerState
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

func writeUdockerState(o options, pid int) error {
	state := fmt.Sprintf("pid=%d\nport=%d\nbind=%s\nspec=%s\n",
		pid, o.port, o.bind, udockerSpec(o))
	return os.WriteFile(udockerStatePath(o), []byte(state), 0o644)
}

func udockerSpec(o options) string {
	sum := sha256.Sum256([]byte(
		strings.Join(udockerRunArgs(o, nil), "\x00") + "\x00" + o.apiKey +
			"\x00" + udockerDir() +
			"\x00" + strings.ToUpper(strings.TrimSpace(os.Getenv("G4F_SANDBOX_UDOCKER_EXECMODE"))),
	))
	return hex.EncodeToString(sum[:8])
}

func udockerPlatform() string {
	switch runtime.GOARCH {
	case "amd64", "arm64":
		return "linux/" + runtime.GOARCH
	case "arm":
		return "linux/arm/v7"
	default:
		return ""
	}
}

func udockerOutput(bin string, args ...string) (string, error) {
	cmd := exec.Command(bin, args...)
	cmd.Env = udockerEnv()
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("udocker %s: %s", strings.Join(args, " "), msg)
	}
	return strings.TrimSpace(stdout.String()), nil
}

func udockerImageExists(bin, image string) bool {
	_, err := udockerOutput(bin, "inspect", image)
	return err == nil
}

func udockerContainerExists(bin, name string) bool {
	// inspect -p only succeeds for extracted containers, not for images.
	_, err := udockerOutput(bin, "inspect", "-p", name)
	return err == nil
}

func udockerRemoveContainer(bin, name string) error {
	if !udockerContainerExists(bin, name) {
		return nil
	}
	cmd := exec.Command(bin, "rm", "-f", name)
	cmd.Env = udockerEnv()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("udocker rm -f %s: %w", name, err)
	}
	return nil
}

func udockerPull(bin string, o options, force bool) error {
	if !force && udockerImageExists(bin, o.image) {
		return nil
	}
	args := []string{"pull"}
	if platform := udockerPlatform(); platform != "" {
		args = append(args, "--platform="+platform)
	}
	args = append(args, o.image)
	fmt.Fprintf(os.Stderr, "Pulling %s with udocker (%s)\n", o.image, udockerPlatform())
	cmd := exec.Command(bin, args...)
	cmd.Env = udockerEnv()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("udocker pull %s: %w", o.image, err)
	}
	return nil
}

func udockerEnsureImage(bin string, o options) error {
	exists := udockerImageExists(bin, o.image)
	if o.noPull {
		if !exists {
			return fmt.Errorf("image %s is not available locally and --no-pull was set", o.image)
		}
		return nil
	}
	if exists && !o.alwaysPull {
		return nil
	}
	return udockerPull(bin, o, true)
}

func udockerCreateContainer(bin string, o options) error {
	fmt.Fprintf(os.Stderr, "Creating udocker container %s from %s\n", o.name, o.image)
	if _, err := udockerOutput(bin, "create", "--name="+o.name, o.image); err != nil {
		return err
	}
	return nil
}

// udockerRunOptions builds the per-execution options. udocker stores no run
// configuration in the container, so every run (the server, exec and
// sandbox-cli) receives the same ports, volumes and environment.
func udockerRunOptions(o options) []string {
	args := []string{
		"run",
		"--nobanner",
		"--publish=" + fmt.Sprintf("%d:8080", o.port),
	}
	if o.mode == modeDesktop {
		args = append(args,
			"--publish="+fmt.Sprintf("%d:5900", o.vncPort),
			"--volume="+filepath.Join(o.dataDir, "config")+":/home/pn/.config",
			"--volume="+filepath.Join(o.dataDir, "local")+":/home/pn/.local",
			"--env=XDG_RUNTIME_DIR=/tmp/sandbox-runtime",
			"--env=SANDBOX_DESKTOP=1",
			"--env=SANDBOX_BLOCK_LOCAL_NETWORK="+strconv.Itoa(boolInt(o.blockLocal && !o.noFirewall)),
			"--env=SANDBOX_FIREWALL_STRICT="+strconv.Itoa(boolInt(o.firewallStrict && !o.noFirewall)),
			"--env=G4F_DOMAIN_SUFFIX="+o.domainSuffix,
			"--env=G4F_DOMAIN_ROUTER="+strconv.Itoa(boolInt(o.domainRouter)),
		)
		if o.appDir != "" {
			args = append(args, "--volume="+o.appDir+":/app/gpt4free")
		}
		if o.app != "" {
			args = append(args, "--env=G4F_DESKTOP_APP="+o.app)
		}
	} else {
		args = append(args,
			"--volume="+filepath.Join(o.dataDir, "har_and_cookies")+":/app/har_and_cookies",
			"--volume="+filepath.Join(o.dataDir, "generated_media")+":/app/generated_media",
		)
	}
	if o.apiKey != "" {
		args = append(args, "--env=G4F_API_KEY="+o.apiKey)
	}
	return args
}

func udockerRunArgs(o options, command []string) []string {
	args := append(udockerRunOptions(o), o.name)
	return append(args, command...)
}

// udockerExecArgs clears the image entrypoint. udocker has no exec command, so
// a second run is used to enter the same container tree; without this the
// desktop entrypoint would start Xvnc/Xfce again before running the command.
func udockerExecArgs(o options, command []string) []string {
	args := append(udockerRunOptions(o), "--entrypoint=", o.name)
	return append(args, command...)
}

func runUdocker(o options) error {
	switch o.action {
	case "up":
		return udockerUp(o)
	case "down":
		return udockerDown(o)
	case "restart":
		if err := udockerDown(o); err != nil {
			return err
		}
		return udockerUp(o)
	case "status":
		return udockerStatus(o)
	case "logs":
		return udockerLogs(o)
	case "shell":
		shell := os.Getenv("SHELL")
		if shell == "" {
			shell = "bash"
		}
		return udockerExec(o, []string{shell})
	case "exec":
		return udockerExec(o, o.forwarded)
	case "cli":
		return udockerCLI(o)
	case "install":
		if o.mode != modeDesktop {
			return fmt.Errorf("on-demand tools live in the desktop image; run: g4f-sandbox --desktop install <tool>")
		}
		if len(o.forwarded) == 0 {
			return fmt.Errorf("usage: g4f-sandbox --desktop install <tool> (see g4f-sandbox --desktop cli list)")
		}
		return udockerCLIWith(o, append([]string{"install"}, o.forwarded...))
	case "pull":
		bin, err := ensureUdocker()
		if err != nil {
			return err
		}
		return udockerPull(bin, o, true)
	case "build":
		return fmt.Errorf("udocker cannot build images; build with Docker or Podman and pass --image")
	default:
		return fmt.Errorf("%s is not available in the udocker backend", o.action)
	}
}

func udockerUp(o options) error {
	if o.buildFirst {
		return fmt.Errorf("udocker cannot build images; build with Docker or Podman and pass --image")
	}
	if err := ensureDataDir(o); err != nil {
		return err
	}
	bin, err := ensureUdocker()
	if err != nil {
		return err
	}

	pidPath, logPath := udockerPaths(o)
	statePath := udockerStatePath(o)
	if pid, ok := readPID(pidPath); ok && processAlive(pid) {
		state, hasState := readUdockerState(o)
		if !hasState || state.Spec == udockerSpec(o) {
			if hasState {
				o = udockerApplyState(o)
			}
			fmt.Printf("sandbox is already running (pid %d)\n", pid)
			printUdockerBanner(o)
			if o.open {
				_ = openBrowser(modeURL(o))
			}
			return nil
		}
		if err := udockerDown(o); err != nil {
			return err
		}
	}
	_ = os.Remove(pidPath)
	_ = os.Remove(statePath)

	if err := udockerEnsureImage(bin, o); err != nil {
		return err
	}
	if err := udockerRemoveContainer(bin, o.name); err != nil {
		return err
	}
	if err := udockerCreateContainer(bin, o); err != nil {
		return err
	}
	if setup, err := udockerSetupArgs(o); err != nil {
		return err
	} else if len(setup) > 0 {
		fmt.Fprintf(os.Stderr, "Configuring udocker execution mode %s\n", strings.ToUpper(os.Getenv("G4F_SANDBOX_UDOCKER_EXECMODE")))
		if _, err := udockerOutput(bin, setup...); err != nil {
			return err
		}
	}

	args := udockerRunArgs(o, nil)
	if !o.detach {
		printUdockerBanner(o)
		cmd := exec.Command(bin, args...)
		cmd.Dir = o.dataDir
		cmd.Env = udockerEnv()
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}

	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open log %s: %w", logPath, err)
	}
	defer logFile.Close()

	cmd := exec.Command(bin, args...)
	cmd.Dir = o.dataDir
	cmd.Env = udockerEnv()
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	configureDetached(cmd)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start udocker: %w", err)
	}
	pid := cmd.Process.Pid
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(pid)+"\n"), 0o644); err != nil {
		_ = terminateProcess(pid)
		return fmt.Errorf("write pid file %s: %w", pidPath, err)
	}
	if err := writeUdockerState(o, pid); err != nil {
		_ = terminateProcess(pid)
		_ = os.Remove(pidPath)
		return fmt.Errorf("write state file: %w", err)
	}
	_ = cmd.Process.Release()

	if err := waitForTCP(fmt.Sprintf("%s:%d", displayHost(o.bind), o.port), 120*time.Second); err != nil {
		tail := tailFile(logPath, 30)
		_ = terminateProcessGroup(pid)
		_ = os.Remove(pidPath)
		_ = os.Remove(statePath)
		if tail != "" {
			return fmt.Errorf("sandbox did not start: %w\n%s", err, tail)
		}
		return fmt.Errorf("sandbox did not start: %w (see %s)", err, logPath)
	}

	printUdockerBanner(o)
	if o.open {
		if err := openBrowser(modeURL(o)); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox: could not open a browser:", err)
			fmt.Fprintln(os.Stderr, "g4f-sandbox: open this URL manually:", modeURL(o))
		}
	}
	return nil
}

func udockerDown(o options) error {
	bin, _ := findUdockerBinary()
	pidPath, _ := udockerPaths(o)
	pid, hasPID := readPID(pidPath)
	running := hasPID && processAlive(pid)
	if running {
		if err := terminateProcessGroup(pid); err != nil {
			_ = terminateProcess(pid)
		}
		for i := 0; i < 50; i++ {
			if !processAlive(pid) {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}
		// Always sweep the group: the udocker wrapper can exit while its
		// PRoot child is still alive, and rm -f would then fail.
		_ = forceKillProcessGroup(pid)
		if processAlive(pid) {
			_ = forceKillProcess(pid)
		}
		fmt.Printf("stopped %s (pid %d)\n", o.name, pid)
	}
	_ = os.Remove(pidPath)
	_ = os.Remove(udockerStatePath(o))

	if o.keep {
		return nil
	}
	if bin == "" {
		if !running {
			fmt.Println("sandbox does not exist")
		}
		return nil
	}
	if !udockerContainerExists(bin, o.name) {
		if !running {
			fmt.Println("sandbox does not exist")
		}
		return nil
	}
	if err := udockerRemoveContainer(bin, o.name); err != nil {
		return err
	}
	fmt.Printf("removed %s\n", o.name)
	return nil
}

func udockerStatus(o options) error {
	o = udockerApplyState(o)
	fmt.Printf("backend:   udocker (user-space containers)\n")
	fmt.Printf("image:     %s\n", o.image)
	fmt.Printf("data:      %s\n", o.dataDir)
	if bin, err := findUdockerBinary(); err != nil {
		fmt.Printf("udocker:   not installed (%s)\n", udockerHint)
	} else {
		fmt.Printf("udocker:   %s\n", bin)
	}
	fmt.Printf("tools:     %s\n", udockerToolsDir())
	if mode := strings.ToUpper(strings.TrimSpace(os.Getenv("G4F_SANDBOX_UDOCKER_EXECMODE"))); mode != "" {
		fmt.Printf("execmode:  %s\n", mode)
	}
	pidPath, logPath := udockerPaths(o)
	pid, ok := readPID(pidPath)
	if ok && processAlive(pid) {
		fmt.Printf("sandbox:   running (pid %d)\n", pid)
		fmt.Printf("api:       %s\n", modeURL(o))
		if o.mode == modeDesktop {
			fmt.Printf("vnc:       %s:%d\n", displayHost(o.bind), o.vncPort)
		}
	} else {
		if ok {
			_ = os.Remove(pidPath)
			_ = os.Remove(udockerStatePath(o))
		}
		fmt.Printf("sandbox:   not running\n")
	}
	fmt.Printf("log:       %s\n", logPath)
	return nil
}

func udockerLogs(o options) error {
	_, logPath := udockerPaths(o)
	if _, err := os.Stat(logPath); err != nil {
		return fmt.Errorf("no log yet (%s); start the sandbox first with g4f-sandbox up", logPath)
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

func udockerExec(o options, command []string) error {
	if len(command) == 0 {
		return fmt.Errorf("no command given; usage: g4f-sandbox exec <command> [args...]")
	}
	bin, err := ensureUdocker()
	if err != nil {
		return err
	}
	if !udockerContainerExists(bin, o.name) {
		return fmt.Errorf("container %s does not exist; start it first with g4f-sandbox up", o.name)
	}
	args := udockerExecArgs(o, command)
	cmd := exec.Command(bin, args...)
	cmd.Dir = o.dataDir
	cmd.Env = udockerEnv()
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func udockerEnsureRunning(o options) error {
	pidPath, _ := udockerPaths(o)
	if pid, ok := readPID(pidPath); ok && processAlive(pid) {
		return nil
	}
	return udockerUp(o)
}

func udockerCLI(o options) error {
	if o.mode != modeDesktop {
		return fmt.Errorf("sandbox-cli is only available in desktop mode; run: g4f-sandbox --desktop cli <args>")
	}
	return udockerCLIWith(o, o.forwarded)
}

func udockerCLIWith(o options, args []string) error {
	o.mode = modeDesktop
	if err := udockerEnsureRunning(o); err != nil {
		return err
	}
	return udockerExec(o, append([]string{"sandbox-cli"}, args...))
}

func printUdockerBanner(o options) {
	fmt.Println()
	if o.mode == modeDesktop {
		fmt.Println("g4f sandbox desktop (Termux/udocker)")
		fmt.Printf("  container: %s\n", o.name)
		fmt.Printf("  image:     %s\n", o.image)
		fmt.Printf("  desktop:   %s\n", modeURL(o))
		fmt.Printf("  api:       http://%s:%d/\n", displayHost(o.bind), o.port)
		fmt.Printf("  vnc:       %s:%d\n", displayHost(o.bind), o.vncPort)
		fmt.Printf("  data:      %s\n", o.dataDir)
		fmt.Printf("  tools:     g4f-sandbox --desktop install <tool>\n")
	} else {
		fmt.Println("g4f sandbox (API only, Termux/udocker)")
		fmt.Printf("  container: %s\n", o.name)
		fmt.Printf("  image:     %s\n", o.image)
		fmt.Printf("  api:       %s\n", modeURL(o))
		fmt.Printf("  data:      %s\n", o.dataDir)
	}
	fmt.Printf("  logs:      g4f-sandbox logs\n")
	fmt.Printf("  stop:      g4f-sandbox down\n")
	if o.apiKey != "" {
		fmt.Println("  password:  G4F_API_KEY is set (any username works in the browser)")
	}
	fmt.Println()
}
