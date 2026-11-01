// g4f-sandbox is a small, self-contained launcher for the g4f sandbox
// container image. It never embeds the image: it finds Docker, Podman or, on
// Termux, udocker, pulls only the image for the selected mode and the host
// architecture, and then creates or reuses the container.
//
// The default mode is API-only and uses the small `latest-slim` image. The
// full Xfce + TigerVNC desktop image is pulled only when --desktop is passed,
// so a first run does not download the desktop stack unless it is wanted.
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
)

// Version is overridden at build time with
// -ldflags "-X main.Version=<version>".
var Version = "0.1.0"

const (
	defaultDesktopImage = "hlohaus789/g4f:latest-sandbox-desktop"
	defaultAPIImage     = "hlohaus789/g4f:latest-slim"
	localDesktopImage   = "g4f-sandbox:desktop"

	defaultDesktopName = "g4f-sandbox-desktop"
	defaultAPIName     = "g4f-sandbox-api"

	defaultPort    = 8080
	defaultVNCPort = 5900
)

type sandboxMode string

const (
	modeAPI     sandboxMode = "api"
	modeDesktop sandboxMode = "desktop"
)

type options struct {
	action string
	mode   sandboxMode

	engine  string // docker, podman, udocker, or empty for autodetect
	backend string // "", "udocker", "termux" (native Termux Python) or "auto"
	image   string
	name    string
	app     string // desktop tool to launch instead of the live wallpaper

	port    int
	vncPort int
	bind    string

	dataDir string
	appDir  string

	apiKey       string
	domainSuffix string
	domainRouter bool

	blockLocal     bool
	firewallStrict bool
	noFirewall     bool

	open     bool
	detach   bool
	recreate bool
	keep     bool

	alwaysPull bool
	noPull     bool

	buildFirst bool
	build      bool
	userID     int
	context    string
	dockerfile string

	imageSet bool

	forwarded []string
}

// Run executes the sandbox launcher with the given command-line arguments and
// returns the process exit code. It is wired to `g4f-go sandbox`.
func Run(args []string) int {
	return run(args)
}

func run(args []string) int {
	o, err := parseArgs(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 2
	}

	switch o.action {
	case "help":
		printHelp()
		return 0
	case "version":
		fmt.Printf("g4f-sandbox %s\n", Version)
		return 0
	}

	// `open` only needs the computed URL, not a container engine.
	if o.action == "open" {
		if useUdocker(o) {
			o = udockerApplyState(o)
		} else if useTermux(o) {
			o = termuxApplyState(o)
		}
		u := modeURL(o)
		fmt.Println(u)
		if err := openBrowser(u); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox: could not open a browser:", err)
			fmt.Fprintln(os.Stderr, "g4f-sandbox: open this URL manually:", u)
			return 1
		}
		return 0
	}

	if useUdocker(o) {
		if err := runUdocker(o); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
			return 1
		}
		return 0
	}

	if useTermux(o) {
		if err := runTermux(o); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
			return 1
		}
		return 0
	}

	e, err := resolveEngine(o)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 1
	}

	switch o.action {
	case "up":
		err = runUp(e, o)
	case "down":
		err = runDown(e, o)
	case "restart":
		err = runRestart(e, o)
	case "status":
		err = runStatus(e, o)
	case "logs":
		err = runLogs(e, o)
	case "shell", "connect":
		err = runExec(e, o, []string{"bash"})
	case "exec":
		err = runExec(e, o, o.forwarded)
	case "cli":
		err = runCLI(e, o)
	case "install":
		err = runInstall(e, o)
	case "pull":
		err = ensureImage(e, o, true)
	case "build":
		err = runBuild(e, o)
	default:
		err = fmt.Errorf("unknown command: %s", o.action)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 1
	}
	return 0
}

// parseArgs accepts options before or after the command, except for the
// passthrough commands (exec, cli, install), where everything after the
// command belongs to the command itself.
func parseArgs(args []string) (options, error) {
	o := defaultOptions()

	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			if len(args[i+1:]) > 0 {
				o.forwarded = append(o.forwarded, args[i+1:]...)
			}
			break
		}
		if !strings.HasPrefix(arg, "-") {
			action, ok := parseCommand(arg)
			if !ok {
				return o, fmt.Errorf("unknown command %q (run g4f-sandbox help)", arg)
			}
			o.action = action
			if action == "exec" || action == "cli" || action == "install" {
				rest := args[i+1:]
				if len(rest) > 0 && rest[0] == "--" {
					rest = rest[1:]
				}
				o.forwarded = append(o.forwarded, rest...)
				break
			}
			continue
		}

		name, value, hasValue := splitFlag(arg)
		nextValue := func() (string, error) {
			if hasValue {
				return value, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("--%s requires a value", name)
			}
			i++
			return args[i], nil
		}

		switch name {
		case "h", "help":
			o.action = "help"
		case "v", "version":
			o.action = "version"
		case "d", "detach":
			o.detach = true
		case "f", "foreground":
			o.detach = false
		case "b", "build":
			o.buildFirst = true
		case "p", "port":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(s)
			if err != nil {
				return o, fmt.Errorf("invalid port %q", s)
			}
			o.port = n
		case "vnc-port":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(s)
			if err != nil {
				return o, fmt.Errorf("invalid VNC port %q", s)
			}
			o.vncPort = n
		case "bind", "bind-address":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.bind = s
		case "image":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.image = s
			o.imageSet = true
		case "name":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.name = s
		case "app":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			if s == "" {
				return o, fmt.Errorf("--app requires a sandbox-cli tool name")
			}
			o.app = s
		case "data-dir":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.dataDir = s
		case "app-dir":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.appDir = s
		case "api-key":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.apiKey = s
		case "domain-suffix":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.domainSuffix = s
		case "no-domain-router":
			o.domainRouter = false
		case "allow-local-network":
			o.blockLocal = false
		case "no-firewall":
			o.noFirewall = true
			o.blockLocal = false
		case "firewall-strict":
			o.firewallStrict = true
		case "no-open":
			o.open = false
		case "recreate":
			o.recreate = true
		case "keep":
			o.keep = true
		case "pull":
			o.alwaysPull = true
		case "no-pull":
			o.noPull = true
		case "desktop", "gui":
			o.mode = modeDesktop
		case "api", "api-only", "no-desktop":
			o.mode = modeAPI
		case "engine":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.engine = s
		case "backend":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.backend = s
		case "termux":
			o.backend = "termux"
		case "udocker":
			o.backend = "udocker"
		case "user-id", "uid":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			n, err := strconv.Atoi(s)
			if err != nil {
				return o, fmt.Errorf("invalid user id %q", s)
			}
			o.userID = n
		case "context":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.context = s
		case "dockerfile":
			s, err := nextValue()
			if err != nil {
				return o, err
			}
			o.dockerfile = s
		default:
			return o, fmt.Errorf("unknown option %q (run g4f-sandbox help)", arg)
		}
	}

	if err := o.finish(); err != nil {
		return o, err
	}
	return o, nil
}

func parseCommand(arg string) (string, bool) {
	switch arg {
	case "up", "start":
		return "up", true
	case "down", "stop":
		return "down", true
	case "restart":
		return "restart", true
	case "status":
		return "status", true
	case "logs":
		return "logs", true
	case "shell", "bash", "connect":
		return "shell", true
	case "exec":
		return "exec", true
	case "cli":
		return "cli", true
	case "install":
		return "install", true
	case "open":
		return "open", true
	case "pull":
		return "pull", true
	case "build":
		return "build", true
	case "version", "--version":
		return "version", true
	case "help", "--help":
		return "help", true
	default:
		return "", false
	}
}

func splitFlag(arg string) (name, value string, hasValue bool) {
	name = strings.TrimLeft(arg, "-")
	if i := strings.IndexByte(name, '='); i >= 0 {
		return name[:i], name[i+1:], true
	}
	return name, "", false
}

func (o *options) finish() error {
	if o.image == "" {
		o.image = defaultImage(o.mode)
	}
	if o.buildFirst && !o.imageSet && o.mode == modeDesktop {
		// A local build replaces the remote desktop tag unless the caller
		// explicitly asked for another image.
		o.image = localDesktopImage
	}
	if o.name == "" {
		o.name = defaultName(o.mode)
	}
	if o.app != "" {
		if o.mode != modeDesktop {
			return fmt.Errorf("--app requires --desktop")
		}
		for _, r := range o.app {
			if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
				continue
			}
			return fmt.Errorf("invalid app %q: use a sandbox-cli tool name (lowercase letters, numbers and hyphens)", o.app)
		}
		if o.app[0] == '-' {
			return fmt.Errorf("invalid app %q: tool name must not start with a hyphen", o.app)
		}
	}
	if o.bind == "" {
		return fmt.Errorf("bind address must not be empty")
	}
	if strings.Contains(o.bind, ":") {
		return fmt.Errorf("IPv6 bind addresses are not supported; use IPv4 or 0.0.0.0")
	}
	if o.port < 1 || o.port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535: %d", o.port)
	}
	if o.mode == modeDesktop {
		if o.vncPort < 1 || o.vncPort > 65535 {
			return fmt.Errorf("VNC port must be between 1 and 65535: %d", o.vncPort)
		}
		if o.port == o.vncPort {
			return fmt.Errorf("API/noVNC port and VNC port must differ (%d)", o.port)
		}
	}
	if o.dataDir == "" {
		return fmt.Errorf("data directory must not be empty")
	}
	absData, err := filepath.Abs(o.dataDir)
	if err != nil {
		return fmt.Errorf("data directory: %w", err)
	}
	o.dataDir = absData
	if o.appDir != "" {
		info, err := os.Stat(o.appDir)
		if err != nil {
			return fmt.Errorf("app directory: %w", err)
		}
		if !info.IsDir() {
			return fmt.Errorf("app directory is not a directory: %s", o.appDir)
		}
		absApp, err := filepath.Abs(o.appDir)
		if err != nil {
			return fmt.Errorf("app directory: %w", err)
		}
		o.appDir = absApp
	}
	if o.userID < 1 {
		return fmt.Errorf("user id must be greater than 0: %d", o.userID)
	}
	if o.domainSuffix == "" {
		return fmt.Errorf("domain suffix must not be empty")
	}
	for _, r := range o.domainSuffix {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '.' || r == '-' {
			continue
		}
		return fmt.Errorf("invalid domain suffix: %s", o.domainSuffix)
	}
	if o.engine != "" && o.engine != "docker" && o.engine != "podman" && o.engine != "udocker" {
		return fmt.Errorf("engine must be docker, podman or udocker: %s", o.engine)
	}
	if o.engine == "udocker" &&
		(o.backend == "docker" || o.backend == "podman" || o.backend == "container") {
		return fmt.Errorf("--engine udocker cannot be combined with --backend %s", o.backend)
	}
	switch o.backend {
	case "", "auto":
		o.backend = ""
	case "udocker":
		if o.engine != "" && o.engine != "udocker" {
			return fmt.Errorf("--backend udocker cannot be combined with --engine %s", o.engine)
		}
		o.engine = ""
	case "termux":
		if o.engine != "" {
			return fmt.Errorf("--termux cannot be combined with --engine %s", o.engine)
		}
		if o.mode == modeDesktop {
			return fmt.Errorf("the native Termux backend runs the API only; use --backend udocker for desktop mode")
		}
	case "docker", "podman", "container":
		if o.backend != "container" && o.engine == "" {
			o.engine = o.backend
		}
		o.backend = ""
	default:
		return fmt.Errorf("backend must be auto, docker, podman, udocker or termux: %s", o.backend)
	}
	if o.engine == "udocker" {
		if o.backend != "" {
			return fmt.Errorf("--engine udocker cannot be combined with --backend %s", o.backend)
		}
		o.backend = "udocker"
		o.engine = ""
	}
	if o.buildFirst && o.mode != modeDesktop && o.backend != "termux" {
		return fmt.Errorf("--build is only available in desktop mode; pass --desktop")
	}
	return nil
}

func defaultOptions() options {
	home, _ := os.UserHomeDir()
	if home == "" {
		home = "."
	}

	m := modeAPI
	switch strings.ToLower(os.Getenv("G4F_SANDBOX_MODE")) {
	case "desktop", "gui":
		m = modeDesktop
	}

	return options{
		action:         "up",
		mode:           m,
		engine:         os.Getenv("G4F_SANDBOX_ENGINE"),
		backend:        os.Getenv("G4F_SANDBOX_BACKEND"),
		port:           envInt("PORT", defaultPort),
		vncPort:        envInt("VNC_PORT", defaultVNCPort),
		bind:           envString("SANDBOX_BIND_ADDR", "127.0.0.1"),
		dataDir:        envString("G4F_SANDBOX_DATA_DIR", filepath.Join(home, ".g4f-sandbox")),
		appDir:         os.Getenv("G4F_APP_DIR"),
		apiKey:         os.Getenv("G4F_API_KEY"),
		domainSuffix:   envString("G4F_DOMAIN_SUFFIX", "g4f.local"),
		domainRouter:   envBool("G4F_DOMAIN_ROUTER", true),
		blockLocal:     envBool("SANDBOX_BLOCK_LOCAL_NETWORK", true),
		firewallStrict: envBool("SANDBOX_FIREWALL_STRICT", false),
		open:           !envBool("G4F_SANDBOX_NO_OPEN", false),
		detach:         true,
		userID:         envInt("G4F_USER_ID", 1000),
	}
}

func defaultImage(m sandboxMode) string {
	if v := os.Getenv("G4F_SANDBOX_IMAGE"); v != "" {
		return v
	}
	if m == modeDesktop {
		return defaultDesktopImage
	}
	return defaultAPIImage
}

func defaultName(m sandboxMode) string {
	if v := os.Getenv("G4F_SANDBOX_NAME"); v != "" {
		return v
	}
	if m == modeDesktop {
		return defaultDesktopName
	}
	return defaultAPIName
}

func envString(name, fallback string) string {
	if v, ok := os.LookupEnv(name); ok && v != "" {
		return v
	}
	return fallback
}

func envInt(name string, fallback int) int {
	v, ok := os.LookupEnv(name)
	if !ok || v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func envBool(name string, fallback bool) bool {
	v, ok := os.LookupEnv(name)
	if !ok || v == "" {
		return fallback
	}
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

// ---------------------------------------------------------------------------
// Container engine
// ---------------------------------------------------------------------------

type engine struct {
	bin string
}

func findEngine(preferred string) (engine, error) {
	candidates := []string{"docker", "podman"}
	if preferred != "" {
		candidates = []string{preferred}
	}
	var tried []string
	for _, name := range candidates {
		var path string
		var err error
		if name == "podman" {
			path, err = findPodmanBinary()
		} else {
			path, err = exec.LookPath(name)
		}
		if err == nil {
			return engine{bin: path}, nil
		}
		tried = append(tried, name)
	}
	return engine{}, fmt.Errorf("container engine not found (%s); install Docker Desktop or Podman (%s), or pass --engine",
		strings.Join(tried, ", "), "https://docs.docker.com/get-docker/")
}

// resolveEngine finds a usable container engine, installing Podman on a host
// that has neither Docker nor Podman, and verifies that its daemon works.
func resolveEngine(o options) (engine, error) {
	e, err := findEngine(o.engine)
	if err != nil {
		// No Docker and no Podman: install the Podman CLI so a fresh host can
		// run the sandbox without a manual setup step. An explicit
		// --engine docker is respected (no surprise Podman install).
		if o.engine == "" || o.engine == "podman" {
			if ierr := installPodman(); ierr == nil {
				e, err = findEngine("podman")
			} else {
				err = fmt.Errorf("%w; automatic Podman install failed: %v", err, ierr)
			}
		}
		if err != nil {
			return engine{}, err
		}
	}
	if err := e.checkReady(); err != nil {
		return engine{}, err
	}
	return e, nil
}

// checkReady distinguishes "CLI installed" from "daemon running", which is the
// most common failure on a fresh Windows/macOS install.
func (e engine) checkReady() error {
	if _, err := e.output("info"); err != nil {
		// A freshly installed Podman on macOS/Windows has no machine yet;
		// create and start the default one before giving up.
		if filepath.Base(e.bin) == "podman" && (runtime.GOOS == "darwin" || runtime.GOOS == "windows") {
			preparePodmanMachine()
			if _, retry := e.output("info"); retry == nil {
				return nil
			}
		}
		return fmt.Errorf("%s is installed but its daemon is not reachable; start Docker Desktop or the Docker/Podman service", filepath.Base(e.bin))
	}
	return nil
}

func (e engine) run(args ...string) error {
	cmd := exec.Command(e.bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", filepath.Base(e.bin), strings.Join(args, " "), err)
	}
	return nil
}

func (e engine) output(args ...string) (string, error) {
	cmd := exec.Command(e.bin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("%s %s: %s", filepath.Base(e.bin), strings.Join(args, " "), msg)
	}
	return strings.TrimSpace(stdout.String()), nil
}

func containerExists(e engine, name string) bool {
	_, err := e.output("inspect", "--format", "{{.Id}}", name)
	return err == nil
}

func containerRunning(e engine, name string) bool {
	out, err := e.output("inspect", "--format", "{{.State.Running}}", name)
	return err == nil && strings.EqualFold(out, "true")
}

func containerImage(e engine, name string) string {
	out, _ := e.output("inspect", "--format", "{{.Config.Image}}", name)
	return out
}

func imageExists(e engine, image string) bool {
	_, err := e.output("image", "inspect", image)
	return err == nil
}

func removeContainer(e engine, name string) error {
	if !containerExists(e, name) {
		return nil
	}
	return e.run("rm", "-f", name)
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

func runUp(e engine, o options) error {
	if o.buildFirst {
		if err := runBuild(e, o); err != nil {
			return err
		}
	}

	if o.mode == modeDesktop {
		return upDesktop(e, o)
	}
	return upAPI(e, o)
}

func upAPI(e engine, o options) error {
	if err := ensureDataDir(o); err != nil {
		return err
	}
	args := apiRunArgs(o)
	return upContainer(e, o, args)
}

func upDesktop(e engine, o options) error {
	if err := ensureDataDir(o); err != nil {
		return err
	}
	args := desktopRunArgs(o)
	return upContainer(e, o, args)
}

func upContainer(e engine, o options, runArgs []string) error {
	exists := containerExists(e, o.name)
	if exists && o.recreate {
		if err := removeContainer(e, o.name); err != nil {
			return err
		}
		exists = false
	}

	if exists {
		spec := runSpec(runArgs)
		currentImage := containerImage(e, o.name)
		currentSpec, _ := e.output("inspect", "--format", `{{index .Config.Labels "g4f.sandbox.spec"}}`, o.name)
		changedImage := currentImage != "" && currentImage != o.image
		changedSpec := currentSpec != "" && currentSpec != spec
		if changedImage || changedSpec {
			fmt.Fprintf(os.Stderr, "g4f-sandbox: container settings changed; recreating %s\n", o.name)
			if err := removeContainer(e, o.name); err != nil {
				return err
			}
			exists = false
		}
	}

	if !exists {
		if err := ensureImage(e, o, false); err != nil {
			return err
		}
		args := append([]string{"run", "-d", "--name", o.name}, runArgs...)
		if err := e.run(args...); err != nil {
			return err
		}
	} else {
		if o.alwaysPull {
			if err := ensureImage(e, o, true); err != nil {
				return err
			}
		}
		if !containerRunning(e, o.name) {
			if err := e.run("start", o.name); err != nil {
				return err
			}
		}
	}

	printBanner(o)
	if o.open {
		u := modeURL(o)
		if err := openBrowser(u); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox: could not open a browser:", err)
			fmt.Fprintln(os.Stderr, "g4f-sandbox: open this URL manually:", u)
		}
	}
	if !o.detach {
		return e.run("logs", "-f", o.name)
	}
	return nil
}

func ensureImage(e engine, o options, force bool) error {
	if o.noPull {
		if !imageExists(e, o.image) {
			return fmt.Errorf("image %s is not available locally and --no-pull was set", o.image)
		}
		return nil
	}
	if !force && imageExists(e, o.image) {
		return nil
	}
	platform := "linux/" + runtime.GOARCH
	if runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64" {
		platform = "the host architecture"
	}
	fmt.Fprintf(os.Stderr, "Pulling %s\n", o.image)
	fmt.Fprintf(os.Stderr, "Multi-arch image; %s downloads only the %s layers.\n", filepath.Base(e.bin), platform)
	if err := e.run("pull", o.image); err != nil {
		return fmt.Errorf("pull %s: %w", o.image, err)
	}
	return nil
}

func runDown(e engine, o options) error {
	if !containerExists(e, o.name) {
		fmt.Printf("container %s does not exist\n", o.name)
		return nil
	}
	if o.keep {
		if err := e.run("stop", o.name); err != nil {
			return err
		}
		fmt.Printf("stopped %s\n", o.name)
		return nil
	}
	if err := e.run("rm", "-f", o.name); err != nil {
		return err
	}
	fmt.Printf("removed %s\n", o.name)
	return nil
}

func runRestart(e engine, o options) error {
	if !containerExists(e, o.name) {
		return runUp(e, o)
	}
	if current := containerImage(e, o.name); current != "" && current != o.image {
		return runUp(e, o)
	}
	if err := e.run("restart", o.name); err != nil {
		return err
	}
	printBanner(o)
	return nil
}

func runStatus(e engine, o options) error {
	if !containerExists(e, o.name) {
		fmt.Printf("container %s does not exist\n", o.name)
		return nil
	}
	state, _ := e.output("inspect", "--format", "{{.State.Status}}", o.name)
	image := containerImage(e, o.name)
	fmt.Printf("container: %s\n", o.name)
	fmt.Printf("status:    %s\n", state)
	fmt.Printf("image:     %s\n", image)
	fmt.Printf("web:       %s\n", modeURL(o))
	if o.mode == modeDesktop {
		fmt.Printf("vnc:       %s:%d\n", displayHost(o.bind), o.vncPort)
	}
	return nil
}

func runLogs(e engine, o options) error {
	if !containerExists(e, o.name) {
		return fmt.Errorf("container %s does not exist; start it first with g4f-sandbox up", o.name)
	}
	return e.run("logs", "-f", o.name)
}

func runExec(e engine, o options, command []string) error {
	if len(command) == 0 {
		return fmt.Errorf("no command given; usage: g4f-sandbox exec <command> [args...]")
	}
	if !containerExists(e, o.name) {
		return fmt.Errorf("container %s does not exist; start it first with g4f-sandbox up", o.name)
	}
	args := append([]string{"exec", "-it", o.name}, command...)
	return e.run(args...)
}

func runCLI(e engine, o options) error {
	if o.mode != modeDesktop {
		return fmt.Errorf("sandbox-cli is only available in desktop mode; run: g4f-sandbox --desktop cli <args>")
	}
	return runExec(e, o, append([]string{"sandbox-cli"}, o.forwarded...))
}

// runInstall installs a sandbox-menu tool on demand. The desktop image ships
// the installer (sandbox-cli) but not the tools themselves, so nothing is
// downloaded until the user asks for it.
func runInstall(e engine, o options) error {
	if o.mode != modeDesktop {
		return fmt.Errorf("on-demand tools live in the desktop image; run: g4f-sandbox --desktop install <tool>")
	}
	if len(o.forwarded) == 0 {
		return fmt.Errorf("usage: g4f-sandbox --desktop install <tool> (see g4f-sandbox --desktop cli list)")
	}
	return runExec(e, o, append([]string{"sandbox-cli", "install"}, o.forwarded...))
}

func runBuild(e engine, o options) error {
	contextDir := o.context
	if contextDir == "" {
		contextDir = "."
	}
	absContext, err := filepath.Abs(contextDir)
	if err != nil {
		return err
	}
	dockerfile := o.dockerfile
	if dockerfile == "" {
		dockerfile = filepath.Join(absContext, "docker", "Dockerfile-sandbox-desktop")
	}
	if _, err := os.Stat(dockerfile); err != nil {
		return fmt.Errorf("desktop Dockerfile not found (%s); pass --context <checkout> or --dockerfile <file>", dockerfile)
	}
	tag := o.image
	if !o.imageSet {
		tag = localDesktopImage
	}
	fmt.Fprintf(os.Stderr, "Building %s from %s\n", tag, absContext)
	args := []string{
		"build",
		"--build-arg", fmt.Sprintf("G4F_USER_ID=%d", o.userID),
		"-f", dockerfile,
		"-t", tag,
		absContext,
	}
	if err := e.run(args...); err != nil {
		return err
	}
	fmt.Printf("built %s\n", tag)
	return nil
}

// ---------------------------------------------------------------------------
// Run arguments
// ---------------------------------------------------------------------------

func apiRunArgs(o options) []string {
	args := []string{
		"--init",
		"--label", "g4f.sandbox=1",
		"--label", "g4f.sandbox.mode=api",
		"-p", fmt.Sprintf("%s:%d:8080", o.bind, o.port),
		"-v", filepath.Join(o.dataDir, "har_and_cookies") + ":/app/har_and_cookies",
		"-v", filepath.Join(o.dataDir, "generated_media") + ":/app/generated_media",
	}
	if o.apiKey != "" {
		args = append(args, "-e", "G4F_API_KEY="+o.apiKey)
	}
	return withSpecLabel(args, o.image)
}

func desktopRunArgs(o options) []string {
	args := []string{
		"--hostname", "sandbox",
		"--init",
		"--shm-size", "2gb",
		"--label", "g4f.sandbox=1",
		"--label", "g4f.sandbox.mode=desktop",
		"-p", fmt.Sprintf("%s:%d:8080", o.bind, o.port),
		"-p", fmt.Sprintf("%s:%d:5900", o.bind, o.vncPort),
		"-v", filepath.Join(o.dataDir, "config") + ":/home/pn/.config",
		"-v", filepath.Join(o.dataDir, "local") + ":/home/pn/.local",
		"-e", "XDG_RUNTIME_DIR=/tmp/sandbox-runtime",
		"-e", "SANDBOX_DESKTOP=1",
		"-e", fmt.Sprintf("SANDBOX_BLOCK_LOCAL_NETWORK=%d", boolInt(o.blockLocal && !o.noFirewall)),
		"-e", fmt.Sprintf("SANDBOX_FIREWALL_STRICT=%d", boolInt(o.firewallStrict && !o.noFirewall)),
		"-e", "G4F_DOMAIN_SUFFIX=" + o.domainSuffix,
		"-e", fmt.Sprintf("G4F_DOMAIN_ROUTER=%d", boolInt(o.domainRouter)),
	}
	if !o.noFirewall {
		args = append(args, "--cap-add", "NET_ADMIN")
	}
	if o.appDir != "" {
		args = append(args, "-v", o.appDir+":/app/gpt4free")
	}
	if o.app != "" {
		args = append(args, "-e", "G4F_DESKTOP_APP="+o.app)
	}
	if o.apiKey != "" {
		args = append(args, "-e", "G4F_API_KEY="+o.apiKey)
	}
	return withSpecLabel(args, o.image)
}

// withSpecLabel fingerprints the run arguments so an existing container is
// recreated when the port, bind address, data directory or image changes.
// Docker options must come before the image, so the label is inserted there.
func withSpecLabel(args []string, image string) []string {
	spec := runSpec(append(append([]string{}, args...), image))
	return append(args, "--label", "g4f.sandbox.spec="+spec, image)
}

func runSpec(args []string) string {
	filtered := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		if args[i] == "--label" && i+1 < len(args) && strings.HasPrefix(args[i+1], "g4f.sandbox.spec=") {
			i++
			continue
		}
		filtered = append(filtered, args[i])
	}
	sum := sha256.Sum256([]byte(strings.Join(filtered, "\x00")))
	return hex.EncodeToString(sum[:8])
}

func ensureDataDir(o options) error {
	dirs := []string{o.dataDir}
	if o.mode == modeDesktop {
		dirs = append(dirs, filepath.Join(o.dataDir, "config"), filepath.Join(o.dataDir, "local"))
	} else {
		dirs = append(dirs, filepath.Join(o.dataDir, "har_and_cookies"), filepath.Join(o.dataDir, "generated_media"))
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create data directory %s: %w", dir, err)
		}
	}
	return nil
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func modeURL(o options) string {
	host := displayHost(o.bind)
	if o.mode == modeDesktop {
		return fmt.Sprintf("http://%s:%d/desktop/", host, o.port)
	}
	return fmt.Sprintf("http://%s:%d/", host, o.port)
}

func displayHost(bind string) string {
	if bind == "" || bind == "0.0.0.0" || bind == "::" {
		return "127.0.0.1"
	}
	return bind
}

func printBanner(o options) {
	fmt.Println()
	if o.mode == modeDesktop {
		fmt.Println("g4f sandbox desktop")
		fmt.Printf("  container: %s\n", o.name)
		fmt.Printf("  image:     %s\n", o.image)
		fmt.Printf("  desktop:   %s\n", modeURL(o))
		fmt.Printf("  api:       http://%s:%d/\n", displayHost(o.bind), o.port)
		fmt.Printf("  vnc:       %s:%d\n", displayHost(o.bind), o.vncPort)
		fmt.Printf("  data:      %s\n", o.dataDir)
		fmt.Printf("  tools:     g4f-sandbox --desktop install <tool>  (installed on demand)\n")
		if o.blockLocal && !o.noFirewall {
			fmt.Println("  egress:    private/link-local networks blocked")
		} else {
			fmt.Println("  egress:    local networks allowed")
		}
	} else {
		fmt.Println("g4f sandbox (API only)")
		fmt.Printf("  container: %s\n", o.name)
		fmt.Printf("  image:     %s\n", o.image)
		fmt.Printf("  api:       %s\n", modeURL(o))
		fmt.Printf("  data:      %s\n", o.dataDir)
		fmt.Println("  desktop:   not downloaded; use --desktop to pull the full image on demand")
	}
	if o.apiKey != "" {
		fmt.Println("  password:  G4F_API_KEY is set (any username works in the browser)")
	}
	fmt.Println()
}

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------

func openBrowser(u string) error {
	start := func(name string, args ...string) error {
		cmd := exec.Command(name, args...)
		if err := cmd.Start(); err != nil {
			return err
		}
		go func() { _ = cmd.Wait() }()
		return nil
	}
	if isTermux() {
		if err := start("termux-open-url", u); err == nil {
			return nil
		}
	}
	name, args := browserCommand(runtime.GOOS, u)
	return start(name, args...)
}

func browserCommand(goos, u string) (string, []string) {
	switch goos {
	case "windows":
		return "rundll32", []string{"url.dll,FileProtocolHandler", u}
	case "darwin":
		return "open", []string{u}
	default:
		return "xdg-open", []string{u}
	}
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

func printHelp() {
	fmt.Printf(`g4f-sandbox %s - launcher for the g4f sandbox image (g4f-go sandbox)

The launcher is under 1 MB compressed. It never embeds the image: Docker,
Podman or udocker pulls only the image for the selected mode and the host
architecture (about 400 MB for API-only, 675 MB for the desktop).

On Termux/Android it detects the environment and uses udocker, the user-space
container runner, so the same sandbox images work without root. udocker is
installed with pip on first use. The older native Termux Python backend is
still available with --backend termux (API only).

Usage:
  g4f-sandbox [COMMAND] [OPTIONS]

Commands:
  up, start       Start the sandbox (default)
  down, stop      Stop and remove the container (--keep only stops it)
  restart         Restart the container
  status          Show the container status
  logs            Follow the container logs
  shell, connect  Open a bash shell in the running container
  exec <cmd...>   Run a command in the container
  cli <args...>   Run sandbox-cli in a desktop container
  install <x...>  Desktop: install a menu tool; native Termux: pip install a package
  open            Open the web UI in a browser
  pull            Pull the image for the selected mode
  build           Build the desktop image from a source checkout
  version         Print the launcher version
  help            Show this help

Modes:
  --api-only      Small API-only image (default; no desktop stack)
  --desktop       Full Xfce + TigerVNC desktop image, pulled only when used
  --app TOOL      Desktop: start a sandbox-cli tool instead of the live wallpaper
  --image IMAGE   Override the image for the selected mode

Options:
  -p, --port PORT        API + noVNC port (default %d)
      --vnc-port PORT    Direct VNC port in desktop mode (default %d)
      --bind ADDR        Host address to publish on (default 127.0.0.1)
      --data-dir DIR     Persistent data directory (default ~/.g4f-sandbox)
      --app-dir DIR      Mount DIR at /app/gpt4free (desktop mode)
      --api-key KEY      Require KEY as the browser/API password
      --domain-suffix S  App hostname suffix (default g4f.local)
      --no-domain-router Disable <app>.<suffix> routing
      --allow-local-network  Let the sandbox reach private networks
      --no-firewall      Do not add NET_ADMIN or install the egress filter
      --firewall-strict  Fail startup when the egress filter cannot load
      --no-open          Do not launch a browser after start
  -d, --detach           Run in the background (default)
  -f, --foreground       Follow logs after start
      --recreate         Recreate an existing container
      --keep             With down/stop, stop without removing
      --pull             Pull the image even when it exists locally
      --no-pull          Never pull; fail when the image is missing
      --engine NAME      docker, podman or udocker (default: autodetect)
                         (when neither Docker nor Podman is installed, Podman
                          is installed with the host package manager; opt out
                          with G4F_SANDBOX_INSTALL_PODMAN=0)
      --backend NAME     auto, docker, podman, udocker or termux (default: auto)
      --udocker          Force the udocker backend (user-space containers)
      --termux           Force the native Termux Python backend (API only)
  -b, --build            Build the desktop image before starting
      --user-id UID      Sandbox user id for a build (default 1000)
      --context DIR      Source checkout for build (default .)
      --dockerfile FILE  Dockerfile for build

Environment:
  G4F_SANDBOX_MODE=api|desktop   G4F_SANDBOX_IMAGE   G4F_SANDBOX_NAME
  G4F_SANDBOX_DATA_DIR           G4F_SANDBOX_ENGINE  G4F_SANDBOX_NO_OPEN
  G4F_SANDBOX_BACKEND=auto|docker|podman|udocker|termux  G4F_SANDBOX_TERMUX=1
  G4F_SANDBOX_INSTALL_PODMAN=0 disables the automatic Podman install
  G4F_SANDBOX_INSTALL_UDOCKER=0 disables the automatic udocker (pip+tools) install
  G4F_SANDBOX_UDOCKER_DIR=/path     udocker root (default ~/.udocker)
  G4F_SANDBOX_UDOCKER_EXECMODE=P1   udocker mode (P1, P2, F1-F4, R1-R3, S1)
  PORT VNC_PORT SANDBOX_BIND_ADDR G4F_API_KEY G4F_APP_DIR G4F_DOMAIN_SUFFIX
  G4F_DOMAIN_ROUTER SANDBOX_BLOCK_LOCAL_NETWORK SANDBOX_FIREWALL_STRICT

Examples:
  g4f-sandbox                      # small API-only sandbox
  g4f-sandbox --desktop            # pull + start the full desktop on demand
  g4f-sandbox --desktop install chrome
  g4f-sandbox --desktop cli list
  g4f-sandbox logs

  # Termux: udocker is installed on demand, no root required
  g4f-sandbox                 # auto-detects Termux and uses udocker
  g4f-sandbox --desktop       # full desktop image under udocker
  g4f-sandbox --backend termux  # legacy native API backend
  g4f-sandbox down
`, Version, defaultPort, defaultVNCPort)
}
