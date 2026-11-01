package sandbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// clearEnv removes the launcher's environment inputs so tests are hermetic.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"G4F_SANDBOX_MODE", "G4F_SANDBOX_IMAGE", "G4F_SANDBOX_NAME",
		"G4F_SANDBOX_DATA_DIR", "G4F_SANDBOX_ENGINE", "G4F_SANDBOX_NO_OPEN",
		"G4F_SANDBOX_BACKEND", "G4F_SANDBOX_TERMUX", "G4F_SANDBOX_INSTALL_PODMAN",
		"G4F_SANDBOX_INSTALL_UDOCKER",
		"G4F_SANDBOX_UDOCKER_DIR", "G4F_SANDBOX_UDOCKER_EXECMODE",
		"TERMUX_VERSION", "PREFIX",
		"PORT", "VNC_PORT", "SANDBOX_BIND_ADDR", "G4F_API_KEY", "G4F_APP_DIR",
		"G4F_DOMAIN_SUFFIX", "G4F_DOMAIN_ROUTER", "SANDBOX_BLOCK_LOCAL_NETWORK",
		"SANDBOX_FIREWALL_STRICT", "G4F_USER_ID",
	} {
		t.Setenv(name, "")
	}
}

func TestParseDefaultsToSmallAPIImage(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs(nil)
	if err != nil {
		t.Fatal(err)
	}
	if o.action != "up" {
		t.Fatalf("action = %q, want up", o.action)
	}
	if o.mode != modeAPI {
		t.Fatalf("mode = %q, want api", o.mode)
	}
	if o.image != defaultAPIImage {
		t.Fatalf("image = %q, want %q", o.image, defaultAPIImage)
	}
	if o.name != defaultAPIName {
		t.Fatalf("name = %q, want %q", o.name, defaultAPIName)
	}
	if o.port != defaultPort || o.vncPort != defaultVNCPort {
		t.Fatalf("ports = %d/%d", o.port, o.vncPort)
	}
	if !o.open || !o.detach {
		t.Fatal("expected open and detach defaults")
	}
}

func TestParseDesktopModePullsDesktopImageOnDemand(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "--port", "9090", "--bind", "0.0.0.0", "--no-open"})
	if err != nil {
		t.Fatal(err)
	}
	if o.mode != modeDesktop {
		t.Fatalf("mode = %q", o.mode)
	}
	if o.image != defaultDesktopImage {
		t.Fatalf("image = %q", o.image)
	}
	if o.name != defaultDesktopName {
		t.Fatalf("name = %q", o.name)
	}
	if o.port != 9090 || o.bind != "0.0.0.0" {
		t.Fatalf("port/bind = %d/%q", o.port, o.bind)
	}
	if o.open {
		t.Fatal("--no-open was ignored")
	}
}

func TestParseCommandAfterFlags(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "status"})
	if err != nil {
		t.Fatal(err)
	}
	if o.action != "status" || o.mode != modeDesktop {
		t.Fatalf("action/mode = %q/%q", o.action, o.mode)
	}
}

func TestParsePassthroughCommands(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "cli", "list", "--verbose"})
	if err != nil {
		t.Fatal(err)
	}
	if o.action != "cli" {
		t.Fatalf("action = %q", o.action)
	}
	if got := strings.Join(o.forwarded, " "); got != "list --verbose" {
		t.Fatalf("forwarded = %q", got)
	}

	o, err = parseArgs([]string{"install", "--", "chrome"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(o.forwarded, " "); got != "chrome" {
		t.Fatalf("install forwarded = %q", got)
	}
}

func TestConnectAndExecDesktop(t *testing.T) {
	clearEnv(t)
	for _, test := range []struct {
		args      []string
		action    string
		forwarded string
	}{
		{[]string{"--desktop", "connect"}, "shell", ""},
		{[]string{"--desktop", "exec", "sh", "-c", "echo hello"}, "exec", "sh -c echo hello"},
	} {
		o, err := parseArgs(test.args)
		if err != nil {
			t.Fatal(err)
		}
		if o.mode != modeDesktop || o.name != defaultDesktopName || o.action != test.action || strings.Join(o.forwarded, " ") != test.forwarded {
			t.Fatalf("parseArgs(%v) = mode %q name %q action %q forwarded %v", test.args, o.mode, o.name, o.action, o.forwarded)
		}
	}
}

func TestDesktopApp(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"up", "--desktop", "--app", "opencode", "--no-open"})
	if err != nil {
		t.Fatal(err)
	}
	if o.app != "opencode" || o.name != defaultDesktopName || o.open {
		t.Fatalf("unexpected desktop options: %+v", o)
	}
	if args := strings.Join(desktopRunArgs(o), " "); !strings.Contains(args, "-e G4F_DESKTOP_APP=opencode") {
		t.Fatalf("desktop args missing app: %s", args)
	}
	withoutApp := o
	withoutApp.app = ""
	if runSpec(desktopRunArgs(o)) == runSpec(desktopRunArgs(withoutApp)) {
		t.Fatal("changing the startup app must recreate the container")
	}
	for _, args := range [][]string{
		{"--app", "opencode"},
		{"--desktop", "--app", ""},
		{"--desktop", "--app", "-bad"},
		{"--desktop", "--app", "foo;bar"},
	} {
		if _, err := parseArgs(args); err == nil {
			t.Errorf("parseArgs(%v) should reject invalid app", args)
		}
	}
}

func TestParseImageOverride(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "--image", "example/test:tag"})
	if err != nil {
		t.Fatal(err)
	}
	if o.image != "example/test:tag" {
		t.Fatalf("image = %q", o.image)
	}
	if !o.imageSet {
		t.Fatal("imageSet should be true")
	}
}

func TestBuildUsesLocalTagUnlessImageIsExplicit(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "--build"})
	if err != nil {
		t.Fatal(err)
	}
	if o.image != localDesktopImage {
		t.Fatalf("image = %q, want %q", o.image, localDesktopImage)
	}

	o, err = parseArgs([]string{"--desktop", "--build", "--image", "example/test:tag"})
	if err != nil {
		t.Fatal(err)
	}
	if o.image != "example/test:tag" {
		t.Fatalf("image = %q", o.image)
	}
}

func TestParseRejectsBadValues(t *testing.T) {
	clearEnv(t)
	for _, args := range [][]string{
		{"--port", "0"},
		{"--port", "70000"},
		{"--port", "not-a-number"},
		{"--desktop", "--port", "8080", "--vnc-port", "8080"},
		{"--bind", "::1"},
		{"--domain-suffix", "bad suffix"},
		{"--engine", "nerdctl"},
		{"--backend", "nope"},
		{"--termux", "--desktop"},
		{"--termux", "--engine", "docker"},
		{"--termux", "--engine", "udocker"},
		{"--backend", "udocker", "--engine", "docker"},
		{"--backend", "docker", "--engine", "udocker"},
		{"--build"},
		{"--unknown-flag"},
		{"unknown-command"},
	} {
		if _, err := parseArgs(args); err == nil {
			t.Fatalf("parseArgs(%v) succeeded, want error", args)
		}
	}
}

func TestAPIRunArgs(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--data-dir", "/tmp/g4f-sandbox-test", "--port", "8123", "--api-key", "secret"})
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(apiRunArgs(o), " ")
	for _, want := range []string{
		"-p 127.0.0.1:8123:8080",
		"/tmp/g4f-sandbox-test/har_and_cookies:/app/har_and_cookies",
		"/tmp/g4f-sandbox-test/generated_media:/app/generated_media",
		"-e G4F_API_KEY=secret",
		defaultAPIImage,
	} {
		if !strings.Contains(args, want) {
			t.Errorf("api args missing %q:\n%s", want, args)
		}
	}
	if strings.Contains(args, "--cap-add") || strings.Contains(args, "--shm-size") {
		t.Errorf("API-only args should not include desktop capabilities:\n%s", args)
	}
}

func TestDesktopRunArgs(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{
		"--desktop", "--data-dir", "/tmp/g4f-sandbox-test",
		"--port", "8123", "--vnc-port", "5923", "--api-key", "secret",
		"--no-domain-router",
	})
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(desktopRunArgs(o), " ")
	for _, want := range []string{
		"--cap-add NET_ADMIN",
		"--shm-size 2gb",
		"-p 127.0.0.1:8123:8080",
		"-p 127.0.0.1:5923:5900",
		"/tmp/g4f-sandbox-test/config:/home/pn/.config",
		"/tmp/g4f-sandbox-test/local:/home/pn/.local",
		"SANDBOX_BLOCK_LOCAL_NETWORK=1",
		"G4F_DOMAIN_ROUTER=0",
		"-e G4F_API_KEY=secret",
		defaultDesktopImage,
	} {
		if !strings.Contains(args, want) {
			t.Errorf("desktop args missing %q:\n%s", want, args)
		}
	}
}

func TestRunSpecLabelComesBeforeImageAndTracksChanges(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--data-dir", "/tmp/g4f-sandbox-test"})
	if err != nil {
		t.Fatal(err)
	}
	args := apiRunArgs(o)
	imageIndex, labelIndex := -1, -1
	for i, arg := range args {
		if arg == o.image {
			imageIndex = i
		}
		if arg == "--label" && i+1 < len(args) && strings.HasPrefix(args[i+1], "g4f.sandbox.spec=") {
			labelIndex = i
		}
	}
	if imageIndex < 0 || labelIndex < 0 {
		t.Fatalf("missing image or spec label in %v", args)
	}
	if labelIndex > imageIndex {
		t.Fatalf("docker option after image: %v", args)
	}

	o2 := o
	o2.port = o.port + 1
	if runSpec(apiRunArgs(o)) == runSpec(apiRunArgs(o2)) {
		t.Fatal("spec did not change with the port")
	}
}

func TestDesktopNoFirewallSkipsCapability(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--desktop", "--no-firewall"})
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(desktopRunArgs(o), " ")
	if strings.Contains(args, "--cap-add") {
		t.Errorf("--no-firewall should skip NET_ADMIN:\n%s", args)
	}
	if !strings.Contains(args, "SANDBOX_BLOCK_LOCAL_NETWORK=0") {
		t.Errorf("--no-firewall should allow local networks:\n%s", args)
	}
}

func TestModeURL(t *testing.T) {
	clearEnv(t)
	o, _ := parseArgs(nil)
	if got := modeURL(o); got != "http://127.0.0.1:8080/" {
		t.Fatalf("api url = %q", got)
	}
	o, _ = parseArgs([]string{"--desktop", "--bind", "0.0.0.0", "--port", "9000"})
	if got := modeURL(o); got != "http://127.0.0.1:9000/desktop/" {
		t.Fatalf("desktop url = %q", got)
	}
}

func TestTermuxDetection(t *testing.T) {
	clearEnv(t)
	if isTermux() {
		t.Fatal("isTermux() should be false in a clean environment")
	}

	t.Setenv("TERMUX_VERSION", "0.118.0")
	if !isTermux() {
		t.Fatal("TERMUX_VERSION was not detected")
	}

	clearEnv(t)
	t.Setenv("PREFIX", "/data/data/com.termux/files/usr")
	if !isTermux() {
		t.Fatal("Termux PREFIX was not detected")
	}
}

func TestUseTermuxBackend(t *testing.T) {
	clearEnv(t)
	t.Setenv("TERMUX_VERSION", "0.118.0")

	o, err := parseArgs(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !useUdocker(o) {
		t.Fatal("Termux environment should use the udocker backend")
	}
	if useTermux(o) {
		t.Fatal("Termux autodetection must not select the native backend")
	}

	o, err = parseArgs([]string{"--engine", "docker"})
	if err != nil {
		t.Fatal(err)
	}
	if useUdocker(o) || useTermux(o) {
		t.Fatal("explicit --engine docker should win over Termux autodetection")
	}

	o, err = parseArgs([]string{"--backend", "termux"})
	if err != nil {
		t.Fatal(err)
	}
	if !useTermux(o) || useUdocker(o) {
		t.Fatal("--backend termux was not honored")
	}

	o, err = parseArgs([]string{"--backend", "udocker"})
	if err != nil {
		t.Fatal(err)
	}
	if !useUdocker(o) || useTermux(o) {
		t.Fatal("--backend udocker was not honored")
	}

	o, err = parseArgs([]string{"--engine", "udocker"})
	if err != nil {
		t.Fatal(err)
	}
	if !useUdocker(o) || o.engine != "" {
		t.Fatalf("--engine udocker should select the udocker backend, got backend=%q engine=%q", o.backend, o.engine)
	}

	o, err = parseArgs([]string{"--backend", "docker"})
	if err != nil {
		t.Fatal(err)
	}
	if useUdocker(o) || useTermux(o) || o.engine != "docker" {
		t.Fatalf("--backend docker should select Docker, got backend=%q engine=%q", o.backend, o.engine)
	}
}

func TestUdockerRunArgs(t *testing.T) {
	clearEnv(t)

	o, err := parseArgs([]string{"--udocker", "--data-dir", "/tmp/g4f-udocker-test", "--port", "8123", "--api-key", "secret"})
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(udockerRunArgs(o, nil), " ")
	for _, want := range []string{
		"run",
		"--publish=8123:8080",
		"--volume=/tmp/g4f-udocker-test/har_and_cookies:/app/har_and_cookies",
		"--volume=/tmp/g4f-udocker-test/generated_media:/app/generated_media",
		"--env=G4F_API_KEY=secret",
		defaultAPIName,
	} {
		if !strings.Contains(args, want) {
			t.Errorf("udocker api args missing %q:\n%s", want, args)
		}
	}

	o, err = parseArgs([]string{
		"--desktop", "--udocker", "--data-dir", "/tmp/g4f-udocker-test",
		"--port", "8123", "--vnc-port", "5923", "--app", "opencode",
	})
	if err != nil {
		t.Fatal(err)
	}
	args = strings.Join(udockerExecArgs(o, []string{"sandbox-cli", "list"}), " ")
	for _, want := range []string{
		"--publish=8123:8080",
		"--publish=5923:5900",
		"--volume=/tmp/g4f-udocker-test/config:/home/pn/.config",
		"--env=SANDBOX_DESKTOP=1",
		"--env=G4F_DESKTOP_APP=opencode",
		"--entrypoint=",
		defaultDesktopName,
		"sandbox-cli list",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("udocker desktop args missing %q:\n%s", want, args)
		}
	}
	if strings.Contains(args, "--cap-add") || strings.Contains(args, "--shm-size") {
		t.Errorf("udocker args must not carry unsupported Docker options:\n%s", args)
	}

	o2 := o
	o2.port++
	if udockerSpec(o) == udockerSpec(o2) {
		t.Fatal("udocker spec did not change with the port")
	}
}

func TestUdockerConfig(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--udocker", "--data-dir", "/tmp/g4f-udocker-test"})
	if err != nil {
		t.Fatal(err)
	}

	// No configuration: no setup step and no UDOCKER_DIR override.
	if args, err := udockerSetupArgs(o); err != nil || args != nil {
		t.Fatalf("udockerSetupArgs() = %v, %v; want nil, nil", args, err)
	}
	for _, entry := range udockerEnv() {
		if strings.HasPrefix(entry, "UDOCKER_DIR=") {
			t.Fatalf("unexpected UDOCKER_DIR in env: %s", entry)
		}
	}

	baseSpec := udockerSpec(o)

	t.Setenv("G4F_SANDBOX_UDOCKER_DIR", "/tmp/udocker-root")
	t.Setenv("G4F_SANDBOX_UDOCKER_EXECMODE", "p2")
	args, err := udockerSetupArgs(o)
	if err != nil {
		t.Fatal(err)
	}
	if udockerSpec(o) == baseSpec {
		t.Fatal("udocker spec did not change with the udocker configuration")
	}
	if got := strings.Join(args, " "); got != "setup --execmode=P2 "+defaultAPIName {
		t.Fatalf("setup args = %q", got)
	}
	if got := udockerToolsDir(); got != "/tmp/udocker-root" {
		t.Fatalf("udockerToolsDir() = %q", got)
	}
	env := strings.Join(udockerEnv(), "\n")
	if !strings.Contains(env, "UDOCKER_DIR=/tmp/udocker-root") {
		t.Fatalf("env missing UDOCKER_DIR:\n%s", env)
	}
	count := 0
	for _, entry := range strings.Split(env, "\n") {
		if strings.HasPrefix(entry, "UDOCKER_DIR=") {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("UDOCKER_DIR should appear once:\n%s", env)
	}

	t.Setenv("G4F_SANDBOX_UDOCKER_EXECMODE", "X9")
	if _, err := udockerSetupArgs(o); err == nil {
		t.Fatal("invalid execution mode was accepted")
	}
}

func TestUdockerToolsInstalled(t *testing.T) {
	clearEnv(t)
	root := t.TempDir()
	t.Setenv("G4F_SANDBOX_UDOCKER_DIR", root)
	if udockerToolsInstalled() {
		t.Fatal("empty udocker root reported as installed")
	}
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	if udockerToolsInstalled() {
		t.Fatal("empty bin directory reported as installed")
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "proot"), []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	if !udockerToolsInstalled() {
		t.Fatal("udocker tools not detected")
	}
}

func TestTermuxServerArgsAndEnv(t *testing.T) {
	clearEnv(t)
	o, err := parseArgs([]string{"--port", "9099", "--api-key", "secret"})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(termuxServerArgs(o), " "); got != "-m g4f --port 9099 --bind 127.0.0.1:9099" {
		t.Fatalf("server args = %q", got)
	}
	env := strings.Join(termuxEnv(o), "\n")
	if !strings.Contains(env, "PYTHONUNBUFFERED=1") || !strings.Contains(env, "G4F_API_KEY=secret") {
		t.Fatalf("termux env missing expected values:\n%s", env)
	}
}

func TestBrowserCommand(t *testing.T) {
	if name, _ := browserCommand("windows", "http://x"); name != "rundll32" {
		t.Fatalf("windows browser = %q", name)
	}
	if name, _ := browserCommand("darwin", "http://x"); name != "open" {
		t.Fatalf("darwin browser = %q", name)
	}
	if name, _ := browserCommand("linux", "http://x"); name != "xdg-open" {
		t.Fatalf("linux browser = %q", name)
	}
}
