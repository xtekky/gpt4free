package freeflow

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    Options
		wantErr bool
	}{
		{name: "default serve", args: nil, want: Options{Action: "serve", Port: DefaultPort}},
		{name: "explicit serve", args: []string{"serve"}, want: Options{Action: "serve", Port: DefaultPort}},
		{name: "start alias", args: []string{"start"}, want: Options{Action: "serve", Port: DefaultPort}},
		{name: "install", args: []string{"install"}, want: Options{Action: "install", Port: DefaultPort}},
		{name: "install force", args: []string{"install", "--force"}, want: Options{Action: "install", Port: DefaultPort, Force: true}},
		{name: "status", args: []string{"status"}, want: Options{Action: "status", Port: DefaultPort}},
		{name: "port flag", args: []string{"serve", "--port", "9000"}, want: Options{Action: "serve", Port: 9000, PortSet: true}},
		{name: "port equals", args: []string{"--port=9000"}, want: Options{Action: "serve", Port: 9000, PortSet: true}},
		{name: "help", args: []string{"help"}, want: Options{Action: "help", Port: DefaultPort}},
		{name: "bad port", args: []string{"--port", "0"}, wantErr: true},
		{name: "missing port", args: []string{"--port"}, wantErr: true},
		{name: "unknown flag", args: []string{"--wat"}, wantErr: true},
		{name: "unknown command", args: []string{"wat"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Parse(tt.args)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Parse(%q) expected an error, got %+v", tt.args, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Parse(%q): %v", tt.args, err)
			}
			if got != tt.want {
				t.Fatalf("Parse(%q) = %+v, want %+v", tt.args, got, tt.want)
			}
		})
	}
}

func TestNodeVersionSupported(t *testing.T) {
	for _, version := range []string{"v22.19.0", "22.19.0", "v24.0.1", "v23.0.0"} {
		if !nodeVersionSupported(version) {
			t.Errorf("nodeVersionSupported(%q) = false, want true", version)
		}
	}
	for _, version := range []string{"", "v22.18.9", "v20.11.1", "v18.20.0", "nope"} {
		if nodeVersionSupported(version) {
			t.Errorf("nodeVersionSupported(%q) = true, want false", version)
		}
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFindPackageAndStage(t *testing.T) {
	home := t.TempDir()
	t.Setenv("G4F_FREEFLOW_HOME", home)

	src := filepath.Join(home, "node_modules", Plugin)
	writeFile(t, filepath.Join(src, "package.json"), `{"version":"1.43.0"}`)
	writeFile(t, filepath.Join(src, "src", "daemon.ts"), "// daemon")
	writeFile(t, filepath.Join(filepath.Dir(src), "undici", "index.js"), "// undici")

	got, err := findPackage()
	if err != nil {
		t.Fatalf("findPackage(): %v", err)
	}
	if got != src {
		t.Fatalf("findPackage() = %q, want %q", got, src)
	}

	daemonDir, daemon, err := stageDaemon(src)
	if err != nil {
		t.Fatalf("stageDaemon(): %v", err)
	}
	if daemon != filepath.Join(daemonDir, "src", "daemon.ts") {
		t.Fatalf("daemon path = %q", daemon)
	}
	for _, path := range []string{
		daemon,
		filepath.Join(daemonDir, "node_modules", "undici", "index.js"),
		filepath.Join(daemonDir, ".version"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected %s after staging: %v", path, err)
		}
	}

	// A second call reuses the staged copy instead of copying again.
	writeFile(t, filepath.Join(daemonDir, "marker"), "kept")
	again, _, err := stageDaemon(src)
	if err != nil {
		t.Fatalf("stageDaemon() second call: %v", err)
	}
	if again != daemonDir {
		t.Fatalf("stageDaemon() second call = %q, want %q", again, daemonDir)
	}
	if _, err := os.Stat(filepath.Join(daemonDir, "marker")); err != nil {
		t.Fatalf("staged copy was replaced without a version change: %v", err)
	}
}

func TestDataDirAndTokenMirrorPiFreeflow(t *testing.T) {
	t.Setenv("PI_FREEFLOW_DATA_DIR", "")
	t.Setenv("PI_CODING_AGENT_DIR", "")

	agent := filepath.Join(t.TempDir(), "agent")
	t.Setenv("PI_CODING_AGENT_DIR", agent)
	if got := dataDir(); got != agent {
		t.Fatalf("dataDir() = %q, want %q", got, agent)
	}

	override := filepath.Join(t.TempDir(), "data")
	t.Setenv("PI_FREEFLOW_DATA_DIR", override)
	if got := dataDir(); got != override {
		t.Fatalf("dataDir() with override = %q, want %q", got, override)
	}
	if got := tokenPath(); got != filepath.Join(override, "token") {
		t.Fatalf("tokenPath() = %q", got)
	}

	writeFile(t, tokenPath(), "secret\n")
	if got := readToken(); got != "secret" {
		t.Fatalf("readToken() = %q, want secret", got)
	}
}

func TestEnvPortFallsBackToDefault(t *testing.T) {
	t.Setenv("PI_FREEFLOW_PORT", "9000")
	if got := envPort(); got != 9000 {
		t.Fatalf("envPort() = %d, want 9000", got)
	}
	t.Setenv("PI_FREEFLOW_PORT", "not-a-port")
	if got := envPort(); got != DefaultPort {
		t.Fatalf("envPort() with invalid value = %d, want %d", got, DefaultPort)
	}
}
