package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestParseModuleFlag(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		wantModule string
		wantRest   []string
	}{
		{"absent", []string{"client", "hi"}, "", []string{"client", "hi"}},
		{"empty", nil, "", nil},
		{"short", []string{"-m", "remote_desktop"}, "remote_desktop", []string{}},
		{"short with args", []string{"-m", "remote_desktop", "--port", "8000"}, "remote_desktop", []string{"--port", "8000"}},
		{"long", []string{"--module", "remote_desktop"}, "remote_desktop", []string{}},
		{"long equals", []string{"--module=remote_desktop", "--port", "8000"}, "remote_desktop", []string{"--port", "8000"}},
		{"attached short", []string{"-mremote_desktop", "--port", "8000"}, "remote_desktop", []string{"--port", "8000"}},
		{"missing value", []string{"-m"}, "", []string{"-m"}},
		{"empty long equals", []string{"--module="}, "", []string{}},
		{"not first", []string{"client", "-m", "x"}, "", []string{"client", "-m", "x"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			module, rest := parseModuleFlag(tc.args)
			if module != tc.wantModule {
				t.Errorf("module = %q, want %q", module, tc.wantModule)
			}
			if !reflect.DeepEqual(rest, tc.wantRest) {
				t.Errorf("rest = %v, want %v", rest, tc.wantRest)
			}
		})
	}
}

func TestModuleFlagNeedsValue(t *testing.T) {
	tests := []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"-m"}, true},
		{[]string{"--module"}, true},
		{[]string{"--module="}, true},
		{[]string{"-m", "remote_desktop"}, false},
		{[]string{"--module=remote_desktop"}, false},
		{[]string{"client"}, false},
	}
	for _, tc := range tests {
		if got := moduleFlagNeedsValue(tc.args); got != tc.want {
			t.Errorf("moduleFlagNeedsValue(%v) = %v, want %v", tc.args, got, tc.want)
		}
	}
}

func TestBundledModulesIncludeRemoteDesktop(t *testing.T) {
	if !isBundledModule("remote_desktop") {
		t.Fatalf("remote_desktop should be bundled, got %v", bundledModules())
	}
	if isBundledModule("definitely_not_bundled") {
		t.Fatal("unexpected bundled module")
	}
}

func TestEnsureBundleExtractsPackageAndWebAssets(t *testing.T) {
	binDir := t.TempDir()

	dir, err := ensureBundle(binDir)
	if err != nil {
		t.Fatalf("ensureBundle() error: %v", err)
	}
	if dir != bundleDir(binDir) {
		t.Fatalf("bundle dir = %q, want %q", dir, bundleDir(binDir))
	}

	// The package entry point and its sibling web assets must both be present:
	// app.py resolves the web root as <package>/../web.
	for _, rel := range []string{
		filepath.Join("remote_desktop", "__main__.py"),
		filepath.Join("remote_desktop", "app.py"),
		filepath.Join("web", "host.html"),
		filepath.Join("web", "view.html"),
	} {
		if _, err := os.Stat(filepath.Join(dir, rel)); err != nil {
			t.Errorf("missing %s: %v", rel, err)
		}
	}

	// A second call must be a no-op that keeps the extracted files.
	if _, err := ensureBundle(binDir); err != nil {
		t.Fatalf("second ensureBundle() error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "remote_desktop", "__main__.py")); err != nil {
		t.Fatalf("bundle lost after re-run: %v", err)
	}
}

func TestEnsureBundleReextractsOnRevisionChange(t *testing.T) {
	binDir := t.TempDir()
	if _, err := ensureBundle(binDir); err != nil {
		t.Fatalf("ensureBundle() error: %v", err)
	}

	// Simulate a stale extraction from an older bundle revision.
	stale := filepath.Join(bundleDir(binDir), "remote_desktop", "stale.py")
	if err := os.WriteFile(stale, []byte("x"), 0o644); err != nil {
		t.Fatalf("write stale file: %v", err)
	}
	if err := os.WriteFile(bundleStampPath(binDir), []byte("0"), 0o644); err != nil {
		t.Fatalf("write stamp: %v", err)
	}

	if _, err := ensureBundle(binDir); err != nil {
		t.Fatalf("ensureBundle() after revision change: %v", err)
	}
	data, err := os.ReadFile(bundleStampPath(binDir))
	if err != nil {
		t.Fatalf("read stamp: %v", err)
	}
	if string(data) != BundleRevision {
		t.Fatalf("stamp = %q, want %q", data, BundleRevision)
	}
}

func TestBundleEnvPrependsBundleToPythonPath(t *testing.T) {
	binDir := t.TempDir()
	env := bundleEnv(binDir)

	var pythonPath string
	count := 0
	for _, kv := range env {
		if len(kv) > len("PYTHONPATH=") && kv[:len("PYTHONPATH=")] == "PYTHONPATH=" {
			pythonPath = kv[len("PYTHONPATH="):]
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly one PYTHONPATH entry, got %d (%v)", count, env)
	}
	want := bundleDir(binDir) + string(os.PathListSeparator)
	if len(pythonPath) < len(want) || pythonPath[:len(want)] != want {
		t.Fatalf("PYTHONPATH = %q, want prefix %q", pythonPath, want)
	}
}

func TestMergeEnvOverridesExistingKeys(t *testing.T) {
	base := []string{"PATH=/bin", "PYTHONPATH=/old", "HOME=/home/x"}
	got := mergeEnv(base, "PYTHONPATH=/new", "PYTHONUTF8=1")

	want := []string{"PATH=/bin", "HOME=/home/x", "PYTHONPATH=/new", "PYTHONUTF8=1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("mergeEnv() = %v, want %v", got, want)
	}
}
