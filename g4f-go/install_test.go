package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLauncherSelfInstall(t *testing.T) {
	tests := []struct {
		args []string
		want bool
	}{
		{nil, true},
		{[]string{"help"}, true},
		{[]string{"--help"}, true},
		{[]string{"--prefix", "/tmp/bin"}, true},
		{[]string{"--prefix=/tmp/bin"}, true},
		{[]string{"--dry-run"}, true},
		{[]string{"g4f"}, false},
		{[]string{"g4f[slim]"}, false},
		{[]string{"g4f==1.5.0"}, false},
		{[]string{"-e", "."}, false},
		{[]string{"--editable", "."}, false},
	}
	for _, tt := range tests {
		if got := launcherSelfInstall(tt.args); got != tt.want {
			t.Errorf("launcherSelfInstall(%q) = %v, want %v", tt.args, got, tt.want)
		}
	}
}

func TestParseInstallOptions(t *testing.T) {
	opts, err := parseInstallOptions([]string{"--prefix", "/opt/bin", "--dry-run"})
	if err != nil {
		t.Fatalf("parseInstallOptions: %v", err)
	}
	if opts.Prefix != "/opt/bin" || !opts.DryRun || opts.Help {
		t.Fatalf("parseInstallOptions() = %+v", opts)
	}

	opts, err = parseInstallOptions([]string{"--prefix=/opt/bin"})
	if err != nil {
		t.Fatalf("parseInstallOptions(--prefix=): %v", err)
	}
	if opts.Prefix != "/opt/bin" {
		t.Fatalf("prefix = %q, want /opt/bin", opts.Prefix)
	}

	opts, err = parseInstallOptions([]string{"help"})
	if err != nil || !opts.Help {
		t.Fatalf("parseInstallOptions(help) = (%+v, %v), want Help", opts, err)
	}

	if _, err := parseInstallOptions([]string{"--prefix"}); err == nil {
		t.Fatal("parseInstallOptions(--prefix) expected an error")
	}
	if _, err := parseInstallOptions([]string{"--what"}); err == nil {
		t.Fatal("parseInstallOptions(--what) expected an error")
	}
}

func TestInstallBinDirUsesPrefixEnv(t *testing.T) {
	want := filepath.Join(t.TempDir(), "bin")
	t.Setenv("G4F_GO_PREFIX", want)
	got, err := installBinDir()
	if err != nil {
		t.Fatalf("installBinDir: %v", err)
	}
	if got != want {
		t.Fatalf("installBinDir() = %q, want %q", got, want)
	}
}

func TestInstallBinDirUsesFirstWritablePathEntry(t *testing.T) {
	t.Setenv("G4F_GO_PREFIX", "")
	first := filepath.Join(t.TempDir(), "first")
	second := filepath.Join(t.TempDir(), "second")
	for _, dir := range []string{first, second} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", first+string(os.PathListSeparator)+second)

	got, err := installBinDir()
	if err != nil {
		t.Fatalf("installBinDir: %v", err)
	}
	if got != first {
		t.Fatalf("installBinDir() = %q, want %q", got, first)
	}
	if !pathContainsDir(first) {
		t.Fatalf("pathContainsDir(%q) = false", first)
	}
	if pathContainsDir(filepath.Join(first, "missing")) {
		t.Fatal("pathContainsDir() matched a directory that is not on PATH")
	}
}

func TestInstallBinDirCreatesMissingHomeBinOnPath(t *testing.T) {
	t.Setenv("G4F_GO_PREFIX", "")
	home := t.TempDir()
	t.Setenv("HOME", home)
	want := filepath.Join(home, ".local", "bin")
	t.Setenv("PATH", want+string(os.PathListSeparator)+"/usr/bin"+string(os.PathListSeparator)+"/bin")

	got, err := installBinDir()
	if err != nil {
		t.Fatalf("installBinDir: %v", err)
	}
	if got != want {
		t.Fatalf("installBinDir() = %q, want %q", got, want)
	}
	if info, err := os.Stat(want); err != nil || !info.IsDir() {
		t.Fatalf("~/.local/bin was not created: %v", err)
	}
}

func TestInstallExecutable(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "source")
	if err := os.WriteFile(src, []byte("#!/bin/sh\necho hi\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(dir, "bin", "g4f-go")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := installExecutable(src, target); err != nil {
		t.Fatalf("installExecutable: %v", err)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "#!/bin/sh\necho hi\n" {
		t.Fatalf("installed content = %q", data)
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("installed mode = %v, want executable", info.Mode())
	}
	if !samePath(target, target) {
		t.Fatal("samePath() did not recognise the target itself")
	}
	if samePath(src, target) {
		t.Fatal("samePath() treated two different files as the same")
	}
}
