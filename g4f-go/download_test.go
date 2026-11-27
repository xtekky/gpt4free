package main

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestEnsureRuntimeUsesConfiguredPython(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses unix executable permissions")
	}
	py := filepath.Join(t.TempDir(), "python3")
	if err := os.WriteFile(py, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("G4F_PYTHON", py)

	got, err := ensureRuntime()
	if err != nil || got != py {
		t.Fatalf("ensureRuntime() = %q, %v; want configured python %q", got, err, py)
	}

	t.Setenv("G4F_PYTHON", filepath.Join(t.TempDir(), "missing"))
	if _, err := ensureRuntime(); err == nil {
		t.Fatal("expected an error for a missing G4F_PYTHON")
	}
}

func TestEnsureRuntimeUsesTermuxPython(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Termux uses unix paths")
	}
	prefix := t.TempDir()
	bin := filepath.Join(prefix, "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	py := filepath.Join(bin, "python3")
	if err := os.WriteFile(py, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TERMUX_VERSION", "1")
	t.Setenv("PREFIX", prefix)

	got, err := ensureRuntime()
	if err != nil || got != py {
		t.Fatalf("ensureRuntime() = %q, %v; want Termux python %q", got, err, py)
	}
}

func TestFindTermuxPythonUsesPrefix(t *testing.T) {
	prefix := t.TempDir()
	bin := filepath.Join(prefix, "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	py := filepath.Join(bin, "python3")
	if err := os.WriteFile(py, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, err := findTermuxPython(prefix)
	if err != nil || got != py {
		t.Fatalf("findTermuxPython() = %q, %v; want %q", got, err, py)
	}
}

func TestFindTermuxPythonRequiresExecutable(t *testing.T) {
	prefix := t.TempDir()
	bin := filepath.Join(prefix, "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bin, "python3"), []byte("not executable"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := findTermuxPython(prefix); err == nil {
		t.Fatal("expected an error when python3 is not executable")
	}
}

func TestRuntimeArchivesPreserveTopPrefixedNames(t *testing.T) {
	for _, format := range []string{"tar.gz", "zip"} {
		t.Run(format, func(t *testing.T) {
			dir := t.TempDir()
			archive := filepath.Join(dir, "runtime."+format)
			writeTestRuntimeArchive(t, archive, map[string]string{
				"./python/python.exe":  "interpreter",
				"./python/python3.dll": "library",
				"./python/Lib/os.py":   "module",
			})

			if err := extractRuntime(dir, archive); err != nil {
				t.Fatal(err)
			}
			for name, want := range map[string]string{
				"python.exe":  "interpreter",
				"python3.dll": "library",
				"Lib/os.py":   "module",
			} {
				got, err := os.ReadFile(filepath.Join(dir, "python-home", filepath.FromSlash(name)))
				if err != nil || string(got) != want {
					t.Errorf("%s: got %q, err %v; want %q", name, got, err, want)
				}
			}
		})
	}
}

func TestInstallDownloadedRuntimeRepairsStaleStamp(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("Android uses the Termux python package")
	}
	dir := t.TempDir()
	stamp := installedOkName(dir)
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stamp, []byte("ok"), 0o644); err != nil {
		t.Fatal(err)
	}
	archive := filepath.Join(dir, "runtime.tar.gz")
	writeTestRuntimeArchive(t, archive, map[string]string{"python/Lib/os.py": "module"})
	if _, err := installDownloadedRuntime(dir, archive); err == nil {
		t.Fatal("expected an error for an archive missing the interpreter")
	}
	if _, err := os.Stat(stamp); !os.IsNotExist(err) {
		t.Fatalf("failed installation left success stamp: %v", err)
	}

	name := "python/bin/python"
	want := filepath.Join(dir, "python-home", "bin", "python")
	if runtime.GOOS == "windows" {
		name = "python/python.exe"
		want = filepath.Join(dir, "python-home", "python.exe")
	}
	writeTestRuntimeArchive(t, archive, map[string]string{name: "interpreter"})
	got, err := installDownloadedRuntime(dir, archive)
	if err != nil || got != want {
		t.Fatalf("installDownloadedRuntime() = %q, %v; want %q", got, err, want)
	}
	if _, err := os.Stat(stamp); err != nil {
		t.Fatalf("successful installation did not write stamp: %v", err)
	}
}

func writeTestRuntimeArchive(t *testing.T, name string, files map[string]string) {
	t.Helper()
	var buf bytes.Buffer
	if filepath.Ext(name) == ".zip" {
		zw := zip.NewWriter(&buf)
		for path, content := range files {
			w, err := zw.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write([]byte(content)); err != nil {
				t.Fatal(err)
			}
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
	} else {
		gw := gzip.NewWriter(&buf)
		tw := tar.NewWriter(gw)
		for path, content := range files {
			if err := tw.WriteHeader(&tar.Header{Name: path, Mode: 0o644, Size: int64(len(content))}); err != nil {
				t.Fatal(err)
			}
			if _, err := tw.Write([]byte(content)); err != nil {
				t.Fatal(err)
			}
		}
		if err := tw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := gw.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(name, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
