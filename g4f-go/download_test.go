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
		t.Skip("Android builds a separate runner")
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
		for name, contents := range files {
			w, err := zw.Create(name)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write([]byte(contents)); err != nil {
				t.Fatal(err)
			}
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
	} else {
		gz := gzip.NewWriter(&buf)
		tw := tar.NewWriter(gz)
		for name, contents := range files {
			if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(contents))}); err != nil {
				t.Fatal(err)
			}
			if _, err := tw.Write([]byte(contents)); err != nil {
				t.Fatal(err)
			}
		}
		if err := tw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := gz.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(name, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
