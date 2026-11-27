//go:build android

package main

import (
	"fmt"
	"os"
	"path/filepath"
)

// Android builds of g4f-go are installed in Termux. The launcher uses the
// interpreter from Termux's `python` package ($PREFIX/bin/python3); no CPython
// archive is downloaded and no dlopen runner is compiled.

// androidInstallDir returns the writable directory used for g4f stamps and
// bundled-module caches. The launcher itself lives in $PREFIX/bin.
func androidInstallDir() string {
	if d := os.Getenv("G4F_ANDROID_FILES_DIR"); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "g4f-go-runtime")
}

// androidPythonExecutable is unused on Termux because ensureRuntime returns
// the system interpreter before any downloaded runtime is consulted.
func androidPythonExecutable(binDir string) (string, error) { return "", nil }

func androidPythonHome(binDir string) string { return "" }

// finalizeAndroidRuntime is unreachable now that Termux uses its own Python
// package; keep the hook so shared code still compiles for android.
func finalizeAndroidRuntime(binDir string) (string, error) {
	return "", fmt.Errorf("android builds use the Termux python package; no downloaded runtime")
}
