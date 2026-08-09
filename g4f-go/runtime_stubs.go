//go:build !android

package main

import "fmt"

// Stubs for the android-only hooks. The real implementations live in
// runtime_android.go (//go:build android); these keep non-android builds
// compiling and are never called on them.

func androidInstallDir() string { return "" }

func androidPythonExecutable(binDir string) (string, error) { return "", nil }
func androidPythonHome(binDir string) string                 { return "" }

func finalizeAndroidRuntime(binDir string) (string, error) {
	return "", fmt.Errorf("finalizeAndroidRuntime called on non-android build")
}
