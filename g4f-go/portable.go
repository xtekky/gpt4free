package main

import (
	"os"
	"path/filepath"
	)

// isTruthy reports whether v is a true-like configuration value.
// executableDir returns the directory containing the g4f-go executable.
func executableDir() string {
	exe, err := os.Executable()
	if err == nil && exe != "" {
		if dir := filepath.Dir(exe); dir != "" && dir != "." {
			return dir
		}
	}
	if len(os.Args) > 0 {
		if dir := filepath.Dir(os.Args[0]); dir != "" && dir != "." {
			return dir
		}
	}
	return "."
}

// isPortable reports whether g4f-go should run in portable mode.
// Portable mode stores all runtime data next to the executable instead
// of in the user's home/config directory.
func isPortable() bool {
	for _, k := range []string{"G4F_PORTABLE", "G4FGO_PORTABLE", "PORTABLE", "G4F_GO_PORTABLE"} {
		if v := os.Getenv(k); v != "" && isTruthy(v) {
			return true
		}
	}
	dir := executableDir()
	if dir == "" || dir == "." {
		return false
	}
	markers := []string{
		".portable",
		"portable",
		"portable.txt",
		".g4f-portable",
		"g4f-portable",
		"g4f.portable",
	}
	for _, m := range markers {
		if _, err := os.Stat(filepath.Join(dir, m)); err == nil {
			return true
		}
	}
	return false
}

// portableBaseDir returns the base directory for portable storage
// (config/runtime files) when running in portable mode.
func portableBaseDir() string {
	return filepath.Join(executableDir(), ".g4f")
}

// portableConfigDir returns the config directory to use in portable mode.
// This replaces ~/.g4f with the portable base.
func portableConfigDir() string {
	return portableBaseDir()
}

// portablePythonEmbedDir returns the python-embed directory for portable mode.
func portablePythonEmbedDir() string {
	return filepath.Join(portableBaseDir(), "python-embed")
}
