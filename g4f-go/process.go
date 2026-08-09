package main

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// pythonLauncher is the shell wrapper that sets PYTHONHOME etc. and execs the
// downloaded python. Written next to the binary after the runtime is ready.
// Note: it must never exec itself ($DIR/python is the wrapper); the real
// interpreter always lives at $DIR/python-home/bin/python on unix.
const pythonLauncher = `#!/bin/sh
# g4f-go launcher for the downloaded CPython runtime.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$DIR/python-home/bin/python"
export PYTHONHOME="$DIR/python-home"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
exec "$PY" "$@"
`

// runPython executes the embedded interpreter with `-m g4f` args and forwards
// stdin/stdout/stderr. The pip bootstrap passes different env via pipEnv().
func runPython(ctx context.Context, exe string, args []string, extraEnv ...string) (int, error) {
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), extraEnv...)
	err := cmd.Run()
	if err != nil {
		if ctx.Err() != nil {
			return 130, nil // interrupted (Ctrl-C)
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), nil
		}
		return 1, err
	}
	return 0, nil
}

// noSignalCtx returns a background context for internal subprocesses (pip)
// which must not be killed by Ctrl-C handling.
func noSignalCtx() context.Context { return context.Background() }

// extractZip unpacks a runtime archive into dest with zip-slip protection.
// The archive has a single top-level directory (e.g. "linux-x64/"); we strip it
// so the runtime lands directly in dest, matching the launcher layout.
func extractZip(r io.ReaderAt, size int64, dest string) error {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return err
	}
	var top string
	for _, f := range zr.File {
		parts := strings.Split(filepath.Clean(f.Name), string(os.PathSeparator))
		if len(parts) > 0 && parts[0] != "." && top == "" {
			top = parts[0]
		}
	}
	if top == "" || top == "." {
		return fmt.Errorf("archive has no top-level directory")
	}

	for _, f := range zr.File {
		rel := f.Name
		if top != "" {
			rel = strings.TrimPrefix(f.Name, top+"/")
			rel = strings.TrimPrefix(rel, top)
		}
		rel = strings.TrimPrefix(rel, "/")
		name := filepath.Clean(rel)
		if name == "." || name == "" {
			continue // skip the top dir itself
		}
		if name == ".." || strings.HasPrefix(name, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe path in archive: %s", f.Name)
		}
		target := filepath.Join(dest, name)
		if !strings.HasPrefix(target, filepath.Clean(dest)+string(os.PathSeparator)) && target != filepath.Clean(dest) {
			return fmt.Errorf("unsafe path in archive: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode())
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			return err
		}
		if err := out.Close(); err != nil {
			rc.Close()
			return err
		}
		rc.Close()
		if err := os.Chmod(target, f.Mode()); err != nil {
			return err
		}
	}
	return nil
}

// pythonExecutable returns the python binary (or launcher) path for a
// downloaded runtime, or an error when no interpreter is present yet.
//
// pbs installs land in binDir/python-home/ with bin/python (unix) or
// python.exe (windows). The android build overrides this with a C runner
// that dlopens libpython (see runtime_android.go). Note that binDir/python
// is the *shell wrapper* (never a real interpreter), so it is not a valid
// fallback here.
func pythonExecutable(binDir string) (string, error) {
	if exe, err := androidPythonExecutable(binDir); exe != "" || err != nil {
		return exe, err
	}
	home := filepath.Join(binDir, "python-home")
	var candidates []string
	if runtime.GOOS == "windows" {
		candidates = []string{filepath.Join(home, "python.exe")}
	} else {
		candidates = []string{filepath.Join(home, "bin", "python")}
	}
	for _, exe := range candidates {
		if fi, err := os.Stat(exe); err == nil && !fi.IsDir() {
			return exe, nil
		}
	}
	return "", fmt.Errorf("no python executable found (runtime not downloaded?)")
}

// pythonHome returns the downloaded interpreter root.
func pythonHome(binDir string) string {
	if h := androidPythonHome(binDir); h != "" {
		return h
	}
	return filepath.Join(binDir, "python-home")
}

// writeLauncher installs the unix shell wrapper after the runtime is ready.
func writeLauncher(binDir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	path := filepath.Join(binDir, "python")
	if err := os.WriteFile(path, []byte(pythonLauncher), 0o755); err != nil {
		return err
	}
	return nil
}

// ensureDir is a tiny helper for CLI commands.
func ensureDir(p string) error { return os.MkdirAll(p, 0o755) }
