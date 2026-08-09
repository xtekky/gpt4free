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
// embedded python. Written next to the binary during extraction.
const pythonLauncher = `#!/bin/sh
# g4f-go launcher for the embedded CPython runtime.
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$DIR/python" ]; then PY="$DIR/python"; else PY="$DIR/python-home/bin/python"; fi
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

// pythonExecutable returns the launcher (unix) or python.exe (windows) path.
//
// The embedded archive is laid out as <name>/python-home/<exe>, so after
// extraction the interpreter lives at binDir/python-home/python.exe on
// windows and binDir/python-home/bin/python on unix. We prefer that location
// and fall back to the legacy binDir/python(.exe) layout produced by older
// archives, mirroring the shell launcher's logic.
func pythonExecutable(binDir string) string {
	home := filepath.Join(binDir, "python-home")
	if runtime.GOOS == "windows" {
		return filepath.Join(home, "python.exe")
	}
	exe := filepath.Join(home, "bin", "python")
	if fi, err := os.Stat(exe); err == nil && !fi.IsDir() {
		return exe
	}
	return filepath.Join(binDir, "python")
}

// pythonHome returns the extracted interpreter root.
func pythonHome(binDir string) string {
	return filepath.Join(binDir, "python-home")
}

// writeLauncher installs the unix shell wrapper after extraction.
func writeLauncher(binDir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	path := pythonExecutable(binDir)
	if err := os.WriteFile(path, []byte(pythonLauncher), 0o755); err != nil {
		return err
	}
	return nil
}

// ensureDir is a tiny helper for CLI commands.
func ensureDir(p string) error { return os.MkdirAll(p, 0o755) }
