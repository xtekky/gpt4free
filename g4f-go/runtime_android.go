//go:build android

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// ---------------------------------------------------------------------------
// Android runtime
//
// The python.org Android package ships libpython3.14.so + the stdlib but no
// `python` executable — it is meant to be loaded with dlopen from a native
// app. g4f-go therefore:
//  1. checks Termux is installed (the `pm list packages` probe below);
//  2. merges the android tarball into the app's writable dir
//     (Android filesDir), landing at <dir>/python-home/;
//  3. builds a tiny C runner (pyandroid_runner.c) with the NDK clang via
//     Termux, dlopens libpython3.14.so and runs the requested script —
//     the same technique CPython's own android.py testbed uses.
//
// The launcher path returned to the caller is that C runner binary.
// ---------------------------------------------------------------------------

// androidInstallDir returns the app-private writable directory (filesDir).
// The android build sets G4F_ANDROID_FILES_DIR via a custom linker or wrapper;
// falling back to the app's home dir keeps plain `go build` usable.
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

// IsTermuxInstalled reports whether the Termux app is present, using the
// Android package manager. Only valid on Android.
func IsTermuxInstalled() (bool, error) {
	// Only valid on Android
	if runtime.GOOS != "android" {
		return false, fmt.Errorf("not running on Android")
	}
	// Run pm list packages to get all installed user packages
	cmd := exec.Command("pm", "list", "packages")
	output, err := cmd.Output()
	if err != nil {
		return false, fmt.Errorf("failed to query package manager: %w", err)
	}
	// Check for Termux's official package name
	return strings.Contains(string(output), "com.termux"), nil
}

// androidPythonHome is the prefix dir of the merged android python layout.
func androidPythonHome(binDir string) string {
	return filepath.Join(binDir, "python-home")
}

// androidPythonExecutable returns the C runner path once the runtime is
// merged (otherwise an error telling the user to install Termux).
func androidPythonExecutable(binDir string) (string, error) {
	runner := filepath.Join(binDir, "pyandroid")
	if fi, err := os.Stat(runner); err == nil && !fi.IsDir() {
		return runner, nil
	}
	if fi, err := os.Stat(androidPythonHome(binDir)); err == nil && fi.IsDir() {
		// runtime present but runner not built yet
		return "", fmt.Errorf("android runtime present but runner missing; rerun g4f-go to build it")
	}
	return "", nil
}

// androidPythonExe returns the interpreter path for the android layout
// (the dlopen target inside the runner).
func androidPythonExe(binDir string) string {
	return filepath.Join(androidPythonHome(binDir), "lib", "libpython3.14.so")
}

// buildAndroidRunner compiles pyandroid_runner.c against the downloaded
// python headers using Termux's clang, then links against libpython.
func buildAndroidRunner(binDir string) error {
	if ok, err := IsTermuxInstalled(); err != nil || !ok {
		return fmt.Errorf("Termux is required to build the python runner (install com.termux and retry)")
	}
	home := androidPythonHome(binDir)
	lib := filepath.Join(home, "lib")
	include := filepath.Join(home, "include", "python3.14")
	src := filepath.Join(binDir, "pyandroid_runner.c")
	if _, err := os.Stat(src); err != nil {
		return fmt.Errorf("pyandroid_runner.c missing: %w", err)
	}

	// Termux clang is at $PREFIX/bin/clang; $PREFIX is /data/data/com.termux/files/usr.
	termuxPrefix := "/data/data/com.termux/files/usr"
	cc := filepath.Join(termuxPrefix, "bin", "clang")
	if _, err := os.Stat(cc); err != nil {
		// fall back to whatever clang is on PATH
		cc = "clang"
	}
	out := filepath.Join(binDir, "pyandroid")
	args := []string{
		src,
		"-o", out,
		"-I", include,
		"-L", lib,
		"-Wl,-rpath," + lib,
		"-lpython3.14",
		"-ldl",
	}
	cmd := exec.Command(cc, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("clang build of pyandroid runner failed: %w", err)
	}
	return nil
}

// C runner source embedded via go:embed below; written to binDir before
// building. It dlopens libpython3.14.so (PYTHONHOME set by env) and runs a
// script passed as argv[1..].
const androidRunnerC = `// pyandroid_runner.c — dlopens the downloaded libpython and runs a script,
// mirroring CPython's own android testbed (main_activity.c).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int argc;
    wchar_t **argv;
} PyConfig;

typedef int (*PyConfig_InitPythonConfig_fn)(PyConfig *);
typedef int (*PyConfig_SetBytesArgv_fn)(PyConfig *, int, char **);
typedef int (*PyConfig_SetBytesString_fn)(PyConfig *, wchar_t **, const char *);
typedef void (*PyConfig_Clear_fn)(PyConfig *);
typedef int (*Py_InitializeFromConfig_fn)(const PyConfig *);
typedef int (*Py_RunMain_fn)(void);
typedef const char *(*Py_GetVersion_fn)(void);

int main(int argc, char **argv) {
    void *h = dlopen("libpython3.14.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen libpython3.14.so: %s\n", dlerror());
        return 1;
    }
    PyConfig_InitPythonConfig_fn Init = (PyConfig_InitPythonConfig_fn)dlsym(h, "PyConfig_InitPythonConfig");
    PyConfig_SetBytesArgv_fn SetArgv = (PyConfig_SetBytesArgv_fn)dlsym(h, "PyConfig_SetBytesArgv");
    PyConfig_SetBytesString_fn SetString = (PyConfig_SetBytesString_fn)dlsym(h, "PyConfig_SetBytesString");
    PyConfig_Clear_fn Clear = (PyConfig_Clear_fn)dlsym(h, "PyConfig_Clear");
    Py_InitializeFromConfig_fn InitFromCfg = (Py_InitializeFromConfig_fn)dlsym(h, "Py_InitializeFromConfig");
    Py_RunMain_fn RunMain = (Py_RunMain_fn)dlsym(h, "Py_RunMain");
    if (!Init || !SetArgv || !SetString || !Clear || !InitFromCfg || !RunMain) {
        fprintf(stderr, "missing python symbols: %s\n", dlerror());
        return 1;
    }
    PyConfig cfg;
    if (Init(&cfg)) return 1;
    if (SetArgv(&cfg, argc, argv)) return 1;
    const char *home = getenv("PYTHONHOME");
    if (home && SetString(&cfg, &cfg.home, home)) return 1;
    if (InitFromCfg(&cfg)) return 1;
    Clear(&cfg);
    return RunMain();
}
`

// writeAndroidRunnerSource writes the C helper next to the runtime.
func writeAndroidRunnerSource(binDir string) error {
	src := filepath.Join(binDir, "pyandroid_runner.c")
	if err := os.WriteFile(src, []byte(androidRunnerC), 0o644); err != nil {
		return err
	}
	return nil
}

// finalizeAndroidRuntime (android): merge is done; write the C runner source
// and build it with Termux's clang.
func finalizeAndroidRuntime(binDir string) (string, error) {
	if err := writeAndroidRunnerSource(binDir); err != nil {
		return "", err
	}
	if err := buildAndroidRunner(binDir); err != nil {
		return "", err
	}
	return filepath.Join(binDir, "pyandroid"), nil
}
