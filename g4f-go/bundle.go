package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// embeddedBundle carries the Python sources that ship inside the g4f-go
// binary. The layout mirrors the repository so that packages keep their
// relative paths (remote_desktop/ and its sibling web/ asset directory).
//
// The contents are produced by sync-bundle.sh from projects/remote-desktop
// and committed, so a plain `go build` works without extra steps.
//
// The `all:` prefix is required: without it go:embed skips files whose names
// start with `_` or `.`, which would drop __init__.py and __main__.py.
//
//go:embed all:bundle
var embeddedBundle embed.FS

// bundleRoot is the directory inside embeddedBundle that holds the payload.
const bundleRoot = "bundle"

// bundleStampName records which bundle revision was extracted into the runtime.
const bundleStampName = ".bundle-revision"

// bundledModuleDeps lists extra PyPI packages a bundled module needs at
// runtime but that are not part of the g4f[slim] install. They are installed
// on demand the first time the module is used.
var bundledModuleDeps = map[string][]string{
	"remote_desktop": {"qrcode", "pynput"},
}

// bundleDir is where the embedded bundle is extracted inside the runtime dir.
func bundleDir(binDir string) string {
	return filepath.Join(binDir, bundleRoot)
}

// bundleStampPath is the marker file used to detect a stale extraction.
func bundleStampPath(binDir string) string {
	return filepath.Join(binDir, ".g4f-runtime", bundleStampName)
}

// bundledModules returns the sorted list of module names shipped in the binary.
func bundledModules() []string {
	entries, err := fs.ReadDir(embeddedBundle, bundleRoot)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// isBundledModule reports whether name is a module shipped inside the binary.
func isBundledModule(name string) bool {
	for _, m := range bundledModules() {
		if m == name {
			return true
		}
	}
	return false
}

// ensureBundle extracts the embedded bundle into the runtime directory when it
// is missing or was extracted from a different bundle revision, and returns the
// directory to put on PYTHONPATH.
func ensureBundle(binDir string) (string, error) {
	dir := bundleDir(binDir)
	stamp := bundleStampPath(binDir)

	if data, err := os.ReadFile(stamp); err == nil && strings.TrimSpace(string(data)) == BundleRevision {
		if _, serr := os.Stat(filepath.Join(dir, "remote_desktop", "__main__.py")); serr == nil {
			return dir, nil
		}
	}

	if err := extractBundle(dir); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(stamp, []byte(BundleRevision), 0o644); err != nil {
		return "", err
	}
	return dir, nil
}

// extractBundle writes every embedded file below dest, replacing stale files
// from a previous revision.
func extractBundle(dest string) error {
	return fs.WalkDir(embeddedBundle, bundleRoot, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(bundleRoot, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return os.MkdirAll(dest, 0o755)
		}
		target := filepath.Join(dest, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := embeddedBundle.ReadFile(path)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}

// bundleEnv is the environment for running bundled modules: the runtime
// environment with the extracted bundle prepended to PYTHONPATH.
func bundleEnv(binDir string) []string {
	env := pipEnv(binDir)
	dir := bundleDir(binDir)
	for i, kv := range env {
		if strings.HasPrefix(kv, "PYTHONPATH=") {
			env[i] = "PYTHONPATH=" + dir + string(os.PathListSeparator) + strings.TrimPrefix(kv, "PYTHONPATH=")
			return env
		}
	}
	return append(env, "PYTHONPATH="+dir)
}

// moduleImportable reports whether the interpreter can import the given module.
func moduleImportable(exe, module string, env ...string) bool {
	probe := fmt.Sprintf("import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(%q) else 1)", module)
	_, err := capturePython(exe, []string{"-c", probe}, env...)
	return err == nil
}

// ensureModuleDeps installs the extra packages a bundled module needs. Missing
// optional dependencies only produce a warning: the module degrades gracefully
// (e.g. remote_desktop runs without pynput input control).
func ensureModuleDeps(binDir, exe, module string) {
	deps := bundledModuleDeps[module]
	if len(deps) == 0 {
		return
	}
	env := bundleEnv(binDir)
	var missing []string
	for _, dep := range deps {
		if !moduleImportable(exe, dep, env...) {
			missing = append(missing, dep)
		}
	}
	if len(missing) == 0 {
		return
	}
	fmt.Printf("Installing extra dependencies for %s: %s\n", module, strings.Join(missing, ", "))
	if err := ensurePip(binDir, exe); err != nil {
		fmt.Fprintf(os.Stderr, "g4f-go: warning: %v\n", err)
		return
	}
	args := append([]string{"-m", "pip", "install", "--no-input"}, missing...)
	code, err := runPython(noSignalCtx(), exe, args, env...)
	if err != nil || code != 0 {
		fmt.Fprintf(os.Stderr, "g4f-go: warning: could not install %s (exit %d): %v\n",
			strings.Join(missing, ", "), code, err)
	}
}

// runBundledModule runs `python -m <module>` with the embedded bundle on
// PYTHONPATH, installing any extra dependencies the module needs.
func runBundledModule(ctx context.Context, binDir, exe, module string, args []string) int {
	if _, err := ensureBundle(binDir); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	ensureModuleDeps(binDir, exe, module)
	env := bundleEnv(binDir)
	if module == "remote_desktop" {
		// The remote desktop needs a relay to reach a phone on cellular; bring
		// up the embedded STUN/TURN server unless one is already configured.
		if turnEnv, stop := remoteDesktopTurnEnv(ctx); len(turnEnv) > 0 {
			defer stop()
			env = append(env, turnEnv...)
		}
	}
	code, err := runPython(ctx, exe, append([]string{"-m", module}, args...), env...)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
	}
	return code
}

// parseModuleFlag extracts a leading `-m <module>` / `--module <module>` flag.
// It returns the module name (empty when absent) and the remaining arguments.
func parseModuleFlag(args []string) (string, []string) {
	if len(args) == 0 {
		return "", args
	}
	arg := args[0]
	switch {
	case arg == "-m" || arg == "--module":
		if len(args) < 2 {
			return "", args
		}
		return args[1], args[2:]
	case strings.HasPrefix(arg, "--module="):
		return strings.TrimPrefix(arg, "--module="), args[1:]
	case strings.HasPrefix(arg, "-m") && len(arg) > 2:
		return arg[2:], args[1:]
	}
	return "", args
}

// moduleFlagNeedsValue reports whether args start with -m/--module but no
// module name follows, so the caller can report a usage error instead of
// forwarding a broken argument list to the interpreter.
func moduleFlagNeedsValue(args []string) bool {
	if len(args) == 0 {
		return false
	}
	switch args[0] {
	case "-m", "--module":
		return len(args) < 2
	case "--module=":
		return true
	}
	return false
}
