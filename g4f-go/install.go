package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// installOptions holds the parsed `g4f-go install` flags.
type installOptions struct {
	Prefix string
	DryRun bool
	Help   bool
}

// launcherSelfInstall reports whether `g4f-go install ...` refers to the
// launcher itself instead of the pip passthrough (`g4f-go install g4f`).
func launcherSelfInstall(args []string) bool {
	if len(args) == 0 {
		return true
	}
	switch args[0] {
	case "help", "--help", "-h", "--prefix", "-p", "--prefix=", "--dry-run", "-n":
		return true
	default:
		return strings.HasPrefix(args[0], "--prefix=")
	}
}

func printInstallHelp() {
	fmt.Print(`g4f-go install - put the g4f-go launcher on PATH

Usage:
  g4f-go install                    install into a bin directory on PATH
  g4f-go install --prefix DIR       install into DIR
  g4f-go install --dry-run          show the destination without writing
  g4f-go install g4f [args...]      pip passthrough (install the g4f package)

The launcher is copied as-is; the downloaded CPython runtime and the installed
g4f package are not touched. The default destination is $G4F_GO_PREFIX when
set, otherwise $PREFIX/bin on Termux, ~/.local/bin when it is on PATH, or the
first writable directory on PATH.
`)
}

// runInstallCommand handles `g4f-go install`: copy this executable into a bin
// directory that is already on PATH.
func runInstallCommand(args []string) int {
	opts, err := parseInstallOptions(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 2
	}
	if opts.Help {
		printInstallHelp()
		return 0
	}
	if opts.Prefix == "" {
		opts.Prefix, err = installBinDir()
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
	}
	opts.Prefix = expandUserPath(opts.Prefix)

	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if resolved, rerr := filepath.EvalSymlinks(exe); rerr == nil {
		exe = resolved
	}
	target := filepath.Join(opts.Prefix, filepath.Base(exe))

	if samePath(exe, target) {
		fmt.Printf("install: g4f-go is already installed at %s\n", target)
		return 0
	}
	if opts.DryRun {
		fmt.Printf("install: would copy %s to %s\n", exe, target)
		return 0
	}
	if err := os.MkdirAll(opts.Prefix, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if err := installExecutable(exe, target); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	fmt.Printf("install: installed g4f-go %s at %s\n", Version, target)
	if !pathContainsDir(opts.Prefix) {
		fmt.Fprintf(os.Stderr, "g4f-go: warning: %s is not on PATH\n", opts.Prefix)
	}
	return 0
}

func parseInstallOptions(args []string) (installOptions, error) {
	opts := installOptions{}
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "help" || arg == "--help" || arg == "-h":
			opts.Help = true
			return opts, nil
		case arg == "--prefix" || arg == "-p":
			if i+1 >= len(args) {
				return opts, fmt.Errorf("install: %s requires a directory", arg)
			}
			i++
			opts.Prefix = args[i]
		case strings.HasPrefix(arg, "--prefix="):
			opts.Prefix = strings.TrimPrefix(arg, "--prefix=")
		case arg == "--dry-run" || arg == "-n":
			opts.DryRun = true
		default:
			return opts, fmt.Errorf("install: unknown option %q", arg)
		}
	}
	if opts.Prefix != "" && strings.TrimSpace(opts.Prefix) == "" {
		return opts, fmt.Errorf("install: --prefix must not be empty")
	}
	return opts, nil
}

// installBinDir picks a writable directory that is already on PATH.
func installBinDir() (string, error) {
	if prefix := strings.TrimSpace(os.Getenv("G4F_GO_PREFIX")); prefix != "" {
		return prefix, nil
	}
	if IsTermuxSystem() {
		if prefix := strings.TrimSpace(os.Getenv("PREFIX")); prefix != "" {
			return filepath.Join(prefix, "bin"), nil
		}
	}

	homeBin := ""
	if home, err := os.UserHomeDir(); err == nil {
		homeBin = filepath.Join(home, ".local", "bin")
	}
	if homeBin != "" && pathContainsDir(homeBin) {
		if err := os.MkdirAll(homeBin, 0o755); err == nil && dirWritable(homeBin) {
			return homeBin, nil
		}
	}
	for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
		dir = strings.TrimSpace(dir)
		if dir == "" || dir == "." {
			continue
		}
		if dirWritable(dir) {
			return dir, nil
		}
	}
	if homeBin != "" {
		if err := os.MkdirAll(homeBin, 0o755); err == nil && dirWritable(homeBin) {
			// Not on PATH yet, but the conventional user bin is still the
			// least surprising destination. The caller prints a warning.
			return homeBin, nil
		}
	}
	return "", fmt.Errorf("install: no writable bin directory on PATH (use: g4f-go install --prefix DIR)")
}

// dirWritable reports whether a directory exists and accepts new files.
func dirWritable(dir string) bool {
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return false
	}
	file, err := os.CreateTemp(dir, ".g4f-go-write-*")
	if err != nil {
		return false
	}
	name := file.Name()
	file.Close()
	_ = os.Remove(name)
	return true
}

// pathContainsDir reports whether dir is one of the PATH entries.
func pathContainsDir(dir string) bool {
	want := cleanPath(dir)
	for _, entry := range filepath.SplitList(os.Getenv("PATH")) {
		if cleanPath(entry) == want {
			return true
		}
	}
	return false
}

func cleanPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		path = resolved
	}
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return filepath.Clean(path)
}

func expandUserPath(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "~\\") {
		if home, err := os.UserHomeDir(); err == nil {
			if path == "~" {
				return home
			}
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

func samePath(a, b string) bool {
	if cleanPath(a) == cleanPath(b) {
		return true
	}
	ai, aerr := os.Stat(a)
	bi, berr := os.Stat(b)
	return aerr == nil && berr == nil && os.SameFile(ai, bi)
}

// installExecutable copies src to target atomically and keeps it executable.
func installExecutable(src, target string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	dir := filepath.Dir(target)
	tmp, err := os.CreateTemp(dir, ".g4f-go-install-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o755); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, target); err != nil {
		if runtime.GOOS != "windows" {
			return err
		}
		// Windows cannot rename over a running executable. Replace the target
		// only when it is not the process we are running from.
		if rerr := os.Remove(target); rerr != nil && !os.IsNotExist(rerr) {
			return err
		}
		if rerr := os.Rename(tmpName, target); rerr != nil {
			return rerr
		}
	}
	return nil
}
