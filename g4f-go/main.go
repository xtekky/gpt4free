package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// printHelp shows the g4f-go usage.
func printHelp() {
	fmt.Printf(`g4f-go %s - gpt4free with a downloaded CPython %s runtime

Usage:
  g4f-go <g4f args...>        run gpt4free (e.g. g4f-go client "hello")
  g4f-go script.py [args...]  run a .py file with the bundled Python
  g4f-go api --port 8080      start the OpenAI-compatible API server
  g4f-go gui                  launch the web GUI
  g4f-go status               show runtime download/install status
  g4f-go install g4f         (re)install the g4f package (network)
  g4f-go bootstrap            refresh the g4f package installation
  g4f-go --version            print version
  g4f-go help                 show this help

The CPython runtime downloads on first run (with progress feedback) into
%s. Set G4F_PYTHON_ONLY=1 to print the interpreter path and exit.
`, Version, PythonVer, installDir())
}

func main() {
	os.Exit(runMain())
}

func runMain() int {
	binDir := installDir()
	args := os.Args[1:]
	if len(args) == 0 {
		printHelp()
		return 0
	}

	py, err := ensureRuntime()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	switch args[0] {
	case "help", "--help", "-h":
		printHelp()
		return 0
	case "--version", "-v":
		fmt.Printf("g4f-go %s (CPython %s)\n", Version, PythonVer)
		return 0
	}

	switch args[0] {
	case "status":
		stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")
		fmt.Printf("binary dir: %s\n", binDir)
		fmt.Printf("python:     %s\n", py)
		if _, serr := os.Stat(py); serr == nil {
			fmt.Println("runtime:    downloaded & extracted")
		} else {
			fmt.Println("runtime:    not downloaded yet (will download on first run)")
		}
		if _, serr := os.Stat(stamp); serr == nil {
			fmt.Println("g4f:        installed")
		} else {
			fmt.Println("g4f:        not installed (will install on first run)")
		}
		code, err := runPython(ctx, py, []string{"--version"})
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	case "install", "uninstall":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: g4f-go install g4f")
		}
		code, err := runPython(ctx, py, append([]string{"-m", "pip"}, args...))
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	case "bootstrap":
		// Refresh the embedded package installation (e.g. after updating g4f-go).
		code, err := runPython(ctx, py, []string{"-m", "pip", "install", "--no-input", "g4f[slim]"}, pipEnv(binDir)...)
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	}

	if os.Getenv("G4F_PYTHON_ONLY") == "1" {
		fmt.Println(py)
		return 0
	}

	// If the first argument is a .py file, run it directly with the bundled
	// Python interpreter (forwarding any remaining args to the script).
	if strings.HasSuffix(args[0], ".py") {
		if _, serr := os.Stat(args[0]); serr != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", serr)
			return 1
		}
		code, err := runPython(ctx, py, args)
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	}

	exe, err := pythonExecutable(binDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if !g4fIsInstalled(binDir) {
		// First call: install g4f.
		start := time.Now()
		if err := installG4F(binDir, exe, start); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
	} else if len(args) > 0 && isUpgradeCommand(args[0]) {
		// Subsequent call with gui/api/dev: upgrade g4f.
		start := time.Now()
		if err := upgradeG4F(binDir, exe, start); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
	}

	// Default: forward everything to the g4f module.
	code, err := runPython(ctx, py, append([]string{"-m", "g4f"}, args...))
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
	}
	return code
}

// normalizeArgs cleans leading `g4f-go` repeats (typos like `g4f-go g4f-go ...`).
func normalizeArgs(args []string) []string {
	for len(args) > 0 {
		args = args[1:]
	}
	// A leading binary-name echo (e.g. argv[0]) is stripped by callers.
	return args
}

// isUpgradeCommand reports whether the subcommand should trigger an
// automatic g4f upgrade (only when g4f is already installed).
func isUpgradeCommand(arg string) bool {
	switch arg {
	case "api", "gui", "dev":
		return true
	}
	return false
}

// hasSubcommand reports whether args start with a known g4f-go subcommand.
func hasSubcommand(args []string) bool {
	if len(args) == 0 {
		return true
	}
	switch args[0] {
	case "help", "--help", "-h", "--version", "-v", "status", "install", "bootstrap":
		return true
	}
	if strings.HasPrefix(args[0], "-") {
		return true
	}
	return false
}
