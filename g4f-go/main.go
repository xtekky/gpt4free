package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// Constants usually defined at build time or in a separate file
const (
	Version   = "1.0.0"
	PythonVer = "3.10"
)

func printHelp() {
	fmt.Printf(`g4f-go %s - gpt4free with embedded Python %s

Usage:
  g4f-go <g4f args...>        run gpt4free (e.g. g4f-go client "hello")
  g4f-go api --port 8080      start the OpenAI-compatible API server
  g4f-go gui                  launch the web GUI
  g4f-go install g4f         (re)install the g4f package into the embedded runtime
  g4f-go status               show runtime status
  g4f-go help                 show this help
  g4f-go --version            show version

Environment:
  G4F_PYTHON_ONLY=1           print the embedded python path and exit (for wrappers)
`, Version, PythonVer)
}

func main() {
	// We use a helper function so we can use "return" for exit codes
	os.Exit(run())
}

func run() int {
	binDir := installDir()
	// os.Args[0] is the program name, we want the actual arguments
	args := normalizeArgs(os.Args[1:])

	if len(args) == 0 {
		printHelp()
		return 0
	}

	// 1. Handle non-python commands first
	switch args[0] {
	case "help", "--help", "-h":
		printHelp()
		return 0
	case "--version", "-v":
		fmt.Printf("g4f-go %s (embedded CPython %s)\n", Version, PythonVer)
		return 0
	}

	// 2. Prepare Python Runtime
	py, err := ensureRuntime()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go runtime error:", err)
		return 1
	}

	if os.Getenv("G4F_PYTHON_ONLY") == "1" {
		fmt.Println(py)
		return 0
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// 3. Handle management commands
	switch args[0] {
	case "status":
		return handleStatus(ctx, binDir, py)
	case "install", "uninstall":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: g4f-go install <package>")
			return 2
		}
		code, err := runPython(ctx, py, append([]string{"-m", "pip"}, args...))
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	case "bootstrap":
		code, err := runPython(ctx, py, []string{"-m", "pip", "install", "--no-input", "g4f[slim]"}, pipEnv(binDir)...)
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
		}
		return code
	}

	// 4. Auto-install G4F if missing before running commands
	exe := pythonExecutable(binDir)
	if err := installG4F(binDir, exe, time.Now()); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}

	// 5. Default: forward everything to the g4f module
	code, err := runPython(ctx, py, append([]string{"-m", "g4f"}, args...))
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go execution error:", err)
		return 1
	}
	return code
}

// --- Helper Functions (Stubs/Implementations) ---

func normalizeArgs(args []string) []string {
	if len(args) > 0 && args[0] == "g4f-go" {
		return normalizeArgs(args[1:])
	}
	return args
}

func installDir() string {
	configDir, _ := os.UserConfigDir()
	return filepath.Join(configDir, "g4f-go")
}

func pythonExecutable(binDir string) string {
	// Adjust for Windows if necessary
	return filepath.Join(binDir, "python")
}

func ensureRuntime() (string, error) {
	dir := installDir()
	pyPath := pythonExecutable(dir)
	// Logic to extract embedded python would go here
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		os.MkdirAll(dir, 0755)
	}
	return pyPath, nil
}

func runPython(ctx context.Context, pyPath string, args []string, env ...string) (int, error) {
	cmd := exec.CommandContext(ctx, pyPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	cmd.Env = append(os.Environ(), env...)

	err := cmd.Run()
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			return exitError.ExitCode(), nil
		}
		return 1, err
	}
	return 0, nil
}

func installG4F(binDir, pyExe string, start time.Time) error {
	stamp := filepath.Join(binDir, ".g4f_installed")
	if _, err := os.Stat(stamp); err == nil {
		return nil // Already installed
	}
	// Logic to install g4f via pip
	os.WriteFile(stamp, []byte(start.String()), 0644)
	return nil
}

func pipEnv(binDir string) []string {
	return []string{fmt.Sprintf("PYTHONPATH=%s", binDir)}
}

func handleStatus(ctx context.Context, binDir, py string) int {
	exe := pythonExecutable(binDir)
	fmt.Printf("binary dir: %s\n", binDir)
	fmt.Printf("python:     %s\n", exe)
	
	code, _ := runPython(ctx, py, []string{"--version"})
	return code
}
