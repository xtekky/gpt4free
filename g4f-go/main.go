package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	// Meta commands must work even without an embedded runtime.
	switch {
	case len(os.Args) == 1:
		printHelp()
		os.Exit(0)
	case os.Args[1] == "--version" || os.Args[1] == "-v":
		fmt.Printf("g4f-go %s (embedded CPython %s)\n", Version, PythonVer)
		os.Exit(0)
	case os.Args[1] == "help" || os.Args[1] == "--help" || os.Args[1] == "-h":
		printHelp()
		os.Exit(0)
	}

	// Make sure the embedded Python runtime is extracted to ~/.g4f/python-embed.
	py, err := ensureRuntime()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Forward all arguments to g4f's own CLI entry point.
	code, err := runPython(ctx, py, append([]string{"-m", "g4f"}, os.Args[1:]...))
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		os.Exit(1)
	}
	os.Exit(code)
}
