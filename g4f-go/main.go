package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	module := "g4f"
	
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
	case os.Args[1] == "install" || os.Args[1] == "uninstall":
		module = "pip" // Fixed: Assign to the existing variable, don't redeclare
	}

	py, err := ensureRuntime()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Runs: python -m [module] [args...]
	code, err := runPython(ctx, py, append([]string{"-m", module}, os.Args[1:]...))
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		os.Exit(1)
	}
	os.Exit(code)
}
