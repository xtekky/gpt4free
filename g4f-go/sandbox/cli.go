package sandbox

import (
	"fmt"
	"os"
)

// RunCLI is the `g4f-go sandbox-cli` entry point. It runs the desktop image's
// sandbox-cli with the given arguments, starting the desktop container first
// when it is not running yet.
func RunCLI(args []string) int {
	if len(args) == 0 || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
		printCLIHelp()
		return 0
	}

	o := defaultOptions()
	o.mode = modeDesktop
	o.action = "cli"
	o.open = false
	o.forwarded = args
	if err := o.finish(); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 1
	}

	if useUdocker(o) {
		if err := runUdocker(o); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
			return 1
		}
		return 0
	}

	if useTermux(o) {
		fmt.Fprintln(os.Stderr, "g4f-sandbox: sandbox-cli needs the desktop container; the native Termux backend runs the API only")
		return 1
	}

	e, err := resolveEngine(o)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 1
	}

	if !containerExists(e, o.name) {
		if err := runUp(e, o); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
			return 1
		}
	} else if !containerRunning(e, o.name) {
		if err := e.run("start", o.name); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
			return 1
		}
	}

	if err := runCLI(e, o); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-sandbox:", err)
		return 1
	}
	return 0
}

func printCLIHelp() {
	fmt.Print(`g4f-go sandbox-cli - run sandbox-cli in the g4f desktop container

The desktop container is started on demand (without opening a browser) and
sandbox-cli runs inside it.

Usage:
  g4f-go sandbox-cli <args...>

Common commands:
  list                  list the tools the desktop menu knows about
  install <tool>        install a menu tool
  <tool>                install <tool> if missing, then run it
  settings              model, API key, HOME, provider and proxy
  reset                 clear the saved settings
  package               show, update or pin the g4f package
  urls                  print the public URLs served by the g4f API
  hosts                 print the /etc/hosts entries for the app domains
  harness               pick a coding agent and a model from the g4f API
  polli-sandbox         create/connect/manage hosted Polli sandboxes

Examples:
  g4f-go sandbox-cli list
  g4f-go sandbox-cli install chrome
  g4f-go sandbox-cli settings --show
`)
}
