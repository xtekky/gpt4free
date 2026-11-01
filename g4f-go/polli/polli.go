// Package polli adds the `g4f-go polli-sandbox` command: it installs the
// Pollinations CLI, creates a hosted Polli (E2B) sandbox and installs g4f-go
// plus sandbox-cli inside it. With --desktop it also installs and starts the
// Xfce/TigerVNC/noVNC desktop.
package polli

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

var sandboxIDPattern = regexp.MustCompile(`(?m)^Sandbox ([A-Za-z0-9-]+) created`)

// Run executes `g4f-go polli-sandbox` / `g4f-go polli`.
func Run(args []string) int {
	if len(args) == 0 {
		return create(nil)
	}
	switch args[0] {
	case "help", "--help", "-h":
		printHelp()
		return 0
	case "install":
		if err := ensurePolli(); err != nil {
			fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
			return 1
		}
		if err := ensureSSH(); err != nil {
			fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
			return 1
		}
		fmt.Println("Polli and the OpenSSH client are installed and ready.")
		return 0
	case "create", "up", "new":
		return create(args[1:])
	case "install-menu", "setup":
		return installMenu(args[1:])
	case "list", "connect", "ssh", "timeout", "pause", "logs", "kill", "ssh-config":
		return passthrough(args)
	default:
		if strings.HasPrefix(args[0], "-") {
			return create(args)
		}
		return passthrough(args)
	}
}

type createOptions struct {
	setupOptions
	template string
	timeout  int
	connect  bool
	noSetup  bool
}

func parseCreate(args []string) (createOptions, error) {
	o := createOptions{
		template: "pollinations",
		timeout:  600,
	}
	var positional []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch arg {
		case "--desktop", "-d":
			o.desktop = true
		case "--no-desktop":
			o.desktop = false
		case "--connect":
			o.connect = true
		case "--no-connect":
			o.connect = false
		case "--no-setup":
			o.noSetup = true
		case "--local":
			o.forceLocal = true
		case "--remote":
			o.forceRemote = true
		case "--template", "--image":
			if i+1 >= len(args) {
				return o, fmt.Errorf("%s requires a value", arg)
			}
			i++
			o.template = args[i]
		case "--timeout":
			if i+1 >= len(args) {
				return o, fmt.Errorf("--timeout requires a value")
			}
			i++
			n, err := strconv.Atoi(args[i])
			if err != nil || n < 0 {
				return o, fmt.Errorf("--timeout must be a whole number of seconds: %s", args[i])
			}
			o.timeout = n
		case "--g4f-go-version", "--version":
			if i+1 >= len(args) {
				return o, fmt.Errorf("%s requires a value", arg)
			}
			i++
			o.version = args[i]
		default:
			if strings.HasPrefix(arg, "-") {
				return o, fmt.Errorf("unknown option %q (run: g4f-go polli-sandbox help)", arg)
			}
			positional = append(positional, arg)
		}
	}
	if len(positional) > 0 {
		o.template = positional[0]
	}
	if len(positional) > 1 {
		n, err := strconv.Atoi(positional[1])
		if err != nil || n < 0 {
			return o, fmt.Errorf("timeout must be a whole number of seconds: %s", positional[1])
		}
		o.timeout = n
	}
	if len(positional) > 2 {
		return o, fmt.Errorf("too many arguments for polli-sandbox create")
	}
	if o.template == "" {
		return o, fmt.Errorf("template must not be empty")
	}
	if o.forceLocal && o.forceRemote {
		return o, fmt.Errorf("--local and --remote cannot be combined")
	}
	return o, nil
}

func create(args []string) int {
	o, err := parseCreate(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 2
	}
	if err := ensurePolli(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	if err := ensureSSH(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	bin, err := findPolli()
	if err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}

	_ = runCommand(bin, "sandbox", "ssh-config")

	fmt.Fprintf(os.Stderr, "polli-sandbox: creating a Polli sandbox (%s, timeout %ds)\n", o.template, o.timeout)
	id, err := createSandbox(bin, o)
	if err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	_ = runCommand(bin, "sandbox", "ssh-config")

	if !o.noSetup {
		if err := setupSandbox(id, o.setupOptions); err != nil {
			fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
			fmt.Fprintf(os.Stderr, "The sandbox %s still exists; retry with:\n  g4f-go polli-sandbox install-menu %s%s\n", id, id, desktopFlag(o.desktop))
			return 1
		}
	}
	printReady(id, o.desktop)
	if o.connect {
		return runInteractive("ssh", sshTarget(id))
	}
	return 0
}

func createSandbox(bin string, o createOptions) (string, error) {
	args := []string{"sandbox", "create", "--timeout", strconv.Itoa(o.timeout), o.template}
	cmd := exec.Command(bin, args...)
	var out bytes.Buffer
	cmd.Stdin = os.Stdin
	cmd.Stdout = io.MultiWriter(os.Stdout, &out)
	cmd.Stderr = io.MultiWriter(os.Stderr, &out)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("polli sandbox create failed: %w", err)
	}
	if m := sandboxIDPattern.FindStringSubmatch(out.String()); len(m) == 2 {
		return m[1], nil
	}
	return "", fmt.Errorf("the sandbox was created but its id was not found in polli's output; run: polli sandbox list")
}

func installMenu(args []string) int {
	o := setupOptions{}
	var id string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--desktop", "-d":
			o.desktop = true
		case "--no-desktop":
			o.desktop = false
		case "--local":
			o.forceLocal = true
		case "--remote":
			o.forceRemote = true
		case "--g4f-go-version", "--version":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "polli-sandbox: --g4f-go-version requires a value")
				return 2
			}
			i++
			o.version = args[i]
		default:
			if strings.HasPrefix(args[i], "-") {
				fmt.Fprintf(os.Stderr, "polli-sandbox: unknown option %q\n", args[i])
				return 2
			}
			if id != "" {
				fmt.Fprintln(os.Stderr, "polli-sandbox: usage: g4f-go polli-sandbox install-menu <id> [--desktop]")
				return 2
			}
			id = args[i]
		}
	}
	if id == "" {
		fmt.Fprintln(os.Stderr, "polli-sandbox: usage: g4f-go polli-sandbox install-menu <id> [--desktop]")
		return 2
	}
	if err := ensurePolli(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	if err := ensureSSH(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	bin, _ := findPolli()
	if bin != "" {
		_ = runCommand(bin, "sandbox", "ssh-config")
	}
	if err := setupSandbox(id, o); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	printReady(id, o.desktop)
	return 0
}

func passthrough(args []string) int {
	if err := ensurePolli(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	if err := ensureSSH(); err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	bin, err := findPolli()
	if err != nil {
		fmt.Fprintln(os.Stderr, "polli-sandbox:", err)
		return 1
	}
	return runInteractive(bin, append([]string{"sandbox"}, args...)...)
}

func validID(id string) bool {
	if id == "" {
		return false
	}
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			continue
		}
		return false
	}
	return true
}

func desktopFlag(desktop bool) string {
	if desktop {
		return " --desktop"
	}
	return ""
}

func printReady(id string, desktop bool) {
	fmt.Println()
	fmt.Printf("Polli sandbox %s is ready.\n", id)
	fmt.Println("  g4f-go and sandbox-cli are installed in /usr/local/bin.")
	fmt.Printf("  connect:  polli sandbox connect %s   (or: ssh %s)\n", id, sshTarget(id))
	if desktop {
		fmt.Println("  desktop:  http://127.0.0.1:8080/desktop/ after:")
		fmt.Printf("            ssh -L 8080:127.0.0.1:8080 %s\n", sshTarget(id))
		fmt.Println("  vnc:      127.0.0.1:5900 after:")
		fmt.Printf("            ssh -L 5900:127.0.0.1:5900 %s\n", sshTarget(id))
	} else {
		fmt.Printf("  shell:    ssh %s\n", sshTarget(id))
		fmt.Printf("  setup:    g4f-go polli-sandbox install-menu %s --desktop\n", id)
	}
	fmt.Println()
}

func printHelp() {
	fmt.Print(`g4f-go polli-sandbox - create a Polli (E2B) sandbox with g4f-go

Installs the Pollinations CLI and the OpenSSH client on this machine, creates a
hosted Polli sandbox, and installs g4f-go and sandbox-cli inside it. With
--desktop it also installs Xfce + TigerVNC + noVNC and starts the desktop.

Usage:
  g4f-go polli-sandbox [create] [options]   create a sandbox (default)
  g4f-go polli-sandbox install              install Polli + OpenSSH only
  g4f-go polli-sandbox install-menu <id>    set up an existing sandbox
  g4f-go polli-sandbox list|connect|logs|pause|kill|timeout|ssh-config ...
                                            forward to "polli sandbox"

Create options:
      --desktop              Install the Xfce/VNC/noVNC desktop
      --template NAME        Sandbox template (default: pollinations)
      --timeout SECONDS      Lifetime in seconds, 0 = never expire (default: 600)
      --connect              SSH into the sandbox after setup
      --no-setup             Create the sandbox but do not install anything in it
      --g4f-go-version V     Install this g4f-go release in the sandbox
      --local                Upload this running Linux g4f-go binary
      --remote               Download the latest g4f-go release in the sandbox

Examples:
  g4f-go polli-sandbox
  g4f-go polli-sandbox --desktop
  g4f-go polli-sandbox --template pollinations-large --timeout 3600 --desktop
  g4f-go polli-sandbox install-menu abc123 --desktop
  g4f-go polli-sandbox list
  g4f-go polli-sandbox connect abc123

Polli sandboxes are billed to the Pollinations account. Run "polli auth login"
first if "polli sandbox create" asks you to log in.
`)
}
