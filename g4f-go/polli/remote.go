package polli

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const installScriptURL = "https://raw.githubusercontent.com/xtekky/gpt4free/main/bin/install.sh"

type setupOptions struct {
	desktop     bool
	version     string
	forceLocal  bool
	forceRemote bool
}

func sshTarget(id string) string {
	return id + ".polli"
}

func sshRun(id, command string) error {
	cmd := exec.Command("ssh", sshTarget(id), command)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ssh %s: %w", sshTarget(id), err)
	}
	return nil
}

func sshOutput(id, command string) (string, error) {
	cmd := exec.Command("ssh", sshTarget(id), command)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("ssh %s: %s", sshTarget(id), strings.TrimSpace(out.String()))
	}
	return strings.TrimSpace(out.String()), nil
}

func remoteArch(id string) string {
	out, err := sshOutput(id, "uname -m")
	if err != nil {
		return ""
	}
	switch strings.ToLower(out) {
	case "x86_64", "amd64":
		return "amd64"
	case "aarch64", "arm64":
		return "arm64"
	default:
		return ""
	}
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// setupSandbox installs g4f-go and sandbox-cli in an existing Polli sandbox
// and, when requested, brings up the Xfce/TigerVNC/noVNC desktop there.
func setupSandbox(id string, o setupOptions) error {
	if !validID(id) {
		return fmt.Errorf("invalid sandbox id %q", id)
	}
	fmt.Fprintf(os.Stderr, "polli-sandbox: preparing %s.polli\n", id)
	if err := sshRun(id, "sudo -n true"); err != nil {
		return fmt.Errorf("%s does not allow passwordless sudo over ssh; run the setup inside it manually", sshTarget(id))
	}

	base := "sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true; " +
		"sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ca-certificates curl tar unzip python3"
	if err := sshRun(id, base); err != nil {
		return fmt.Errorf("could not install base packages in %s: %w", id, err)
	}

	arch := remoteArch(id)
	useLocal := !o.forceRemote && runtime.GOOS == "linux" && arch == runtime.GOARCH
	if o.forceLocal {
		if runtime.GOOS != "linux" || (arch != "" && arch != runtime.GOARCH) {
			return fmt.Errorf("--local needs a Linux %s g4f-go build; this host is %s/%s", arch, runtime.GOOS, runtime.GOARCH)
		}
		useLocal = true
	}

	var binary []byte
	if useLocal {
		exe, err := os.Executable()
		if err != nil {
			return fmt.Errorf("locate the running g4f-go binary: %w", err)
		}
		binary, err = os.ReadFile(exe)
		if err != nil {
			return fmt.Errorf("read %s: %w", exe, err)
		}
	}

	if err := uploadAssets(id, binary); err != nil {
		return err
	}

	if !useLocal {
		fmt.Fprintf(os.Stderr, "polli-sandbox: installing g4f-go in %s.polli\n", id)
		args := []string{"--prefix", "/usr/local/bin"}
		if o.version != "" {
			args = append(args, "--version", o.version)
		}
		quoted := make([]string, 0, len(args))
		for _, a := range args {
			quoted = append(quoted, shellQuote(a))
		}
		cmd := fmt.Sprintf("curl -fsSL %s | sudo -n bash -s -- %s",
			shellQuote(installScriptURL), strings.Join(quoted, " "))
		if err := sshRun(id, cmd); err != nil {
			return fmt.Errorf("could not install g4f-go in %s: %w", id, err)
		}
	}

	if o.desktop {
		fmt.Fprintf(os.Stderr, "polli-sandbox: installing the desktop in %s.polli\n", id)
		if err := sshRun(id, "sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends novnc websockify"); err != nil {
			return fmt.Errorf("could not install noVNC in %s: %w", id, err)
		}
		if err := sshRun(id, "sudo -n /usr/local/bin/polli-sandbox-bootstrap.sh --desktop"); err != nil {
			return fmt.Errorf("the desktop bootstrap failed in %s: %w", id, err)
		}
	}

	if err := sshRun(id, "test -x /usr/local/bin/g4f-go && test -x /usr/local/bin/sandbox-cli"); err != nil {
		return fmt.Errorf("g4f-go or sandbox-cli is missing after setup in %s", id)
	}
	return nil
}

func uploadAssets(id string, g4fGo []byte) error {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)

	type assetFile struct {
		name string
		data []byte
		mode int64
	}
	files := []assetFile{
		{"usr/local/bin/sandbox-cli", sandboxCLI, 0o755},
		{"usr/local/bin/polli-sandbox-bootstrap.sh", bootstrapScript, 0o755},
		{"usr/local/bin/patch-xfce-menu.py", patchXfceMenu, 0o755},
		{"usr/local/bin/sandbox-discover", sandboxDiscover, 0o755},
		{"usr/local/bin/sandbox-docs", sandboxDocs, 0o755},
		{"usr/local/bin/start-browser", startBrowser, 0o755},
	}
	if len(g4fGo) > 0 {
		files = append(files, assetFile{"usr/local/bin/g4f-go", g4fGo, 0o755})
	}

	for _, f := range files {
		hdr := &tar.Header{
			Name:     f.name,
			Mode:     f.mode,
			Size:     int64(len(f.data)),
			Typeflag: tar.TypeReg,
			ModTime:  time.Now(),
			Uid:      0,
			Gid:      0,
			Uname:    "root",
			Gname:    "root",
		}
		if err := tw.WriteHeader(hdr); err != nil {
			return fmt.Errorf("write tar header: %w", err)
		}
		if _, err := tw.Write(f.data); err != nil {
			return fmt.Errorf("write tar data: %w", err)
		}
	}
	if err := tw.Close(); err != nil {
		return fmt.Errorf("close tar: %w", err)
	}
	if err := gz.Close(); err != nil {
		return fmt.Errorf("close gzip: %w", err)
	}

	cmd := exec.Command("ssh", sshTarget(id), "sudo -n tar -C / -xzf -")
	cmd.Stdin = &buf
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("could not copy the sandbox assets into %s: %w", id, err)
	}
	return nil
}
