package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Lightpanda is the headless browser g4f drives over CDP. g4f-go installs the
// g4f-copilot/browser fork, which ships a Chrome-impersonating default
// fingerprint (Chrome/152 UA plus matching client hints) and self-contained
// Linux zips. Upstream lightpanda-io/browser publishes no Windows build, so
// Windows keeps using the community port qidiai/lightpanda-windows-port.
const (
	browserVersion        = "v0.1.0"
	browserWindowsVersion = "portable-v0.2.5"
	browserDefaultHost    = "127.0.0.1"
	browserDefaultPort    = 9222
)

// browserV8Version is the zig-v8-fork release the Linux x64 build links
// against. The binary records libc_v8.so as a DT_NEEDED entry, so the shared
// library has to sit next to it (see patchBrowserRunpath).
const browserV8Version = "v0.5.9"

// browserSpec describes one platform's Lightpanda download.
type browserSpec struct {
	Key     string // manifest-style platform key, e.g. "linux-x64"
	Version string // pinned release tag
	URL     string
	Size    int64
	SHA256  string
	Archive bool   // true when the download is a zip instead of a bare binary
	Binary  string // executable name inside the install directory
	// Lib is a shared library the binary needs next to it, downloaded
	// separately and verified on its own.
	Lib *browserLib
}

// browserLib is a companion shared library shipped outside the release zip.
type browserLib struct {
	Name   string // file name inside the install directory
	URL    string
	Size   int64
	SHA256 string
}

// browserSpecForHost returns the pinned download for the current OS/arch.
func browserSpecForHost() (*browserSpec, error) {
	base := "https://github.com/g4f-copilot/browser/releases/download/" + browserVersion + "/"
	switch runtime.GOOS {
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return &browserSpec{
				Key: "linux-x64", Version: browserVersion,
				URL:     base + "lightpanda.zip",
				Size:    71155619,
				SHA256:  "d0c82aa314fa58ed9a64ef773e0c23cde79321214d0110bcdcf7e749b74e553c",
				Archive: true,
				Binary:  "lightpanda",
				Lib: &browserLib{
					Name: "libc_v8.so",
					URL: "https://github.com/lightpanda-io/zig-v8-fork/releases/download/" +
						browserV8Version + "/libc_v8_15.5.35.13_linux_x86_64.so",
					Size:   80223768,
					SHA256: "e9dd2fab029d235a945e3d0fa82e9f120e6a46961ed434672fee5cf9741bce88",
				},
			}, nil
		case "arm64":
			return &browserSpec{
				Key: "linux-arm64", Version: browserVersion,
				URL:     base + "lightpanda-arm64.zip",
				Size:    56096169,
				SHA256:  "be56a5105fa275357060e588de48e76db2945f9a233b5f88b531459dfa55e7c2",
				Archive: true,
				Binary:  "lightpanda",
			}, nil
		}
	case "windows":
		if runtime.GOARCH == "amd64" {
			return &browserSpec{
				Key: "windows-amd64", Version: browserWindowsVersion,
				URL: "https://github.com/qidiai/lightpanda-windows-port/releases/download/" +
					browserWindowsVersion + "/lightpanda-portable-windows-x64.zip",
				Size:    30754781,
				SHA256:  "7e29b7bdcc8f48290cc8138e796fccdd6fae2b8666b8f1ab30495ca7d70a3b3c",
				Archive: true,
				Binary:  "lightpanda.exe",
			}, nil
		}
	}
	return nil, fmt.Errorf("no Lightpanda build for %s/%s", runtime.GOOS, runtime.GOARCH)
}

// browserDir is the install location, next to the downloaded CPython runtime.
func browserDir() string { return filepath.Join(configDir(), "browser") }

// browserHTTPCacheDir is where Lightpanda keeps its HTTP cache. Mirrors
// g4f.requests.cdp.get_http_cache_dir().
func browserHTTPCacheDir() string {
	return filepath.Join(configDir(), "cache", "http_cache")
}

// browserServeArgs builds the `lightpanda serve` arguments shared by the
// foreground command and the automatic headless startup. The extra flags keep
// the browser from reaching private networks, load iframes (needed by the
// cookie fetchers) and cache HTTP responses.
//
// No --user-agent/--user-agent-suffix is passed: the g4f-copilot fork already
// sends a full Chrome UA (Chrome/152 on Windows) plus matching client hints,
// and --user-agent-suffix would append a second "Chrome/152.0.0.0 Safari/537.36"
// to it. g4f/requests/cdp.py still passes the suffix, which is harmless there
// because it masks the UA over CDP anyway.
func browserServeArgs(host string, port int) []string {
	return []string{
		"serve", "--host", host, "--port", strconv.Itoa(port),
		"--block-private-networks",
		"--load-resources", "iframe",
		"--http-cache-dir", browserHTTPCacheDir(),
	}
}

// browserStampPath records which spec is installed, so a version bump
// re-installs on the next run.
func browserStampPath() string { return filepath.Join(browserDir(), ".version") }

// browserInstalledVersion returns the stamp contents ("" when not installed).
func browserInstalledVersion() string {
	data, err := os.ReadFile(browserStampPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// installedBrowserBinary returns the installed executable, or an error when
// the browser is not installed for this host.
func installedBrowserBinary() (string, error) {
	spec, err := browserSpecForHost()
	if err != nil {
		return "", err
	}
	exe := filepath.Join(browserDir(), spec.Binary)
	if fi, err := os.Stat(exe); err != nil || fi.IsDir() {
		return "", fmt.Errorf("browser not installed (run: g4f-go browser install)")
	}
	return exe, nil
}

// ensureBrowser returns the installed browser, installing it on first use.
func ensureBrowser() (string, error) {
	if exe, err := installedBrowserBinary(); err == nil {
		return exe, nil
	}
	if err := installBrowser(false); err != nil {
		return "", err
	}
	return installedBrowserBinary()
}

// installBrowser downloads, verifies and installs Lightpanda for this host.
func installBrowser(force bool) error {
	spec, err := browserSpecForHost()
	if err != nil {
		return err
	}
	dir := browserDir()
	exe := filepath.Join(dir, spec.Binary)
	want := spec.Key + " " + spec.Version

	if !force && browserInstalledVersion() == want {
		if fi, err := os.Stat(exe); err == nil && !fi.IsDir() && browserLibInstalled(dir, spec) {
			fmt.Printf("browser: already installed (%s)\n", exe)
			return nil
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	fmt.Printf("browser: installing Lightpanda %s for %s\n", spec.Version, spec.Key)
	fmt.Printf("  %s\n", spec.URL)

	tmp, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	start := time.Now()
	if err := downloadBrowser(tmp, spec.URL, spec.Size); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	fmt.Printf("browser: downloaded %s in %s\n", humanBytes(spec.Size), time.Since(start).Round(time.Second))

	if err := verifyBrowserDownload(tmpName, spec.Size, spec.SHA256); err != nil {
		return err
	}

	if spec.Archive {
		if err := extractBrowserZip(tmpName, dir, spec.Binary); err != nil {
			return err
		}
	} else {
		// Windows cannot rename over an existing file.
		os.Remove(exe)
		if err := os.Rename(tmpName, exe); err != nil {
			return err
		}
	}
	if err := os.Chmod(exe, 0o755); err != nil {
		return err
	}

	if spec.Lib != nil {
		if err := installBrowserLib(dir, spec); err != nil {
			return err
		}
		if err := patchBrowserRunpath(exe); err != nil {
			return err
		}
	}

	out, err := exec.Command(exe, "version").CombinedOutput()
	if err != nil {
		return fmt.Errorf("installed binary failed to run: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	fmt.Printf("browser: %s\n", strings.TrimSpace(string(out)))

	if err := os.WriteFile(browserStampPath(), []byte(want+"\n"), 0o644); err != nil {
		return err
	}
	fmt.Printf("browser: installed to %s\n", exe)
	return nil
}

// downloadBrowser streams the pinned release into f with live progress.
func downloadBrowser(f *os.File, url string, size int64) error {
	resp, err := (&http.Client{}).Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: HTTP %s", resp.Status)
	}
	total := resp.ContentLength
	if total <= 0 {
		total = size
	}
	if _, err := copyWithProgress(f, resp.Body, total, time.Now()); err != nil {
		return fmt.Errorf("download interrupted: %w", err)
	}
	return nil
}

// verifyBrowserDownload checks the pinned size and sha256.
func verifyBrowserDownload(path string, size int64, sha string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if size > 0 {
		fi, err := f.Stat()
		if err != nil {
			return err
		}
		if fi.Size() != size {
			return fmt.Errorf("size mismatch: got %d, expected %d", fi.Size(), size)
		}
	}
	if sha == "" {
		return nil
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, sha) {
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, sha)
	}
	fmt.Println("browser: sha256 verified")
	return nil
}

// installBrowserLib downloads the companion shared library the binary links
// against and places it next to the executable.
func installBrowserLib(dir string, spec *browserSpec) error {
	lib := spec.Lib
	target := filepath.Join(dir, lib.Name)
	if fi, err := os.Stat(target); err == nil && !fi.IsDir() && fi.Size() == lib.Size {
		return nil
	}
	fmt.Printf("browser: fetching %s\n", lib.Name)
	fmt.Printf("  %s\n", lib.URL)

	tmp, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	if err := downloadBrowser(tmp, lib.URL, lib.Size); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := verifyBrowserDownload(tmpName, lib.Size, lib.SHA256); err != nil {
		return err
	}
	os.Remove(target)
	if err := os.Rename(tmpName, target); err != nil {
		return err
	}
	return os.Chmod(target, 0o755)
}

// browserLibInstalled reports whether the spec's companion library is present.
func browserLibInstalled(dir string, spec *browserSpec) bool {
	if spec.Lib == nil {
		return true
	}
	fi, err := os.Stat(filepath.Join(dir, spec.Lib.Name))
	return err == nil && !fi.IsDir()
}

// patchBrowserRunpath rewrites the ELF RUNPATH of the installed binary to
// $ORIGIN so the loader finds libc_v8.so next to the executable. The release
// binary records the absolute path of the build machine, which does not exist
// on user systems. The new value is written in place over the old one, which
// is safe because the string table entry is not referenced by any symbol.
func patchBrowserRunpath(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	off, size, err := elfRunpath(data)
	if err != nil {
		return err
	}
	if off < 0 {
		return nil
	}
	const want = "$ORIGIN"
	if size < len(want)+1 {
		return fmt.Errorf("RUNPATH entry too small for %s", want)
	}
	cur := data[off : off+size]
	if i := bytes.IndexByte(cur, 0); i >= 0 && string(cur[:i]) == want {
		return nil
	}
	for i := range cur {
		cur[i] = 0
	}
	copy(cur, want)
	if err := os.WriteFile(path, data, 0o755); err != nil {
		return err
	}
	fmt.Println("browser: patched RUNPATH to $ORIGIN")
	return nil
}

// elfRunpath locates the DT_RUNPATH string in an ELF64 image, returning its
// file offset and the size of the string table slot. off is -1 when the binary
// has no RUNPATH.
func elfRunpath(data []byte) (off, size int, err error) {
	if len(data) < 64 || !bytes.Equal(data[:4], []byte{0x7f, 'E', 'L', 'F'}) {
		return 0, 0, fmt.Errorf("not an ELF file")
	}
	if data[4] != 2 {
		return 0, 0, fmt.Errorf("unsupported ELF class %d", data[4])
	}
	var le binary.ByteOrder = binary.LittleEndian
	if data[5] == 2 {
		le = binary.BigEndian
	}
	shoff := le.Uint64(data[0x28:])
	shentsize := int(le.Uint16(data[0x3a:]))
	shnum := int(le.Uint16(data[0x3c:]))
	if shoff == 0 || shentsize == 0 || shnum == 0 {
		return 0, 0, fmt.Errorf("ELF has no section headers")
	}

	type section struct {
		typ, link uint32
		off, size uint64
	}
	sections := make([]section, 0, shnum)
	for i := 0; i < shnum; i++ {
		base := int(shoff) + i*shentsize
		if base+64 > len(data) {
			return 0, 0, fmt.Errorf("truncated section header table")
		}
		sections = append(sections, section{
			typ:  le.Uint32(data[base+4:]),
			off:  le.Uint64(data[base+0x18:]),
			size: le.Uint64(data[base+0x20:]),
			link: le.Uint32(data[base+0x28:]),
		})
	}

	const (
		shtDynamic = 6
		dtNull     = 0
		dtRunpath  = 29
	)
	for _, s := range sections {
		if s.typ != shtDynamic || s.link >= uint32(len(sections)) {
			continue
		}
		strtab := sections[s.link]
		if s.off+s.size > uint64(len(data)) || strtab.off+strtab.size > uint64(len(data)) {
			return 0, 0, fmt.Errorf("truncated dynamic section")
		}
		dyn := data[s.off : s.off+s.size]
		for i := 0; i+16 <= len(dyn); i += 16 {
			tag := le.Uint64(dyn[i:])
			if tag == dtNull {
				break
			}
			if tag != dtRunpath {
				continue
			}
			val := le.Uint64(dyn[i+8:])
			if val >= strtab.size {
				return 0, 0, fmt.Errorf("RUNPATH offset out of range")
			}
			start := strtab.off + val
			end := strtab.off + strtab.size
			n := bytes.IndexByte(data[start:end], 0)
			if n < 0 {
				return 0, 0, fmt.Errorf("unterminated RUNPATH string")
			}
			return int(start), n + 1, nil
		}
	}
	return -1, 0, nil
}

// extractBrowserZip unpacks a release zip into dir. The Windows portable zip
// has a single top-level directory which is stripped, so lightpanda.exe lands
// directly in dir next to the runtime DLLs it needs; the Linux zips are flat
// and contain just the executable.
func extractBrowserZip(path, dir, binary string) error {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer zr.Close()

	top := ""
	for _, f := range zr.File {
		name := strings.TrimPrefix(filepath.ToSlash(f.Name), "./")
		if i := strings.IndexByte(name, '/'); i > 0 {
			top = name[:i]
			break
		}
	}

	root := filepath.Clean(dir)
	for _, f := range zr.File {
		rel := strings.TrimPrefix(filepath.ToSlash(f.Name), "./")
		if top != "" {
			rel = strings.TrimPrefix(rel, top+"/")
		}
		rel = strings.Trim(rel, "/")
		if rel == "" || rel == "." {
			continue
		}
		if rel == ".." || strings.HasPrefix(rel, "../") || strings.Contains(rel, "/../") {
			return fmt.Errorf("unsafe path in archive: %s", f.Name)
		}
		target := filepath.Join(root, filepath.FromSlash(rel))
		if !strings.HasPrefix(target, root+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe path in archive: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		// The zip carries no unix modes, so set them explicitly.
		mode := os.FileMode(0o644)
		if filepath.Base(rel) == binary {
			mode = 0o755
		}
		if err := writeZipEntry(f, target, mode); err != nil {
			return err
		}
	}
	return nil
}

// writeZipEntry copies one archive member to target with the given mode.
func writeZipEntry(f *zip.File, target string, mode os.FileMode) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, rc); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Chmod(target, mode)
}

// browserServeOptions holds the parsed `browser serve` flags.
type browserServeOptions struct {
	Host string
	Port int
	Args []string
}

// parseBrowserServeArgs splits the g4f-go flags from the ones forwarded to
// `lightpanda serve`.
func parseBrowserServeArgs(args []string) (*browserServeOptions, error) {
	opts := &browserServeOptions{Host: browserDefaultHost, Port: browserDefaultPort}
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--host" || a == "--port":
			if i+1 >= len(args) {
				return nil, fmt.Errorf("%s requires a value", a)
			}
			i++
			if a == "--host" {
				opts.Host = args[i]
				continue
			}
			p, err := strconv.Atoi(args[i])
			if err != nil {
				return nil, fmt.Errorf("invalid port %q", args[i])
			}
			opts.Port = p
		case strings.HasPrefix(a, "--host="):
			opts.Host = strings.TrimPrefix(a, "--host=")
		case strings.HasPrefix(a, "--port="):
			p, err := strconv.Atoi(strings.TrimPrefix(a, "--port="))
			if err != nil {
				return nil, fmt.Errorf("invalid port %q", a)
			}
			opts.Port = p
		default:
			opts.Args = append(opts.Args, a)
		}
	}
	return opts, nil
}

// runBrowserServe runs the CDP server in the foreground. Port 0 picks a free
// port, which is what the automatic headless startup uses.
func runBrowserServe(ctx context.Context, args []string) int {
	opts, err := parseBrowserServeArgs(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 2
	}
	exe, err := ensureBrowser()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	port := opts.Port
	if port == 0 {
		if port, err = freePort(); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
	}

	cmd := exec.CommandContext(ctx, exe,
		append(browserServeArgs(opts.Host, port), opts.Args...)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if err := waitForCDP(opts.Host, port, 15*time.Second); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go: warning:", err)
	} else {
		fmt.Printf("browser: CDP endpoint ready at http://%s:%d\n", opts.Host, port)
	}
	fmt.Printf("browser: point g4f at it with G4F_BROWSER_HOST=%s G4F_BROWSER_PORT=%d\n", opts.Host, port)

	err = cmd.Wait()
	if err == nil {
		return 0
	}
	if ctx.Err() != nil {
		return 130
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	fmt.Fprintln(os.Stderr, "g4f-go:", err)
	return 1
}

// browserStatus prints the install state of the browser.
func browserStatus() int {
	spec, err := browserSpecForHost()
	if err != nil {
		fmt.Printf("browser:     unsupported on %s/%s\n", runtime.GOOS, runtime.GOARCH)
		return 1
	}
	fmt.Printf("browser:     Lightpanda %s (%s)\n", spec.Version, spec.Key)
	fmt.Printf("install dir: %s\n", browserDir())
	exe := filepath.Join(browserDir(), spec.Binary)
	if fi, err := os.Stat(exe); err == nil && !fi.IsDir() {
		fmt.Printf("binary:      %s (%s)\n", exe, humanBytes(fi.Size()))
		if out, err := exec.Command(exe, "version").Output(); err == nil {
			fmt.Printf("version:     %s\n", strings.TrimSpace(string(out)))
		}
	} else {
		fmt.Println("binary:      not installed (run: g4f-go browser install)")
	}
	if v := browserInstalledVersion(); v != "" {
		fmt.Printf("stamp:       %s\n", v)
	}
	return 0
}

// uninstallBrowser removes the browser install directory.
func uninstallBrowser() int {
	dir := browserDir()
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		fmt.Printf("browser: not installed (%s)\n", dir)
		return 0
	}
	if err := os.RemoveAll(dir); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	fmt.Printf("browser: removed %s\n", dir)
	return 0
}

// runBrowserCommand handles `g4f-go browser [install|serve|status|path|uninstall]`.
func runBrowserCommand(ctx context.Context, args []string) int {
	if len(args) == 0 {
		printBrowserHelp()
		return 0
	}
	switch args[0] {
	case "help", "--help", "-h":
		printBrowserHelp()
		return 0
	case "install":
		force := false
		for _, a := range args[1:] {
			if a == "--force" || a == "-f" {
				force = true
			}
		}
		if err := installBrowser(force); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
		return 0
	case "serve", "run", "start":
		return runBrowserServe(ctx, args[1:])
	case "status":
		return browserStatus()
	case "path":
		exe, err := installedBrowserBinary()
		if err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
		fmt.Println(exe)
		return 0
	case "uninstall", "remove":
		return uninstallBrowser()
	}
	fmt.Fprintf(os.Stderr, "g4f-go: unknown browser subcommand %q\n", args[0])
	printBrowserHelp()
	return 2
}

// printBrowserHelp shows the `g4f-go browser` usage.
func printBrowserHelp() {
	version := browserVersion
	if spec, err := browserSpecForHost(); err == nil {
		version = spec.Version
	}
	fmt.Printf(`g4f-go browser - manage the headless browser g4f drives over CDP

Usage:
  g4f-go browser install [--force]   download and install Lightpanda %s
  g4f-go browser serve [flags]       run the CDP server in the foreground
  g4f-go browser status              show install state
  g4f-go browser path                print the installed binary path
  g4f-go browser uninstall           remove the installed browser

Serve flags (anything else is forwarded to `+"`lightpanda serve`"+`):
  --host <HOST>   bind address (default %s)
  --port <PORT>   CDP port (default %d, 0 picks a free port)

Lightpanda always runs with --block-private-networks, --load-resources iframe
and an HTTP cache under %s. The g4f-copilot build already sends a Chrome user
agent, so no UA flag is passed.

When g4f runs headless and no CDP endpoint is configured, g4f-go starts the
installed browser automatically and points g4f at it. Set G4F_BROWSER_PORT to
use an existing endpoint instead, or G4F_BROWSER_HEADLESS=false to opt out.

Installed to: %s
`, version, browserDefaultHost, browserDefaultPort, browserHTTPCacheDir(), browserDir())
}

// headlessActive reports whether g4f will run its browser headless. g4f
// defaults to headless (BrowserConfig.headless = True) and only turns it off
// via --no-headless or G4F_BROWSER_HEADLESS=false.
func headlessActive(args []string) bool {
	if v := os.Getenv("G4F_BROWSER_HEADLESS"); v != "" {
		return isTruthy(v)
	}
	for _, a := range args {
		if a == "--no-headless" {
			return false
		}
	}
	return true
}

// isTruthy mirrors g4f's env parsing for boolean flags.
func isTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// browserAutoStartPidPath records the PID of the browser g4f-go started, so a
// leaked instance (e.g. after a hard kill) is reclaimed on the next run.
func browserAutoStartPidPath() string { return filepath.Join(browserDir(), ".autostart.pid") }

// reapStaleBrowser kills a previously auto-started browser that outlived its
// g4f-go parent. Only PIDs recorded in the pid file are touched.
func reapStaleBrowser() {
	data, err := os.ReadFile(browserAutoStartPidPath())
	if err != nil {
		return
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		os.Remove(browserAutoStartPidPath())
		return
	}
	if p, err := os.FindProcess(pid); err == nil {
		p.Kill()
	}
	os.Remove(browserAutoStartPidPath())
}

// browserAutoStartEnv starts the installed browser when g4f will run headless
// and no CDP endpoint was configured, then returns the env vars that point g4f
// at it plus a cleanup func. It is a no-op (nil, nil) when the browser is not
// installed, an endpoint is already configured, or headless mode is off.
func browserAutoStartEnv(args []string) ([]string, func()) {
	if !headlessActive(args) {
		return nil, nil
	}
	if os.Getenv("G4F_BROWSER_PORT") != "" {
		return nil, nil
	}
	// Only the CDP transport can use an external browser process.
	if mode := os.Getenv("G4F_BROWSER_MODE"); mode != "" && mode != "cdp" {
		return nil, nil
	}
	exe, err := installedBrowserBinary()
	if err != nil {
		return nil, nil
	}
	reapStaleBrowser()
	port, err := freePort()
	if err != nil {
		return nil, nil
	}

	cmd := exec.Command(exe, browserServeArgs(browserDefaultHost, port)...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return nil, nil
	}
	pidPath := browserAutoStartPidPath()
	os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)+"\n"), 0o644)

	stop := func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		cmd.Wait()
		os.Remove(pidPath)
	}
	if err := waitForCDP(browserDefaultHost, port, 10*time.Second); err != nil {
		stop()
		return nil, nil
	}
	fmt.Printf("browser: started Lightpanda on %s:%d (headless)\n", browserDefaultHost, port)
	return []string{
		"G4F_BROWSER_HOST=" + browserDefaultHost,
		"G4F_BROWSER_PORT=" + strconv.Itoa(port),
	}, stop
}

// waitForCDP polls the CDP endpoint until it answers or the timeout expires.
func waitForCDP(host string, port int, timeout time.Duration) error {
	url := fmt.Sprintf("http://%s:%d/json/version", host, port)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("CDP endpoint %s did not become ready within %s", url, timeout)
}

// freePort asks the OS for an unused local port.
func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}
