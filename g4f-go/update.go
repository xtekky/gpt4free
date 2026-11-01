package main

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// releaseRepo is the GitHub repository that publishes g4f-go releases.
const releaseRepo = "xtekky/gpt4free"

// releaseAPIBase is the GitHub REST endpoint used for release discovery.
const releaseAPIBase = "https://api.github.com/repos/" + releaseRepo

// releaseInfo is the subset of the GitHub release API used by `g4f-go update`.
type releaseInfo struct {
	TagName string         `json:"tag_name"`
	Assets  []releaseAsset `json:"assets"`
}

// releaseAsset is one downloadable release file.
type releaseAsset struct {
	Name               string `json:"name"`
	BrowserDownloadURL string `json:"browser_download_url"`
	Digest             string `json:"digest"`
}

// updateOptions holds the parsed `g4f-go update` flags.
type updateOptions struct {
	Check   bool
	Force   bool
	Version string
}

// runUpdateCommand handles `g4f-go update`: replace this binary with the
// latest published g4f-go release.
func runUpdateCommand(args []string) int {
	opts := updateOptions{}
	for i := 0; i < len(args); i++ {
		switch arg := args[i]; {
		case arg == "--check" || arg == "-c":
			opts.Check = true
		case arg == "--force" || arg == "-f":
			opts.Force = true
		case arg == "--version" || arg == "-V":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "g4f-go: update --version requires a release tag")
				return 2
			}
			i++
			opts.Version = args[i]
		case strings.HasPrefix(arg, "--version="):
			opts.Version = strings.TrimPrefix(arg, "--version=")
		case arg == "help" || arg == "--help" || arg == "-h":
			printUpdateHelp()
			return 0
		default:
			fmt.Fprintf(os.Stderr, "g4f-go: update: unknown option %q\n", arg)
			fmt.Fprintln(os.Stderr, "usage: g4f-go update [--check] [--force] [--version <tag>]")
			return 2
		}
	}
	if err := selfUpdate(opts); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	return 0
}

func printUpdateHelp() {
	fmt.Print(`g4f-go update - update the g4f-go launcher itself

Usage:
  g4f-go update                 install the latest GitHub release
  g4f-go update --check         only report whether an update is available
  g4f-go update --force         reinstall even when already up to date
  g4f-go update --version TAG   install a specific release (e.g. v0.2.0)

The launcher is replaced in place; the downloaded CPython runtime and the
installed g4f package are left untouched. Use ` + "`g4f-go upgrade`" + ` to update
the Python package instead.
`)
}

// selfUpdate resolves, downloads and applies a g4f-go release.
func selfUpdate(opts updateOptions) error {
	tag := opts.Version
	var release *releaseInfo
	var err error
	if tag == "" {
		fmt.Println("update: checking the latest g4f-go release")
		release, err = fetchRelease("latest")
		if err != nil {
			return err
		}
	} else {
		release, err = fetchRelease(tag)
		if err != nil {
			return err
		}
	}
	tag = release.TagName
	if tag == "" {
		return fmt.Errorf("release has no tag")
	}

	current := normalizeVersion(Version)
	next := normalizeVersion(tag)
	if !opts.Force && compareVersions(current, next) >= 0 {
		fmt.Printf("update: g4f-go %s is already up to date (latest %s)\n", current, next)
		return nil
	}
	if opts.Check {
		fmt.Printf("update: g4f-go %s -> %s is available\n", current, next)
		return nil
	}

	asset, err := findReleaseAsset(release.Assets, next)
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	if resolved, rerr := filepath.EvalSymlinks(exe); rerr == nil {
		exe = resolved
	}
	binName := releaseBinaryName(runtime.GOOS)

	fmt.Printf("update: installing g4f-go %s for %s/%s\n", next, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("  %s\n", asset.BrowserDownloadURL)

	dir := filepath.Dir(exe)
	archive, err := os.CreateTemp(dir, ".g4f-go-update-*.zip")
	if err != nil {
		return err
	}
	archiveName := archive.Name()
	defer os.Remove(archiveName)
	if err := downloadRelease(archive, asset.BrowserDownloadURL); err != nil {
		archive.Close()
		return err
	}
	if err := archive.Close(); err != nil {
		return err
	}
	if err := verifyReleaseDigest(archiveName, asset.Digest); err != nil {
		return err
	}

	newExe, err := extractReleaseBinary(archiveName, binName, dir)
	if err != nil {
		return err
	}
	defer os.Remove(newExe)
	if err := os.Chmod(newExe, 0o755); err != nil {
		return err
	}
	if err := replaceExecutable(exe, newExe); err != nil {
		return err
	}
	fmt.Printf("update: g4f-go updated to %s\n", next)
	fmt.Printf("  %s\n", exe)
	return nil
}

// fetchRelease returns one release ("latest" or a tag) from GitHub.
func fetchRelease(tag string) (*releaseInfo, error) {
	endpoint := releaseAPIBase + "/releases/latest"
	if tag != "" && tag != "latest" {
		endpoint = releaseAPIBase + "/releases/tags/" + tag
	}
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "g4f-go/"+Version)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("checking releases: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("checking releases: HTTP %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}
	var release releaseInfo
	if err := json.NewDecoder(resp.Body).Decode(&release); err != nil {
		return nil, fmt.Errorf("decoding release: %w", err)
	}
	return &release, nil
}

// findReleaseAsset picks the g4f-go zip for the current OS/arch.
func findReleaseAsset(assets []releaseAsset, version string) (*releaseAsset, error) {
	want := releaseAssetName(version, runtime.GOOS, runtime.GOARCH)
	for i := range assets {
		if assets[i].Name == want {
			return &assets[i], nil
		}
	}
	// Older releases may have been built with a different version spelling.
	suffix := fmt.Sprintf("-%s-%s.zip", runtime.GOOS, runtime.GOARCH)
	for i := range assets {
		name := assets[i].Name
		if strings.HasPrefix(name, "g4f-go-") && strings.HasSuffix(name, suffix) {
			return &assets[i], nil
		}
	}
	var names []string
	for _, asset := range assets {
		if strings.HasSuffix(asset.Name, ".zip") {
			names = append(names, asset.Name)
		}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("release %s has no zip asset (no build for %s/%s?)", version, runtime.GOOS, runtime.GOARCH)
	}
	return nil, fmt.Errorf("release %s has no build for %s/%s (assets: %s)",
		version, runtime.GOOS, runtime.GOARCH, strings.Join(names, ", "))
}

// releaseAssetName is the archive name produced by build-all.sh.
func releaseAssetName(version, goos, goarch string) string {
	return fmt.Sprintf("g4f-go-%s-%s-%s.zip", normalizeVersion(version), goos, goarch)
}

// releaseBinaryName is the executable name inside a release archive.
func releaseBinaryName(goos string) string {
	if goos == "windows" {
		return "g4f-go.exe"
	}
	return "g4f-go"
}

// normalizeVersion strips a leading "v" and surrounding whitespace.
func normalizeVersion(v string) string {
	return strings.TrimPrefix(strings.TrimSpace(v), "v")
}

// compareVersions compares dotted numeric versions ("0.1.0", "v1.2.3").
// Non-numeric suffixes are ignored, so "0.2.0-rc1" compares as "0.2.0".
func compareVersions(a, b string) int {
	as := versionParts(a)
	bs := versionParts(b)
	for i := 0; i < len(as) || i < len(bs); i++ {
		var av, bv int
		if i < len(as) {
			av = as[i]
		}
		if i < len(bs) {
			bv = bs[i]
		}
		if av < bv {
			return -1
		}
		if av > bv {
			return 1
		}
	}
	return 0
}

func versionParts(v string) []int {
	v = normalizeVersion(v)
	var parts []int
	for _, field := range strings.Split(v, ".") {
		n := 0
		for i := 0; i < len(field); i++ {
			if field[i] < '0' || field[i] > '9' {
				break
			}
			n = n*10 + int(field[i]-'0')
		}
		parts = append(parts, n)
	}
	return parts
}

// downloadRelease streams url into f.
func downloadRelease(f *os.File, url string) error {
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: HTTP %s", resp.Status)
	}
	_, err = io.Copy(f, resp.Body)
	if err != nil {
		return fmt.Errorf("download interrupted: %w", err)
	}
	return nil
}

// verifyReleaseDigest checks the GitHub asset digest when it is published.
func verifyReleaseDigest(path, digest string) error {
	digest = strings.TrimPrefix(strings.TrimSpace(digest), "sha256:")
	if digest == "" {
		fmt.Println("update: release has no published digest; skipping sha256 check")
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, digest) {
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, digest)
	}
	fmt.Println("update: sha256 verified")
	return nil
}

// extractReleaseBinary extracts the named executable from a release zip into
// a temporary file next to the current binary (same filesystem for rename).
func extractReleaseBinary(archivePath, binName, dir string) (string, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	zr, err := zip.NewReader(f, info.Size())
	if err != nil {
		return "", err
	}
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || filepath.Base(f.Name) != binName {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", err
		}
		out, err := os.CreateTemp(dir, ".g4f-go-update-bin-*")
		if err != nil {
			rc.Close()
			return "", err
		}
		_, copyErr := io.Copy(out, rc)
		rc.Close()
		if closeErr := out.Close(); copyErr == nil {
			copyErr = closeErr
		}
		if copyErr != nil {
			os.Remove(out.Name())
			return "", copyErr
		}
		return out.Name(), nil
	}
	return "", fmt.Errorf("release archive does not contain %s", binName)
}

// replaceExecutable atomically replaces target with source.
func replaceExecutable(target, source string) error {
	if runtime.GOOS == "windows" {
		// Windows cannot replace a running executable. Move it aside first,
		// then put the new binary in place. The old file is removed on the
		// next update if the current process still has it open.
		old := target + ".old"
		_ = os.Remove(old)
		if err := os.Rename(target, old); err != nil {
			return fmt.Errorf("replacing %s: %w (close g4f-go and retry)", target, err)
		}
		if err := os.Rename(source, target); err != nil {
			_ = os.Rename(old, target)
			return err
		}
		_ = os.Remove(old)
		return nil
	}
	if err := os.Rename(source, target); err != nil {
		return err
	}
	return nil
}

// runUpgradeCommand handles `g4f-go upgrade`: update the Python g4f package in
// the downloaded runtime. With a -v pin it installs exactly that version;
// otherwise it upgrades to the latest release and clears any stored pin.
func runUpgradeCommand(binDir, py, pin string, pinExplicit bool) int {
	start := time.Now()
	if pin != "" && pinExplicit {
		if err := installG4F(binDir, py, pin, start); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go:", err)
			return 1
		}
		return 0
	}
	if err := upgradeG4F(binDir, py, start); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if err := writeStamp(binDir); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if err := writePinnedG4FVersion(binDir, ""); err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go: clearing version pin:", err)
		return 1
	}
	return 0
}

// writeStamp records a successful g4f package installation.
func writeStamp(binDir string) error {
	stamp := filepath.Join(binDir, ".g4f-runtime", ".installed")
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		return err
	}
	return os.WriteFile(stamp, []byte("g4f installed"), 0o644)
}

// uninstallOptions holds the parsed `g4f-go uninstall` flags.
type uninstallOptions struct {
	KeepBinary bool
	KeepData   bool
	DryRun     bool
}

// uninstallTargets returns the managed directories removed by uninstall.
func uninstallTargets(binDir string) []string {
	seen := map[string]bool{}
	var dirs []string
	for _, dir := range []string{binDir, browserDir(), pythonCacheDir()} {
		if dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		dirs = append(dirs, dir)
	}
	return dirs
}

// runUninstallCommand handles `g4f-go uninstall`: remove the launcher and the
// data it manages (downloaded runtime, bundle, browser, cache).
func runUninstallCommand(args []string) int {
	opts := uninstallOptions{}
	for _, arg := range args {
		switch arg {
		case "--keep-binary":
			opts.KeepBinary = true
		case "--keep-data":
			opts.KeepData = true
		case "--dry-run", "-n":
			opts.DryRun = true
		case "help", "--help", "-h":
			printUninstallHelp()
			return 0
		default:
			fmt.Fprintf(os.Stderr, "g4f-go: uninstall: unknown option %q\n", arg)
			fmt.Fprintln(os.Stderr, "usage: g4f-go uninstall [--keep-binary] [--keep-data] [--dry-run]")
			return 2
		}
	}

	binDir := installDir()
	if !opts.KeepData {
		for _, dir := range uninstallTargets(binDir) {
			if _, err := os.Stat(dir); os.IsNotExist(err) {
				fmt.Printf("uninstall: %s (not present)\n", dir)
				continue
			}
			if opts.DryRun {
				fmt.Printf("uninstall: would remove %s\n", dir)
				continue
			}
			if err := os.RemoveAll(dir); err != nil {
				fmt.Fprintf(os.Stderr, "g4f-go: removing %s: %v\n", dir, err)
				return 1
			}
			fmt.Printf("uninstall: removed %s\n", dir)
		}
	}

	if opts.KeepBinary {
		fmt.Println("uninstall: kept the g4f-go binary (--keep-binary)")
		return 0
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	if resolved, rerr := filepath.EvalSymlinks(exe); rerr == nil {
		exe = resolved
	}
	if opts.DryRun {
		fmt.Printf("uninstall: would remove %s\n", exe)
		return 0
	}
	if err := os.Remove(exe); err != nil {
		fmt.Fprintf(os.Stderr, "g4f-go: removing %s: %v\n", exe, err)
		fmt.Fprintln(os.Stderr, "g4f-go: close any running g4f-go process and remove the file manually")
		return 1
	}
	fmt.Printf("uninstall: removed %s\n", exe)
	fmt.Println("uninstall: done")
	return 0
}

func printUninstallHelp() {
	fmt.Print(`g4f-go uninstall - remove g4f-go and its managed data

Usage:
  g4f-go uninstall                 remove the binary, runtime, browser and caches
  g4f-go uninstall --keep-binary   keep the g4f-go executable
  g4f-go uninstall --keep-data     keep the runtime, browser and caches
  g4f-go uninstall --dry-run       show what would be removed

User configuration outside the managed directories (for example cookies or
model settings in the g4f config directory) is left untouched. To remove a
single Python package instead, use: g4f-go uninstall g4f
`)
}
