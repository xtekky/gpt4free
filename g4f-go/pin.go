package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// g4fPackage is the PyPI name of the package installed into the runtime.
const g4fPackage = "g4f"

// latestToken unpins a previously pinned g4f version (`g4f-go -v latest ...`).
const latestToken = "latest"

// parseVersionFlag recognizes a leading `-v <version>` pin and strips it from
// args:
//
//	g4f-go -v 1.5.0 client "hi"  ->  pin "1.5.0", rest [client hi]
//	g4f-go -v latest api          ->  pin "",     rest [api]
//	g4f-go -v                     ->  no pin (stays a --version query)
//	g4f-go -v client              ->  no pin ("client" is not a version)
//
// explicit reports whether the flag was present, so callers can distinguish
// "-v latest" (drop the persisted pin) from no flag at all (keep it).
func parseVersionFlag(args []string) (pin string, explicit bool, rest []string) {
	if len(args) < 2 || args[0] != "-v" || !looksLikeVersion(args[1]) {
		return "", false, args
	}
	pin = args[1]
	if pin == latestToken {
		pin = ""
	}
	return pin, true, args[2:]
}

// looksLikeVersion reports whether s can be a g4f package version: a
// digit-leading PEP 440 version ("1.5.0", "0.5.0.post1") or "latest".
func looksLikeVersion(s string) bool {
	if s == latestToken {
		return true
	}
	return s != "" && s[0] >= '0' && s[0] <= '9'
}

// g4fPin strips a leading `-v <version>` pin from args and resolves the
// effective pin. An explicit `-v latest` clears a previously stored pin right
// away; a version pin is persisted by installG4F only once that version is
// actually installed, so the stored pin always matches the installed package.
func g4fPin(binDir string, args []string) (pin string, explicit bool, rest []string) {
	pin, explicit, rest = parseVersionFlag(args)
	if !explicit {
		return readPinnedG4FVersion(binDir), false, rest
	}
	if pin == "" {
		if err := writePinnedG4FVersion(binDir, ""); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go: clearing version pin:", err)
		}
	}
	return pin, true, rest
}

// pinnedVersionStamp is the file recording the g4f version pinned via -v.
func pinnedVersionStamp(binDir string) string {
	return filepath.Join(binDir, ".g4f-runtime", ".g4f-version")
}

// builtinStamp marks a runtime whose g4f came from a local checkout (an
// editable install, as in the sandbox desktop image) instead of PyPI. The
// package is authoritative: the launcher must not replace it with the latest
// PyPI release on api/gui runs.
func builtinStamp(binDir string) string {
	return filepath.Join(binDir, ".g4f-runtime", ".builtin")
}

func builtinG4F(binDir string) bool {
	_, err := os.Stat(builtinStamp(binDir))
	return err == nil
}

// markBuiltinG4F records that the runtime holds a local/editable g4f and that
// it is installed. Explicit PyPI installs and upgrades clear the marker again.
func markBuiltinG4F(binDir string) error {
	dir := filepath.Join(binDir, ".g4f-runtime")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, ".installed"), []byte("g4f installed"), 0o644); err != nil {
		return err
	}
	return os.WriteFile(builtinStamp(binDir), []byte("builtin\n"), 0o644)
}

func clearBuiltinG4F(binDir string) {
	if err := os.Remove(builtinStamp(binDir)); err != nil && !os.IsNotExist(err) {
		fmt.Fprintln(os.Stderr, "g4f-go: clearing builtin marker:", err)
	}
}

// isEditableG4FInstall reports whether a `g4f-go install ...` passthrough
// installs g4f from a local checkout, e.g. `install -e /app/gpt4free[all]`.
// Those installs are treated as the image's builtin package.
func isEditableG4FInstall(args []string) bool {
	if len(args) == 0 || args[0] != "install" {
		return false
	}
	for _, arg := range args[1:] {
		switch {
		case arg == "-e", arg == "--editable":
			return true
		case strings.HasPrefix(arg, "--editable="):
			return true
		}
		if strings.HasPrefix(arg, "-") {
			continue
		}
		name := arg
		if i := strings.IndexByte(name, '['); i >= 0 {
			name = name[:i]
		}
		if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "/") || strings.HasPrefix(name, "~") {
			return true
		}
	}
	return false
}

// readPinnedG4FVersion returns the persisted pin ("" when unpinned).
func readPinnedG4FVersion(binDir string) string {
	data, err := os.ReadFile(pinnedVersionStamp(binDir))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// writePinnedG4FVersion persists pin; an empty pin removes the stamp again.
func writePinnedG4FVersion(binDir, pin string) error {
	stamp := pinnedVersionStamp(binDir)
	if pin == "" {
		if err := os.Remove(stamp); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(stamp), 0o755); err != nil {
		return err
	}
	return os.WriteFile(stamp, []byte(pin+"\n"), 0o644)
}

// g4fSpec is the pip requirement string for installing g4f, pinned or not.
func g4fSpec(pin string) string {
	if pin == "" {
		return g4fPackage + "[slim]"
	}
	return g4fPackage + "[slim]==" + pin
}

// needsInstall reports whether the g4f package has to be (re)installed: it is
// not installed yet, or a different version than the pin is installed.
func needsInstall(binDir, pin string) bool {
	if !g4fIsInstalled(binDir) {
		return true
	}
	return pin != "" && readPinnedG4FVersion(binDir) != pin
}

// syncPinWithInstallSpec keeps the persisted pin consistent with an explicit
// `g4f-go install ...` pip passthrough: a bare "g4f" unpins (fresh install of
// the latest release), "g4f==X.Y.Z" pins to X.Y.Z, anything else leaves the
// pin untouched.
func syncPinWithInstallSpec(binDir string, pipArgs []string) {
	if len(pipArgs) == 0 || pipArgs[0] != "install" {
		return
	}
	for _, arg := range pipArgs[1:] {
		name := arg
		if i := strings.IndexAny(name, "=; \t"); i >= 0 {
			name = name[:i]
		}
		if i := strings.IndexByte(name, '['); i >= 0 {
			name = name[:i]
		}
		if name != g4fPackage {
			continue
		}
		pin := ""
		if i := strings.Index(arg, "=="); i >= 0 {
			pin = arg[i+2:]
		}
		if err := writePinnedG4FVersion(binDir, pin); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go: writing version pin:", err)
		}
		// An explicit PyPI g4f install replaces the image's builtin package.
		clearBuiltinG4F(binDir)
		return
	}
}
