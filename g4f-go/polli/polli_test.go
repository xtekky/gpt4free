package polli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCreateDefaults(t *testing.T) {
	o, err := parseCreate(nil)
	if err != nil {
		t.Fatal(err)
	}
	if o.template != "pollinations" || o.timeout != 600 {
		t.Fatalf("defaults = %q/%d", o.template, o.timeout)
	}
	if o.desktop || o.connect || o.noSetup {
		t.Fatalf("unexpected defaults: %+v", o)
	}
}

func TestParseCreateFlagsAndPositionals(t *testing.T) {
	o, err := parseCreate([]string{
		"--desktop", "--connect", "--template", "pollinations-large",
		"--timeout", "3600", "--g4f-go-version", "1.2.3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !o.desktop || !o.connect {
		t.Fatalf("flags not set: %+v", o)
	}
	if o.template != "pollinations-large" || o.timeout != 3600 {
		t.Fatalf("template/timeout = %q/%d", o.template, o.timeout)
	}
	if o.version != "1.2.3" {
		t.Fatalf("version = %q", o.version)
	}

	o, err = parseCreate([]string{"pollinations-azure", "120", "--desktop"})
	if err != nil {
		t.Fatal(err)
	}
	if o.template != "pollinations-azure" || o.timeout != 120 || !o.desktop {
		t.Fatalf("positionals = %+v", o)
	}
}

func TestParseCreateRejectsBadValues(t *testing.T) {
	for _, args := range [][]string{
		{"--timeout", "-1"},
		{"--timeout", "soon"},
		{"--template"},
		{"--unknown"},
		{"a", "b", "c"},
		{"--local", "--remote"},
	} {
		if _, err := parseCreate(args); err == nil {
			t.Fatalf("parseCreate(%v) succeeded, want error", args)
		}
	}
}

func TestSandboxIDPattern(t *testing.T) {
	out := "Sandbox abc-123 created\nid: abc-123\n"
	m := sandboxIDPattern.FindStringSubmatch(out)
	if len(m) != 2 || m[1] != "abc-123" {
		t.Fatalf("parsed id = %v", m)
	}

	if validID("abc-123") == false {
		t.Fatal("validID rejected a valid id")
	}
	for _, id := range []string{"", "ABC", "a_b", "a.b", "../x"} {
		if validID(id) {
			t.Errorf("validID(%q) = true", id)
		}
	}
}

func TestEmbeddedSandboxAssets(t *testing.T) {
	if len(sandboxCLI) == 0 || !strings.HasPrefix(string(sandboxCLI), "#!/usr/bin/env bash") {
		t.Fatal("embedded sandbox-cli is missing or not a bash script")
	}
	if len(bootstrapScript) == 0 || !strings.HasPrefix(string(bootstrapScript), "#!/usr/bin/env bash") {
		t.Fatal("embedded polli bootstrap is missing or not a bash script")
	}
}

func TestEmbeddedSandboxAssetsMatchBin(t *testing.T) {
	for _, asset := range []struct {
		name string
		data []byte
	}{
		{"sandbox-cli", sandboxCLI},
		{"polli-sandbox-bootstrap.sh", bootstrapScript},
		{"patch-xfce-menu.py", patchXfceMenu},
		{"sandbox-discover", sandboxDiscover},
		{"sandbox-docs", sandboxDocs},
		{"start-browser", startBrowser},
	} {
		t.Run(asset.name, func(t *testing.T) {
			source, err := os.ReadFile(filepath.Join("..", "..", "bin", asset.name))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(asset.data, source) {
				t.Fatalf("embedded %s differs from bin/%s; copy the updated file to g4f-go/polli/assets/", asset.name, asset.name)
			}
		})
	}
}

func TestDesktopFlag(t *testing.T) {
	if desktopFlag(false) != "" {
		t.Fatal("desktopFlag(false) should be empty")
	}
	if desktopFlag(true) != " --desktop" {
		t.Fatalf("desktopFlag(true) = %q", desktopFlag(true))
	}
}
