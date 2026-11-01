package main

import (
	"testing"

	"g4f-go/sandbox"
)

func TestDesktopSandboxShortcut(t *testing.T) {
	args := desktopSandboxArgs([]string{"--no-open", "--port", "9090", "--app", "opencode"})
	want := []string{"up", "--no-open", "--port", "9090", "--app", "opencode", "--desktop"}
	if len(args) != len(want) {
		t.Fatalf("args = %v, want %v", args, want)
	}
	for i := range want {
		if args[i] != want[i] {
			t.Fatalf("args = %v, want %v", args, want)
		}
	}
	if !hasSubcommand([]string{"start-desktop-sandbox"}) {
		t.Fatal("shortcut should be recognized as a launcher subcommand")
	}
	// Help is handled without downloading Python or starting a container.
	if code := sandbox.Run(desktopSandboxArgs([]string{"--help"})); code != 0 {
		t.Fatalf("sandbox shortcut help returned %d", code)
	}
}
