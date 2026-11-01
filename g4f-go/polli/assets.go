package polli

import _ "embed"

// The sandbox assets are copied from bin/ at build time so the Polli command
// can install the same sandbox-cli, helper scripts and desktop bootstrap that
// the container image ships, even when g4f-go runs from a standalone release.
var (
	//go:embed assets/sandbox-cli
	sandboxCLI []byte

	//go:embed assets/polli-sandbox-bootstrap.sh
	bootstrapScript []byte

	//go:embed assets/patch-xfce-menu.py
	patchXfceMenu []byte

	//go:embed assets/sandbox-discover
	sandboxDiscover []byte

	//go:embed assets/sandbox-docs
	sandboxDocs []byte

	//go:embed assets/start-browser
	startBrowser []byte
)
