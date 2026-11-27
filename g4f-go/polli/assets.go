package polli

import _ "embed"

// The sandbox assets are generated from bin/ by g4f-go/sync-assets.sh, which
// runs before builds and tests. They are not tracked source copies.
// polli_test.go checks that the generated files stay in sync with bin/.
var (
	//go:embed assets/sandbox-cli
	sandboxCLI []byte

	//go:embed assets/polli-sandbox-bootstrap.sh
	bootstrapScript []byte

	//go:embed assets/install-desktop.sh
	installDesktopScript []byte

	//go:embed assets/patch-xfce-menu.py
	patchXfceMenu []byte

	//go:embed assets/sandbox-discover
	sandboxDiscover []byte

	//go:embed assets/sandbox-docs
	sandboxDocs []byte

	//go:embed assets/start-browser
	startBrowser []byte
)
