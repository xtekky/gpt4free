package polli

import _ "embed"

// The sandbox assets are tracked snapshots of bin/. They must be present in a
// clean checkout so go build (including the Docker builder) can embed them.
// polli_test.go checks that the snapshots stay in sync with bin/.
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
