package main

import (
	_ "embed"
	"fmt"
)

//go:embed runtime.json
var embeddedManifest []byte

func init() {
	// Validate at startup: a malformed runtime.json would otherwise only
	// surface on first download.
	if _, err := parseRuntimeManifest(embeddedManifest); err != nil {
		panic(fmt.Sprintf("g4f-go: embedded runtime.json is invalid: %v", err))
	}
}

// readRuntimeManifest is overridden by the embedded copy so releases work
// without a runtime.json next to the binary.
func readRuntimeManifest() (*RuntimeManifest, error) {
	return parseRuntimeManifest(embeddedManifest)
}
