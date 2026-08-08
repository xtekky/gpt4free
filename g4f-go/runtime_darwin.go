//go:build darwin

package main

import "embed"

//go:embed embed/darwin
var embedRuntimeArchive embed.FS
