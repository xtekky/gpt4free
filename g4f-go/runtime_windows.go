//go:build windows

package main

import "embed"

//go:embed embed/windows
var embedRuntimeArchive embed.FS
