//go:build linux

package main

import "embed"

//go:embed embed/linux
var embedRuntimeArchive embed.FS
