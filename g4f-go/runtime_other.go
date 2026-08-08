//go:build !linux && !windows && !darwin && !freebsd && !openbsd && !netbsd && !dragonfly

package main

import "embed"

//go:embed embed/other
var embedRuntimeArchive embed.FS
