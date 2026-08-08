//go:build freebsd || openbsd || netbsd || dragonfly

package main

import "embed"

//go:embed embed/bsd
var embedRuntimeArchive embed.FS
