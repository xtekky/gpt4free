package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// configDir mirrors g4f.config.get_config_dir(): ~/.g4f when it exists,
// otherwise the platform config location, falling back to ~/.g4f.
func configDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	defaultDir := filepath.Join(home, ".g4f")
	if _, err := os.Stat(defaultDir); err == nil {
		return defaultDir
	}
	var fallback string
	switch runtime.GOOS {
	case "windows":
		appdata := os.Getenv("APPDATA")
		if appdata == "" {
			appdata = filepath.Join(home, "AppData", "Roaming")
		}
		fallback = filepath.Join(appdata, "g4f")
	case "darwin":
		fallback = filepath.Join(home, "Library", "Application Support", "g4f")
	default:
		fallback = filepath.Join(home, ".config", "g4f")
	}
	if _, err := os.Stat(fallback); err == nil {
		return fallback
	}
	return defaultDir
}

// pythonCacheDir mirrors g4f.config.get_cache_dir(): the central cache
// directory used by the g4f Python package (model lists, scrape caches, ...),
// honoring the G4F_CACHE_DIR override.
func pythonCacheDir() string {
	if d := os.Getenv("G4F_CACHE_DIR"); d != "" {
		return d
	}
	return filepath.Join(configDir(), "cache")
}

// runCacheCommand handles `g4f-go cache [clear]`.
func runCacheCommand(args []string) int {
	if len(args) > 0 && args[0] == "clear" {
		return clearCache()
	}
	fmt.Printf("cache dir: %s\n", pythonCacheDir())
	fmt.Println("usage: g4f-go cache clear   remove all cached g4f data")
	return 0
}

// clearCache removes the contents of the central g4f cache directory.
func clearCache() int {
	dir := pythonCacheDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("cache dir: %s (already empty)\n", dir)
			return 0
		}
		fmt.Fprintln(os.Stderr, "g4f-go:", err)
		return 1
	}
	removed := 0
	for _, entry := range entries {
		if err := os.RemoveAll(filepath.Join(dir, entry.Name())); err != nil {
			fmt.Fprintln(os.Stderr, "g4f-go: removing", entry.Name(), ":", err)
			return 1
		}
		removed++
	}
	fmt.Printf("cleared %d item(s) from %s\n", removed, dir)
	return 0
}
