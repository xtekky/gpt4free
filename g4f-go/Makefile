.PHONY: all fetch bundle assets build clean vet test dev

all: fetch bundle build

# Fetch the runtime manifest (validates URLs; runtimes download at first run)
fetch:
	./fetch-python.sh

# Refresh the embedded Python bundle from projects/ (remote_desktop + web)
bundle:
	./sync-bundle.sh

# Regenerate the embedded bin/ commands (sandbox-cli, bootstrap, installer, ...)
assets:
	./sync-assets.sh

# Cross-compile launchers for every platform and zip releases into dist/
build: assets
	./build-all.sh

clean:
	rm -rf dist

vet: assets
	go vet ./...

test: assets
	go test ./...

# Single-platform dev build (fast iteration)
dev: assets
	go build -o g4f-go .
