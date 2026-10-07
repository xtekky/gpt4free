.PHONY: all fetch bundle build clean vet test

all: fetch bundle build

# Fetch the runtime manifest (validates URLs; runtimes download at first run)
fetch:
	./fetch-python.sh

# Refresh the embedded Python bundle from projects/ (remote_desktop + web)
bundle:
	./sync-bundle.sh

# Cross-compile launchers for every platform and zip releases into dist/
build:
	./build-all.sh

clean:
	rm -rf dist

vet:
	go vet ./...

test:
	go test ./...

# Single-platform dev build (fast iteration)
dev:
	go build -o g4f-go .
