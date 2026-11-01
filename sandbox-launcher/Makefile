.PHONY: all build clean dev test vet

all: test build

# Cross-compile every supported OS/arch and zip releases into dist/
build:
	./build-all.sh

clean:
	rm -rf dist

test:
	go test ./...

vet:
	go vet ./...

# Single-platform dev build (fast iteration)
dev:
	go build -o g4f-sandbox .
