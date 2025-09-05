#!/bin/bash
# Nuitka build script for g4f
# This script builds g4f executables using Nuitka for different platforms and architectures

set -e

# Default values
PLATFORM=${PLATFORM:-$(uname -s | tr '[:upper:]' '[:lower:]')}
ARCHITECTURE=${ARCHITECTURE:-$(uname -m)}
VERSION=${G4F_VERSION:-0.0.0-dev}
OUTPUT_DIR=${OUTPUT_DIR:-dist}

# Normalize architecture names
case "${ARCHITECTURE}" in
    "x86_64"|"amd64")
        ARCH="x64"
        ;;
    "arm64"|"aarch64")
        ARCH="arm64"
        ;;
    "armv7l"|"armhf")
        ARCH="armv7"
        ;;
    *)
        ARCH="${ARCHITECTURE}"
        ;;
esac

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Building g4f with Nuitka..."
echo "Platform: ${PLATFORM}"
echo "Architecture: ${ARCH} (${ARCHITECTURE})"
echo "Version: ${VERSION}"
echo "Output: ${OUTPUT_DIR}"

# Set output filename based on platform
case "${PLATFORM}" in
    "windows"|"win32")
        OUTPUT_NAME="g4f-windows-${VERSION}-${ARCH}.exe"
        NUITKA_ARGS="--windows-console-mode=attach --onefile"
        ;;
    "darwin"|"macos")
        OUTPUT_NAME="g4f-macos-${VERSION}-${ARCH}"
        NUITKA_ARGS="--macos-create-app-bundle --onefile"
        ;;
    "linux")
        OUTPUT_NAME="g4f-linux-${VERSION}-${ARCH}"
        NUITKA_ARGS="--onefile"
        ;;
    *)
        OUTPUT_NAME="g4f-${PLATFORM}-${VERSION}-${ARCH}"
        NUITKA_ARGS="--onefile"
        ;;
esac

# Basic Nuitka arguments
NUITKA_COMMON_ARGS="
    --standalone
    --output-filename=${OUTPUT_NAME}
    --output-dir=${OUTPUT_DIR}
    --remove-output
    --no-pyi-file
    --assume-yes-for-downloads
    --show-progress
    --show-memory
"

# Platform-specific optimizations
if [[ "${PLATFORM}" == "windows" ]] && [[ -f "projects/windows/icon.ico" ]]; then
    NUITKA_ARGS="${NUITKA_ARGS} --windows-icon-from-ico=projects/windows/icon.ico"
fi

# Build command
echo "Running Nuitka build..."
python -m nuitka ${NUITKA_COMMON_ARGS} ${NUITKA_ARGS} g4f_cli.py

echo "Build completed: ${OUTPUT_DIR}/${OUTPUT_NAME}"

# Verify the build
if [[ -f "${OUTPUT_DIR}/${OUTPUT_NAME}" ]]; then
    echo "✓ Build successful!"
    ls -la "${OUTPUT_DIR}/${OUTPUT_NAME}"
else
    echo "✗ Build failed - output file not found"
    exit 1
fi