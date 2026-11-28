#!/usr/bin/env bash
#
# install-android.sh — install all Android build requirements inside the
# gpt4free Docker container.
#
# Installs (pinned, matches projects/android/README.md):
#   * system deps: curl, unzip, zip, git, wget, xz-utils, file, OpenJDK 17,
#     32-bit compat libs (needed by the Android NDK toolchain)
#   * JDK 17 (JAVA_HOME)
#   * Gradle 8.7 (GRADLE_HOME)
#   * Android SDK command-line tools, platform 34, build-tools 34.0.0,
#     NDK 26.3.11579264, CMake 3.22.1 (ANDROID_HOME), licenses accepted
#
# Usage (run inside the container, e.g. via `docker compose run`):
#   docker compose run --rm gpt4free /app/bin/install-android.sh
#
# Environment:
#   ANDROID_TOOLS_DIR   install root (default: /opt/android-tools)
#   ANDROID_HOME        Android SDK dir (default: $ANDROID_TOOLS_DIR/android-sdk)
#   JAVA_HOME           JDK dir (default: $ANDROID_TOOLS_DIR/jdk)
#   GRADLE_HOME         Gradle dir (default: $ANDROID_TOOLS_DIR/gradle-8.7)
#
# Idempotent: safe to re-run; already-installed components are skipped.

set -euo pipefail

ANDROID_TOOLS_DIR="${ANDROID_TOOLS_DIR:-/opt/android-tools}"
ANDROID_HOME="${ANDROID_HOME:-$ANDROID_TOOLS_DIR/android-sdk}"
JAVA_HOME="${JAVA_HOME:-$ANDROID_TOOLS_DIR/jdk}"
GRADLE_HOME="${GRADLE_HOME:-$ANDROID_TOOLS_DIR/gradle-8.7}"

export DEBIAN_FRONTEND=noninteractive

GRADLE_VERSION="8.7"
ANDROID_SDK_TOOLS_VERSION="11.0"
ANDROID_PLATFORM="android-34"
BUILD_TOOLS_VERSION="34.0.0"
NDK_VERSION="26.3.11579264"
CMAKE_VERSION="3.22.1"

log()  { printf '[install-android] %s\n' "$*"; }
warn() { printf '[install-android] warning: %s\n' "$*" >&2; }
die()  { printf '[install-android] error: %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing command: $1 (install it first)"
}

# ---------------------------------------------------------------------------
# 1. system dependencies
# ---------------------------------------------------------------------------
install_system_deps() {
    log "installing system build dependencies"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl ca-certificates unzip zip git wget xz-utils file \
        openjdk-17-jdk-headless \
        libc6-i386 lib32ncurses6 lib32stdc++6 \
        >/dev/null 2>&1
    log "system dependencies installed"
}

# ---------------------------------------------------------------------------
# 2. JDK 17
# ---------------------------------------------------------------------------
install_jdk() {
    if [ -d "$JAVA_HOME/bin" ] && [ -x "$JAVA_HOME/bin/java" ]; then
        log "JDK 17 already present at $JAVA_HOME"
        return 0
    fi
    install_system_deps

    # Use the system OpenJDK 17, symlinked to JAVA_HOME for a stable path.
    if ! command -v java >/dev/null 2>&1; then
        die "openjdk-17-jdk-headless did not provide a java binary"
    fi
    local sys_home
    sys_home="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
    mkdir -p "$ANDROID_TOOLS_DIR"
    ln -sfn "$sys_home" "$JAVA_HOME"
    log "JDK 17 installed: JAVA_HOME=$JAVA_HOME ($sys_home)"
}

# ---------------------------------------------------------------------------
# 3. Gradle 8.7
# ---------------------------------------------------------------------------
install_gradle() {
    if [ -x "$GRADLE_HOME/bin/gradle" ]; then
        log "Gradle $GRADLE_VERSION already present at $GRADLE_HOME"
        return 0
    fi
    need_cmd unzip
    local gradle_zip="gradle-${GRADLE_VERSION}-bin.zip"
    local url="https://services.gradle.org/distributions/${gradle_zip}"
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    log "downloading Gradle $GRADLE_VERSION"
    curl -fsSL --retry 3 --retry-delay 2 -o "$tmp/$gradle_zip" "$url"
    mkdir -p "$ANDROID_TOOLS_DIR"
    unzip -qo "$tmp/$gradle_zip" -d "$ANDROID_TOOLS_DIR"
    if [ -d "$ANDROID_TOOLS_DIR/gradle-$GRADLE_VERSION" ]; then
        mv -f "$ANDROID_TOOLS_DIR/gradle-$GRADLE_VERSION" "$GRADLE_HOME"
    fi
    chmod +x "$GRADLE_HOME/bin/gradle"
    log "Gradle $GRADLE_VERSION installed: GRADLE_HOME=$GRADLE_HOME"
}

# ---------------------------------------------------------------------------
# 4. Android SDK + platform + build-tools + NDK + CMake
# ---------------------------------------------------------------------------
install_android_sdk() {
    local cmds_dir="$ANDROID_HOME/cmdline-tools"
    local latest="$cmds_dir/latest"
    if [ -x "$latest/bin/sdkmanager" ]; then
        log "Android SDK command-line tools already present"
    else
        need_cmd unzip
        local sdk_zip="commandlinetools-linux-${ANDROID_SDK_TOOLS_VERSION}_latest.zip"
        local url="https://dl.google.com/android/repository/${sdk_zip}"
        local tmp
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' RETURN
        log "downloading Android SDK command-line tools"
        curl -fsSL --retry 3 --retry-delay 2 -o "$tmp/$sdk_zip" "$url"
        mkdir -p "$cmds_dir"
        unzip -qo "$tmp/$sdk_zip" -d "$cmds_dir"
        if [ -x "$cmds_dir/bin/sdkmanager" ]; then
            mkdir -p "$latest"
            cp -a "$cmds_dir/." "$latest/"
            rm -rf "$cmds_dir"/bin "$cmds_dir"/lib "$cmds_dir"/source.properties "$cmds_dir"/notice.txt
        fi
        chmod +x "$latest/bin/sdkmanager"
        log "Android SDK command-line tools installed"
    fi

    export ANDROID_HOME
    local sm="$latest/bin/sdkmanager"
    log "installing Android SDK components:"
    log "  platform $ANDROID_PLATFORM, build-tools $BUILD_TOOLS_VERSION"
    log "  NDK $NDK_VERSION, CMake $CMAKE_VERSION"
    "$sm" "platforms;$ANDROID_PLATFORM" \
          "build-tools;$BUILD_TOOLS_VERSION" \
          "ndk;$NDK_VERSION" \
          "cmake;$CMAKE_VERSION" >/dev/null 2>&1 || {
        warn "sdkmanager component install failed; retrying with --channel=3"
        "$sm" --channel=3 "platforms;$ANDROID_PLATFORM" \
                    "build-tools;$BUILD_TOOLS_VERSION" \
                    "ndk;$NDK_VERSION" \
                    "cmake;$CMAKE_VERSION" >/dev/null 2>&1 || \
            die "failed to install Android SDK components"
    }
    log "Android SDK components installed"

    log "accepting Android SDK licenses"
    yes | "$sm" --licenses >/dev/null 2>&1 || \
        warn "could not auto-accept all licenses (release signing may need manual acceptance)"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    mkdir -p "$ANDROID_TOOLS_DIR"
    install_jdk
    install_gradle
    install_android_sdk

    log "============================================================"
    log "All Android build requirements installed:"
    log "  JAVA_HOME=$JAVA_HOME"
    log "  ANDROID_HOME=$ANDROID_HOME"
    log "  GRADLE_HOME=$GRADLE_HOME"
    log ""
    log "To build the APK inside the container:"
    log "  docker compose run --rm gpt4free /app/bin/build-android.sh build"
    log "============================================================"
}

main "$@"
