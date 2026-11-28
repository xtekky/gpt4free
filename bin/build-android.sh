#!/usr/bin/env bash
#
# build-android.sh — install the Android build toolchain and build the g4f APK
# inside the gpt4free Docker container.
#
# Usage (run inside the container, e.g. via `docker compose run`):
#   ./bin/build-android.sh          install toolchain + build the APK (default)
#   ./bin/build-android.sh install  # install the toolchain only
#   ./bin/build-android.sh build    # build the APK (installs toolchain first)
#   ./bin/build-android.sh clean    # remove the installed toolchain
#
# The gpt4free checkout is expected at /app (the compose bind mount), so the
# Android project lives at /app/projects/android and the APK lands in
# /app/projects/android/app/build/outputs/apk/.
#
# Environment:
#   ANDROID_TOOLS_DIR   install dir for JDK/Gradle/SDK (default: /opt/android-tools)
#   ANDROID_HOME        Android SDK dir (default: $ANDROID_TOOLS_DIR/android-sdk)
#   JAVA_HOME           JDK dir (default: $ANDROID_TOOLS_DIR/jdk)
#   GRADLE_HOME         Gradle dir (default: $ANDROID_TOOLS_DIR/gradle-8.7)
#   G4F_VERSION         app version name (default: 2.0.0)
#   BUILD_VARIANT       Debug|Release (default: Debug)
#   G4F_KEYSTORE / G4F_KEYSTORE_PASSWORD / G4F_KEY_ALIAS / G4F_KEY_PASSWORD
#                       release signing (optional; debug key is used otherwise)
#
# Toolchain versions (pinned, matches projects/android/README.md):
#   JDK 17, Gradle 8.7, Android SDK command-line tools, platform 34,
#   build-tools 34.0.0, NDK 26.3.11579264, CMake 3.22.1

set -euo pipefail

ANDROID_TOOLS_DIR="${ANDROID_TOOLS_DIR:-/opt/android-tools}"
ANDROID_HOME="${ANDROID_HOME:-$ANDROID_TOOLS_DIR/android-sdk}"
JAVA_HOME="${JAVA_HOME:-$ANDROID_TOOLS_DIR/jdk}"
GRADLE_HOME="${GRADLE_HOME:-$ANDROID_TOOLS_DIR/gradle-8.7}"
G4F_VERSION="${G4F_VERSION:-2.0.0}"
BUILD_VARIANT="${BUILD_VARIANT:-Debug}"

ANDROID_PROJECT="/app/projects/android"
GRADLE_VERSION="8.7"
ANDROID_SDK_TOOLS_VERSION="11.0"
ANDROID_PLATFORM="android-34"
BUILD_TOOLS_VERSION="34.0.0"
NDK_VERSION="26.3.11579264"
CMAKE_VERSION="3.22.1"

export DEBIAN_FRONTEND=noninteractive

log()  { printf '[build-android] %s\n' "$*"; }
warn() { printf '[build-android] warning: %s\n' "$*" >&2; }
die()  { printf '[build-android] error: %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing command: $1 (install it first)"
}

# ---------------------------------------------------------------------------
# system dependencies
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
# JDK 17
# ---------------------------------------------------------------------------
install_jdk() {
    if [ -d "$JAVA_HOME/bin" ] && [ -x "$JAVA_HOME/bin/java" ]; then
        log "JDK 17 already present at $JAVA_HOME"
        return 0
    fi
    install_system_deps
    log "JDK 17 provided by openjdk-17-jdk-headless at $(dirname "$(dirname "$(command -v javac)")")"
    # Point JAVA_HOME at the system JDK if our target dir is empty.
    if [ ! -d "$JAVA_HOME" ]; then
        local sys_java
        sys_java="$(command -v java || true)"
        if [ -n "$sys_java" ]; then
            local sys_home
            sys_home="$(dirname "$(dirname "$(readlink -f "$sys_java")")")"
            mkdir -p "$ANDROID_TOOLS_DIR"
            ln -sfn "$sys_home" "$JAVA_HOME"
            log "JAVA_HOME -> $JAVA_HOME ($sys_home)"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Gradle 8.7
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
    # unzip creates gradle-<version>/; rename to our stable GRADLE_HOME.
    if [ -d "$ANDROID_TOOLS_DIR/gradle-$GRADLE_VERSION" ]; then
        mv -f "$ANDROID_TOOLS_DIR/gradle-$GRADLE_VERSION" "$GRADLE_HOME"
    fi
    chmod +x "$GRADLE_HOME/bin/gradle"
    log "Gradle $GRADLE_VERSION installed at $GRADLE_HOME"
}

# ---------------------------------------------------------------------------
# Android SDK command-line tools + platform + build-tools + NDK + CMake
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
        # unzip extracts cmdline-tools/ directly; move it under latest/.
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
    log "installing Android SDK components (platform $ANDROID_PLATFORM, build-tools $BUILD_TOOLS_VERSION, NDK $NDK_VERSION, CMake $CMAKE_VERSION)"
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
        warn "could not auto-accept all licenses (release builds may need manual acceptance)"
}

install_toolchain() {
    mkdir -p "$ANDROID_TOOLS_DIR"
    install_jdk
    install_gradle
    install_android_sdk
    log "Android build toolchain ready:"
    log "  JAVA_HOME=$JAVA_HOME"
    log "  ANDROID_HOME=$ANDROID_HOME"
    log "  GRADLE_HOME=$GRADLE_HOME"
}

clean_toolchain() {
    log "removing Android build toolchain at $ANDROID_TOOLS_DIR"
    rm -rf "$ANDROID_TOOLS_DIR"
    log "toolchain removed"
}

# ---------------------------------------------------------------------------
# build the APK
# ---------------------------------------------------------------------------
build_apk() {
    install_toolchain

    need_cmd git
    if [ ! -f "$ANDROID_PROJECT/gradlew" ]; then
        die "Android project not found at $ANDROID_PROJECT/gradlew (is the repo mounted at /app?)"
    fi

    export JAVA_HOME ANDROID_HOME GRADLE_HOME
    export PATH="$JAVA_HOME/bin:$GRADLE_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/$BUILD_TOOLS_VERSION:$PATH"

    local variant_arg
    case "$BUILD_VARIANT" in
        Debug)  variant_arg="assembleDebug" ;;
        Release) variant_arg="assembleRelease" ;;
        *) die "unknown BUILD_VARIANT: $BUILD_VARIANT (use Debug or Release)" ;;
    esac

    log "building $BUILD_VARIANT APK (G4F_VERSION=$G4F_VERSION)"
    log "  project: $ANDROID_PROJECT"
    log "  command: $variant_arg --no-daemon"

    cd "$ANDROID_PROJECT"
    G4F_VERSION="$G4F_VERSION" "$GRADLE_HOME/bin/gradle" "$variant_arg" --no-daemon \
        -Pg4fKeystore="${G4F_KEYSTORE:-}" \
        -Pg4fKeystorePassword="${G4F_KEYSTORE_PASSWORD:-}" \
        -Pg4fKeyAlias="${G4F_KEY_ALIAS:-g4f}" \
        -Pg4fKeyPassword="${G4F_KEY_PASSWORD:-}"

    local apk
    apk="$ANDROID_PROJECT/app/build/outputs/apk/${BUILD_VARIANT,,}/app-${BUILD_VARIANT,,}.apk"
    if [ -f "$apk" ]; then
        log "APK built: $apk"
        ls -lh "$apk"
    else
        warn "expected APK at $apk was not produced; check the build output above"
    fi
}

# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
case "${1:-build}" in
    install)
        # Delegate to the dedicated install command so the full Android
        # toolchain can be installed on its own, without building an APK.
        exec "$(dirname "$0")/install-android.sh" "$@"
        ;;
    build)
        build_apk
        ;;
    clean)
        clean_toolchain
        ;;
    -h|--help|help)
        sed -n '2,40p' "$0"
        ;;
    *)
        die "unknown command: $1 (use: install | build | clean)"
        ;;
esac
