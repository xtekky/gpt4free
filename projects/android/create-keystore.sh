#!/usr/bin/env bash
# Create the stable Android release keystore once and reuse it for every build.
#
#   ./create-keystore.sh [path]     # default: ~/keystores/g4f-release.jks
#
# Keystores are password protected; if the file already exists this script
# refuses to touch it (keytool would fail with "keystore password was
# incorrect" because it cannot open it with a new password). Back up the file
# and the password: losing them means every future update needs an uninstall.
set -euo pipefail

KEYSTORE="${1:-$HOME/keystores/g4f-release.jks}"
ALIAS="${G4F_KEY_ALIAS:-g4f}"

if [[ -e "$KEYSTORE" ]]; then
    echo "Keystore already exists: $KEYSTORE" >&2
    echo "Reuse it, or delete it yourself if nothing has been signed with it yet." >&2
    echo "Check it with: keytool -list -keystore '$KEYSTORE'" >&2
    exit 1
fi

KEYTOOL="keytool"
if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/keytool" ]]; then
    KEYTOOL="$JAVA_HOME/bin/keytool"
elif ! command -v keytool >/dev/null 2>&1; then
    echo "keytool not found; install a JDK 17 or set JAVA_HOME" >&2
    exit 1
fi

mkdir -p "$(dirname "$KEYSTORE")"

read -r -s -p "New keystore password: " pass
echo
read -r -s -p "Repeat password: " pass2
echo
if [[ "$pass" != "$pass2" ]]; then
    echo "Passwords do not match" >&2
    exit 1
fi
if [[ ${#pass} -lt 6 ]]; then
    echo "Use at least 6 characters (keytool minimum)" >&2
    exit 1
fi

# Same store and key password: PKCS12 keystores (the JDK default) cannot hold
# two different passwords, and apksigner/APK signing needs both to match.
"$KEYTOOL" -genkeypair \
    -keystore "$KEYSTORE" \
    -alias "$ALIAS" \
    -keyalg RSA -keysize 2048 -validity 10000 \
    -storepass "$pass" -keypass "$pass" \
    -dname "CN=G4F, O=gpt4free, C=DE"

"$KEYTOOL" -list -keystore "$KEYSTORE" -storepass "$pass" >/dev/null

echo
echo "Created: $KEYSTORE (alias: $ALIAS)"
echo "Back it up together with the password."
echo
echo "Use it:"
echo "  export G4F_KEYSTORE=$KEYSTORE"
echo "  export G4F_KEYSTORE_PASSWORD='<the password you just entered>'"
echo "  export G4F_KEY_ALIAS=$ALIAS"
echo
echo "GitHub secret ANDROID_KEYSTORE_BASE64:"
base64 -w0 "$KEYSTORE" 2>/dev/null || base64 "$KEYSTORE"
echo
