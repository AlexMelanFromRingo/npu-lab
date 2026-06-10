#!/usr/bin/env bash
# Installs Android SDK 35 + NDK 26 + build-tools 35 + cmake 3.22 into ~/Android/sdk.
# Idempotent: skips packages that are already present.

set -euo pipefail

SDK_ROOT="${ANDROID_HOME:-$HOME/Android/sdk}"
CMDLINE_URL="https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip"

mkdir -p "$SDK_ROOT/cmdline-tools"

if [ ! -d "$SDK_ROOT/cmdline-tools/latest/bin" ]; then
    echo "[1/3] Downloading Android cmdline-tools..."
    tmp=$(mktemp -d)
    curl -fsSL "$CMDLINE_URL" -o "$tmp/cmdline.zip"
    unzip -q "$tmp/cmdline.zip" -d "$tmp/extract"
    mkdir -p "$SDK_ROOT/cmdline-tools/latest"
    mv "$tmp/extract/cmdline-tools/"* "$SDK_ROOT/cmdline-tools/latest/"
    rm -rf "$tmp"
else
    echo "[1/3] cmdline-tools already installed"
fi

export ANDROID_HOME="$SDK_ROOT"
export PATH="$SDK_ROOT/cmdline-tools/latest/bin:$SDK_ROOT/platform-tools:$PATH"

echo "[2/3] Accepting licenses..."
yes | sdkmanager --licenses >/dev/null 2>&1 || true

echo "[3/3] Installing SDK packages..."
sdkmanager \
    "platform-tools" \
    "platforms;android-35" \
    "build-tools;35.0.0" \
    "ndk;26.3.11579264" \
    "cmake;3.22.1"

cat <<EOF

✓ Android SDK ready at $SDK_ROOT

To make persistent in your shell, add to ~/.bashrc:
    export ANDROID_HOME=$SDK_ROOT
    export PATH=\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$PATH

Next:
    1. Install QNN SDK from QPM (see docs/04-qnn-sdk-setup.md)
    2. Copy libQnn*.so into android-app/app/src/main/jniLibs/arm64-v8a/
    3. Set qnn.sdk.root in android-app/local.properties
    4. cd android-app && ./gradlew :app:assembleDebug
EOF
