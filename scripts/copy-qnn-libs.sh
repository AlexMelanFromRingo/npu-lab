#!/usr/bin/env bash
# Copies the QNN runtime libraries the APK needs out of a local QAIRT SDK
# install into android-app/app/src/main/jniLibs/arm64-v8a/.
#
# These files are Qualcomm-licensed and are NOT committed to the repo —
# download the SDK (free) per docs/04-qnn-sdk-setup.md, then run this once:
#
#   scripts/copy-qnn-libs.sh [QNN_SDK_ROOT]
#
# Includes the Hexagon-side (DSP6 ELF) libraries: they deliberately go into
# the SAME jniLibs/arm64-v8a folder — see ARCHITECTURE.md §3.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/android-app/app/src/main/jniLibs/arm64-v8a"

QNN_SDK_ROOT="${1:-${QNN_SDK_ROOT:-}}"
if [[ -z "${QNN_SDK_ROOT}" ]]; then
    QNN_SDK_ROOT="$(sed -n 's/^qnn\.sdk\.root=//p' "${REPO_ROOT}/android-app/local.properties" 2>/dev/null || true)"
fi
if [[ -z "${QNN_SDK_ROOT}" || ! -d "${QNN_SDK_ROOT}/lib/aarch64-android" ]]; then
    echo "ERROR: QNN_SDK_ROOT not found (got: '${QNN_SDK_ROOT}')." >&2
    echo "       Pass it as \$1, export it, or set qnn.sdk.root in android-app/local.properties." >&2
    exit 2
fi

HEXAGON_ARCH="${HEXAGON_ARCH:-v81}"   # Snapdragon 8 Elite Gen 5
A64="${QNN_SDK_ROOT}/lib/aarch64-android"
HEX="${QNN_SDK_ROOT}/lib/hexagon-${HEXAGON_ARCH}/unsigned"

mkdir -p "${DEST}"

# Host-side (aarch64) libraries.
for f in libQnnSystem.so libQnnHtp.so \
         "libQnnHtp$(tr '[:lower:]' '[:upper:]' <<<"${HEXAGON_ARCH:0:1}")${HEXAGON_ARCH:1}Stub.so" \
         "libQnnHtp$(tr '[:lower:]' '[:upper:]' <<<"${HEXAGON_ARCH:0:1}")${HEXAGON_ARCH:1}CalculatorStub.so" \
         libQnnHtpProfilingReader.so libQnnHtpNetRunExtensions.so \
         libQnnCpu.so libQnnGpu.so; do
    cp -v "${A64}/${f}" "${DEST}/"
done

# Optional: online-prepare for DLC models (~90 MB). Comment out to slim the APK.
cp -v "${A64}/libQnnHtpPrepare.so" "${DEST}/" || true

# Hexagon-side (DSP6) libraries — yes, into jniLibs/arm64-v8a (see docs).
UP="$(tr '[:lower:]' '[:upper:]' <<<"${HEXAGON_ARCH:0:1}")${HEXAGON_ARCH:1}"
cp -v "${HEX}/libQnnHtp${UP}Skel.so" "${DEST}/"
cp -v "${HEX}/libQnnHtp${UP}.so" "${DEST}/"

echo
echo "Done. $(ls "${DEST}" | wc -l) libraries in ${DEST}"
