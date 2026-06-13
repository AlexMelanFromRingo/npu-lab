#!/usr/bin/env bash
# Copies the QNN runtime libraries the APK needs out of a local QAIRT SDK
# install into android-app/app/src/main/jniLibs/arm64-v8a/.
#
# These files are Qualcomm-licensed and are NOT committed to the repo —
# download the SDK (free) per docs/04-qnn-sdk-setup.md, then run this once:
#
#   scripts/copy-qnn-libs.sh [QNN_SDK_ROOT]
#
# Multi-chip: by default it bundles the HTP skels for several recent Snapdragon
# arches so the APK runs DLC models on more than just the S26 Ultra. Override:
#
#   HTP_ARCHES="v81"            scripts/copy-qnn-libs.sh   # S26 Ultra only (lean)
#   HTP_ARCHES="v73 v75 v79 v81" scripts/copy-qnn-libs.sh  # 8 Gen 2 … Elite Gen 5
#
# Each arch costs ~22 MB (DSP6 skel + prepare lib). See ARCHITECTURE.md §3.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/android-app/app/src/main/jniLibs/arm64-v8a"

QNN_SDK_ROOT="${1:-${QNN_SDK_ROOT:-}}"
if [[ -z "${QNN_SDK_ROOT}" ]]; then
    QNN_SDK_ROOT="$(sed -n 's/^qnn\.sdk\.root=//p' "${REPO_ROOT}/android-app/local.properties" 2>/dev/null || true)"
fi
if [[ -z "${QNN_SDK_ROOT}" || ! -d "${QNN_SDK_ROOT}/lib/aarch64-android" ]]; then
    echo "ERROR: QNN_SDK_ROOT not found (got: '${QNN_SDK_ROOT}')." >&2
    exit 2
fi

HTP_ARCHES="${HTP_ARCHES:-v73 v75 v79 v81}"
A64="${QNN_SDK_ROOT}/lib/aarch64-android"

mkdir -p "${DEST}"

# Host-side (aarch64) backend + system libraries (arch-independent).
for f in libQnnSystem.so libQnnHtp.so libQnnHtpProfilingReader.so \
         libQnnHtpNetRunExtensions.so libQnnCpu.so libQnnGpu.so; do
    cp -v "${A64}/${f}" "${DEST}/"
done

# Online-prepare backend for DLC models (~90 MB, arch-independent host lib).
cp -v "${A64}/libQnnHtpPrepare.so" "${DEST}/" || true

# Per-arch: aarch64 stubs + Hexagon (DSP6) skel & prepare lib.
for arch in ${HTP_ARCHES}; do
    up="$(tr '[:lower:]' '[:upper:]' <<<"${arch:0:1}")${arch:1}"   # v81 → V81
    hex="${QNN_SDK_ROOT}/lib/hexagon-${arch}/unsigned"
    if [[ ! -d "${hex}" ]]; then
        echo "WARN: ${hex} not in SDK — skipping ${arch}" >&2
        continue
    fi
    cp -v "${A64}/libQnnHtp${up}Stub.so" "${DEST}/"
    cp -v "${A64}/libQnnHtp${up}CalculatorStub.so" "${DEST}/"
    cp -v "${hex}/libQnnHtp${up}Skel.so" "${DEST}/"
    cp -v "${hex}/libQnnHtp${up}.so" "${DEST}/"
    echo "  + arch ${arch}"
done

echo
echo "Done. $(ls "${DEST}" | wc -l) libraries, $(du -sh "${DEST}" | cut -f1) in ${DEST}"
echo "Bundled HTP arches: ${HTP_ARCHES}"
