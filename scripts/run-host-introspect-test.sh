#!/usr/bin/env bash
# Builds and runs the host-side introspection test: the app's real
# binary_info_walk.h code against the real model .bin files, through the QNN
# SDK's x86_64 libQnnSystem.so. No device needed.
#
# Usage: scripts/run-host-introspect-test.sh [QNN_SDK_ROOT]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

QNN_SDK_ROOT="${1:-${QNN_SDK_ROOT:-}}"
if [[ -z "${QNN_SDK_ROOT}" ]]; then
    # Same default as android-app/local.properties
    QNN_SDK_ROOT="$(sed -n 's/^qnn\.sdk\.root=//p' "${REPO_ROOT}/android-app/local.properties" 2>/dev/null || true)"
fi
if [[ -z "${QNN_SDK_ROOT}" || ! -d "${QNN_SDK_ROOT}/include/QNN" ]]; then
    echo "ERROR: QNN_SDK_ROOT not found (got: '${QNN_SDK_ROOT}')" >&2
    exit 2
fi

BUILD_DIR="${REPO_ROOT}/tools/host_introspect/build"
cmake -S "${REPO_ROOT}/tools/host_introspect" -B "${BUILD_DIR}" \
      -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${BUILD_DIR}" --parallel >/dev/null

export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
exec "${BUILD_DIR}/host_introspect" "${REPO_ROOT}/models"
