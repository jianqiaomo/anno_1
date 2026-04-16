#!/bin/bash

set -euo pipefail

# Default to release mode if no argument provided
BUILD_TYPE=${1:-Release}

# Script is in CNN/script, so CNN root is one level up
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BUILD_DIR="${PROJECT_ROOT}/cmake-build-${BUILD_TYPE,,}"

cd "${PROJECT_ROOT}"

mkdir -p "${BUILD_DIR}"

# If this build dir was configured for a different source tree, reset it.
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    HOME_DIR=$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "${BUILD_DIR}/CMakeCache.txt" | cut -d= -f2- || true)
    if [ "${HOME_DIR}" != "${PROJECT_ROOT}" ]; then
        rm -rf "${BUILD_DIR}"/*
    fi
fi

cd "${BUILD_DIR}"
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -G "CodeBlocks - Unix Makefiles" ..
make

cd "${PROJECT_ROOT}"

if [ ! -d "data" ]
then
    tar -xzvf data.tar.gz -C .
fi

cd script
