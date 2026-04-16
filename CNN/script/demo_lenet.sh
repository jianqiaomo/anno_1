#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CNN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mode="all"
build_type="Release"

if [[ $# -ge 1 ]]; then
    case "${1,,}" in
        build|run|all)
            mode="${1,,}"
            shift
            ;;
        debug)
            build_type="Debug"
            shift
            ;;
        release)
            build_type="Release"
            shift
            ;;
        *)
            ;;
    esac
fi

if [[ $# -ge 1 ]]; then
    case "$1" in
        debug|Debug)
            build_type="Debug"
            shift
            ;;
        release|Release)
            build_type="Release"
            shift
            ;;
    esac
fi

run_file="${CNN_ROOT}/cmake-build-${build_type,,}/src/demo_lenet_run"
out_file="${CNN_ROOT}/output/single/demo-result-lenet5.txt"

lenet_i="${CNN_ROOT}/data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv"
lenet_c="${CNN_ROOT}/data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv"
lenet_o="${CNN_ROOT}/output/single/lenet5.mnist.relu.max-1-infer.csv"

if [[ "$mode" == "build" || "$mode" == "all" ]]; then
    "${SCRIPT_DIR}/build.sh" "${build_type}"
    /usr/bin/cmake --build "${CNN_ROOT}/cmake-build-${build_type,,}" --target demo_lenet_run -- -j 6
fi

if [[ "$mode" == "run" || "$mode" == "all" ]]; then
    mkdir -p "${CNN_ROOT}/output/single"
    mkdir -p "${CNN_ROOT}/log/single"
    "${run_file}" "${lenet_i}" "${lenet_c}" "${lenet_o}" 1 > "${out_file}"
fi
