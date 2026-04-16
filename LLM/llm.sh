#!/bin/bash

# Example usage:
# To build and run in debug mode with O0 optimization:
# ./llm.sh debug O0 arg1 arg2
# To build and run in release mode with O3 optimization:
# ./llm.sh release O3 arg1 arg2
# ./llm.sh release O3 gpt2-small --squeeze-merge | tee ../output/gpt2-small/SqueezeMerge_1/software_cout.log
# ./llm.sh release O3 gpt2-medium --squeeze-merge --skip-commit  | tee ../output/gpt2-medium/SqueezeMerge_1/software_cout.log
# ./llm.sh release O3 opt-125m --squeeze-merge --skip-commit | tee ../output/opt-125m/SqueezeMerge_1/software_cout.log

set -x

# Check if debug mode is enabled
DEBUG_MODE=${1:-false}
OPT_LEVEL=${2:-O0}  # Default to O0 optimization
shift 2  # Remove first two arguments
ARGS="$@"  # Remaining arguments to pass to the executable

if [ "$DEBUG_MODE" = "debug" ] || [ "$DEBUG_MODE" = "gdb" ]; then
    echo "Building in Debug mode for gdb debugging with -${OPT_LEVEL} optimization..."
    ./build.sh Debug ${OPT_LEVEL}
    BUILD_DIR="cmake-build-debug"
else
    echo "Building in Release mode with -${OPT_LEVEL} optimization..."
    ./build.sh Release ${OPT_LEVEL}
    BUILD_DIR="cmake-build-release"
fi

/usr/bin/cmake --build ./${BUILD_DIR} --target demo_llm_run -- -j 6

run_file=./${BUILD_DIR}/src/demo_llm_run

if [ "$DEBUG_MODE" = "debug" ] || [ "$DEBUG_MODE" = "gdb" ]; then
    echo "Running with gdb debugger..."
    gdb --args ${run_file}
else
    echo "Running normally..."
    ${run_file} ${ARGS}
fi