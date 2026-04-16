#!/bin/bash

# Default to debug mode if no argument provided
BUILD_TYPE=${1:-Debug}
OPT_LEVEL=${2:-O0}  # Default to O0 optimization
BUILD_DIR="cmake-build-${BUILD_TYPE,,}"  # Convert to lowercase

echo "Building in ${BUILD_TYPE} mode with -${OPT_LEVEL} optimization..."

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DOPT_LEVEL=${OPT_LEVEL} -G "CodeBlocks - Unix Makefiles" ..
make
cd ..

#if [ ! -d "./data" ]
#then
#    tar -xzvf data.tar.gz
#fi
#cd script
