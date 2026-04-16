#!/bin/bash

set -x

# Check for debug argument
if [[ "$1" == "debug" || "$1" == "Debug" ]]; then
    BUILD_TYPE="Debug"
else
    BUILD_TYPE="Release"
fi

./build.sh ${BUILD_TYPE}
/usr/bin/cmake --build ../cmake-build-${BUILD_TYPE,,} --target demo_vgg_run -- -j 6

run_file=../cmake-build-${BUILD_TYPE,,}/src/demo_vgg_run
out_file=../output/single/demo-result-vgg11.txt

mkdir -p ../output/single
mkdir -p ../log/single

vgg11_i=../data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv
vgg11_c=../data/vgg11/vgg11.cifar.relu-1-scale-zeropoint-uint8.csv
vgg11_o=../output/single/vgg11.cifar.relu-1-infer.csv
vgg11_n=../data/vgg11/vgg11-config.csv

${run_file} ${vgg11_i} ${vgg11_c} ${vgg11_o} ${vgg11_n} 1 > ${out_file}