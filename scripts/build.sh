#!/bin/bash

# build OpenBLAS
pushd lib/OpenBLAS-0.3.26
make clean
make USE_OPENMP=1
popd

# build fftw
pushd lib/fftw-3.3.10
make clean
./configure --enable-float --enable-threads --enable-openmp
make
popd

# build Tensor
make clean
make
