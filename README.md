# Tensor

An efficient library for developing numerical and scientific software in C++
that syntactically resembles languages/libraries like Fortran, MATLAB, or Numpy.

## Features

* Supports multidimensional arrays (or Tensors) and imposes no limit on the
dimensionality of these Tensors.
* Implements many of useful signal processing functions, such as convolution,
FFT, matrix multiplication, and various windowing functions (hamming, hanning,
hann).
* Uses OpenBLAS for improved matrix multiply performance.
* Uses fftw for improved FFT performance.

## Syntax Examples

## Timing Comparisons (MATLAB)

## Alternatives

Tensor was developed to improve my understanding of more advanced modern C++
language features, such as template metaprogramming, move semantics, ranges, and
concepts.  Many alternatives exist such as xtensor, Armadillo, blitz++, Blaze, 
Eigen, etc., all of which are probably better suited for use in a production 
environment.  I've done my best to ensure all of the underlying functions are 
implemented efficiently and correctly and have tests to prevent regressions, but 
the other libraries previously listed are considerably more mature and have 
undergone more rigorous testing. That being said, I've experimented with many of 
these libraries and found quirks with their APIs that I didn't like and have 
attempted to remedy in my own implementation, so if you end up using it and have 
feedback, I'm always open to it.

## Notable Differences from Other Languages and Libraries

1. Tensors are 0-indexed instead of 1-indexed. This includes the indexing of
dimensions.

``` C++
// MATLAB
tensor = permute(tensor, [3 1 2]);

// C++ Tensor
tensor = permute(tensor, {2, 0, 1});
```

2. Data is arranged in column-major format. This is similar to MATLAB and
Fortran, but differs from C and Numpy which use row-major format.

## TODO

* interp
* conv
* minval
* maxval
* where
* windows
* ones
* zeros
* eye

## Licensing

This project is licensed under the terms of the GNU GPL-3.0-or-later.
