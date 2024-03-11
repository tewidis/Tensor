# Tensor

An efficient library for developing numerical and scientific software in C++
that syntactically resembles languages/libraries like Fortran, MATLAB, or Numpy.

## Features

* Supports multidimensional arrays (or Tensors) and imposes no limit on the
dimensionality of these Tensors.
* Employs expression templates to support lazy evaluation of mathematical
functions. This improves performance by deferring computation until the result
is needed and collapsing consecutive operations into a single loop.
* Implements many of useful signal processing functions, such as convolution,
FFT, matrix multiplication, and various windowing functions (hamming, hanning,
hann).
* Uses OpenBLAS for improved matrix multiply performance.
* Uses fftw for improved FFT performance.
* Supports constant-time shape-changing operations, such as reshape and
permute.

## Syntax Examples

## Timing Comparisons (MATLAB)

## Alternatives

Tensor was developed to improve my understanding of more advanced modern C++
concepts (pun intended), such as template metaprogramming, move semantics,
ranges, and concepts.  Many alternatives exist such as xtensor, Armadillo, 
blitz++, Blaze, Eigen, etc., all of which are probably better suited for use in 
a production environment.  I've done my best to ensure all of the underlying 
functions are implemented efficiently and correctly and have tests to prevent 
regressions, but the other libraries previously listed are considerably more 
mature and have undergone more rigorous testing. That being said, I've 
experimented with many of these libraries and found quirks with their APIs that 
I didn't like and have attempted to remedy in my own implementation, so if you
end up using it and have feedback, I'm always open to it.

## Notable Differences from MATLAB

1. Tensors are 0-indexed instead of 1-indexed. This includes the indexing of
dimensions.

``` C++
// MATLAB
tensor = permute(tensor, [3 1 2]);

// C++ Tensor
tensor = permute(tensor, {2, 0, 1});
```

## TODO

* median
* flip
* hypot
* cart2sph
* sph2cart
* interp
* circshift
* conv
* ipermute
* squeeze
* filter
* where
* mod
* fix
* rem

## Licensing

Tensor is licensed using MIT? GPL? Figure this out.
