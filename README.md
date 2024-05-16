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
dimensions. Tensor uses curly braces instead of brackets.

``` C++
// MATLAB Permute
tensor = permute(tensor, [3 1 2]);

// Tensor Permute
tensor = gt::permute(tensor, {2, 0, 1});
```

2. Data is arranged in column-major format. This is similar to MATLAB and
Fortran, but differs from C and Numpy which use row-major format.
3. Use \* instead of .\* for element-wise multiplication.

``` C++
// MATLAB Matrix Multiplication
tensor = t1 * t2;

// Tensor Matrix Multiplication
tensor = gt::matmul(t1, t2);
```

4. Use transpose and ctranspose instead of .' and ' respectively. These
functions are also available in MATLAB and are clearer.

``` C++
// MATLAB Transpose
tensor = t1.';

// MATLAB Conjugate Transpose
tensor = t1';

// Tensor Transpose
tensor = gt::transpose(t1);

// Tensor Conjugate Transpose
tensor = gt::ctranspose(t1);
```

5. To take the minimum/maximum of a Tensor and a value, use minval and maxval
respectively. To take the minimum/maximum across a dimension, use min and max.

``` C++
// MATLAB minimum comparison with value
tensor = min(tensor, 1);

// MATLAB minimum across the second dimension
tensor = min(tensor, [], 2);

// Tensor minimum comparison with value
tensor = gt::minval(tensor, 1);

// Tensor minimum across the second dimension
tensor = gt::min(tensor, 1);
```

## Structure

1. Arithmetic
    * Addition
        - +, sum, cumsum, movsum
    * Subtraction/Differentiation
        - -, diff, gradient
    * Multiplication
        - * prod, cumprod
    * Division/Integration
        - /, trapz, cumtrapz
    * Exponentiation
        - pow, log, log2, log1p, log10, exp, expm1, exp2, sqrt
    * Modular arithmetic
        - mod, rem, ceil, fix, floor, round, sign
    * Complex
        - abs, angle, conj, real, imag, unwrap
    * Classifications
        - isinf, isnan, isfinite, isreal, any, all
2. Trigonometry
    * Sine
        - sin, sind, asin, asind, sinh, asinh
    * Cosine
        - cos, cosd, acos, acosd, cosh, acosh
    * Tangent
        - tan, tand, atan, atand, atan2, atan2d, tanh, atanh
    * Cosecant
        - csc, cscd, acsc, acscd, csch, acsch
    * Secant
        - sec, secd, asec, asecd, sech, asech
    * Cotangent
        - cot, cotd, acot, acotd, coth, acoth
    * Hypotenuse
        - hypot
    * Conversions
        - deg2rad, rad2deg, cart2pol, pol2cart, cart2sph, sph2cart
3. Tensor Operations
    * Tensor Creation
        - zeros, ones, eye, cat, repmat, repelem, rand, diag
    * Size, Shape, and Order
        - length, size, ndims, numel
        - isscalar, issorted, isvector, ismatrix, isempty
        - broadcast
    * Reshape and Rearrange
        - flip, rot90, permute, ipermute, circshift, shiftdim, reshape, squeeze,
        flatten
    * Indexing
        - ind2sub, sub2ind
4. Interpolation
    * Linear
        - interp1, interp2, interp3
    * Other
        - pchip, makima, spline, padecoef
    * Grid Creation
        - linspace, logspace, meshgrid, ndgrid
5. Statistics
    * Basic Statistics
        - min, mink, max, maxk, bounds, mean, median, mode, stddev, var
    * Cumulative Statistics
        - cummax, cummin
    * Moving Statistics
        - movmax, movmean, movmedian, movmin, movprod, movstd, movsum, movvar
    * Percentiles and Quantiles
        - prctile, quantile, iqr
    * Forecasting Metrics
        - rmse, mape
    * Covariance and Correlation
        - cov, corrcoef, xcov, xcorr
6. Linear Algebra
    * Linear Equations
        - mldivide, mrdivide, inv, pinv
    * Eigenvalues and Singular Values
        - eig, svd
    * Matrix Decomposition
        - lu, qr, chol
    * Matrix Operations
        - transpose, ctranspose, kron, cross, dot
    * Matrix Structure
        - bandwidth, tril, triu
    * Matrix Properties
        - norm, vecnorm, cond, det, null, orth, rank, rref, trace, subspace
7. Random
    * Uniform Distributions
        - rand (uniform real), randi (uniform int)
    * Bernoulli Distributions
        - bernoulli, binomial, negative_binomial, geometric
    * Poisson Distributions
        - poisson, exponential, gamma, weibull, extreme_value
    * Normal Distributions
        - randn, lognormal, chi_squared, cauchy, fisher_f, student_t
8. Signal Processing
    * Convolution
        - conv1, conv2, conv3
    * Fourier Analysis
        - fft, fft2, ifft, ifft2, fftn, ifftn, fftshift, ifftshift
    * Windows
        - barthann, bartlett, blackman, blackmanharris, bohman, chebyshev,
        flattop, gaussian, hamming, hann, hanning, kaiser, nuttall, parzen,
        rect, taylor, triang, tukey

## Licensing

This project is licensed under the terms of the GNU GPL-3.0-or-later.
