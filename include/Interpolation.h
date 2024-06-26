/* 
 * This file is part of Tensor.
 * 
 * Tensor is free software: you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation, either version 3 of the License, or any later version.
 * 
 * Tensor is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR * A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with 
 * Tensor. If not, see <https://www.gnu.org/licenses/>. 
 */

#pragma once

#include "Tensor.h"

namespace gt {

/* produces N linearly spaced points between min and max */
template<typename T> requires std::is_arithmetic_v<T>
inline constexpr Tensor<T> linspace(T min, T max, size_t N)
{
    assert((N > 0) && "Error in linspace: N must be greater than 0");
    Tensor<T> output({N});
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = min + (max - min) / (N - 1) * i;
    }

    return output;
}

/* produces N logarithmically spaced points between min and max */
template<typename T> requires std::is_arithmetic_v<T>
inline constexpr Tensor<T> logspace(T min, T max, size_t N)
{
    assert((N > 0) && "Error in logspace: N must be greater than 0");
    return pow(10.0f, linspace(min, max, N));
}

template<typename T>
inline constexpr std::tuple<Tensor<T>,Tensor<T>> meshgrid(const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    return std::make_pair(
        repmat(reshape(lhs, {1, lhs.size()}), {rhs.size(), 1}),
        repmat(reshape(rhs, {rhs.size(), 1}), {1, lhs.size()})
    );
}

template<typename T>
inline constexpr std::tuple<Tensor<T>,Tensor<T>,Tensor<T>>
    meshgrid(const Tensor<T>& t0, const Tensor<T>& t1, const Tensor<T>& t2)
{
    return std::make_tuple(
        repmat(reshape(t0, {1, t0.size(), 1}), {t1.size(), 1, t2.size()}),
        repmat(reshape(t1, {t1.size(), 1, 1}), {1, t0.size(), t2.size()}),
        repmat(reshape(t2, {1, 1, t2.size()}), {t1.size(), t0.size(), 1})
    );
}

template<typename T, typename... Ts>
inline constexpr std::vector<Tensor<T>> ndgrid(const Tensor<T>& arg, const Tensor<Ts>&... args)
{
    std::vector<Tensor<T>> output = {static_cast<Tensor<T>>(args)...};
    return output;
}

/* return the closest index in input to value using a binary search 
 * if value is equidistant between two input values, returns the lower */
template<typename T>
inline constexpr size_t binary_search(const Tensor<T>& input, T value)
{
    assert(is_sorted(input) && "Error in binary_search: input isn't sorted");

    if (input.size() == 1) {
        return 0;
    }

    bool is_increasing = input[1] > input[0];
    size_t low = 0;
    size_t high = input.size() - 1;

    /* handle the case where value is outside of the array bounds */
    if ((is_increasing && value <= input[low]) || (!is_increasing && value >= input[low])) {
        return low;
    }
    if ((is_increasing && value >= input[high]) || (!is_increasing && value <= input[high])) {
        return high;
    }

    /* binary search */
    while (low < high) {
        if (high - low <= 1) {
            break;
        }

        size_t mid = (low + high) / 2;

        if ((is_increasing && input[mid] <= value) || (!is_increasing && input[mid] >= value)) {
            low = mid;
        } else {
            high = mid;
        }
    }

    /* pick the closer of low and high */
    size_t index;
    if (std::abs(input[low] - value) <= std::abs(input[high] - value)) {
        index = low;
    } else {
        index = high;
    }

    return index;
}

/* helper function for interpolation
 * returns the indices in x surrounding xi */
template<typename T>
inline constexpr std::tuple<size_t,size_t> interpolation_bounds(const Tensor<T>& x, T xi)
{
    size_t x1 = binary_search(x, xi);

    size_t x2;
    if (x.size() > 1) {
        if (x[1] > x[0]) {
            /* increasing case */
            if (x1 == x.size() - 1 || (xi < x[x1] && x1 != 0)) {
                x1 = x1 - 1;
            }
            x2 = x1 + 1;
        } else {
            /* decreasing case */
            if (x1 == 0 || (xi < x[x1] && x1 != x.size() - 1)) {
                x1 = x1 + 1;
            }
            x2 = x1 - 1;
        }
    } else {
        x2 = x1;
    }

    return std::make_pair(x1, x2);
}

/* 1D linear interpolation; x must be sorted
 * extrapolates for values of xi outside of x */
template<typename T>
inline constexpr Tensor<T> interp1(const Tensor<T>& x, const Tensor<T>& y, const Tensor<T>& xi)
{
    assert(ndims(x) == 1 && "Error in interp1: x is not one-dimensional");
    assert(ndims(y) == 1 && "Error in interp1: y is not one-dimensional");
    assert(ndims(xi) == 1 && "Error in interp1: xi is not one-dimensional");
    assert(x.size() == y.size() && "Error in interp1: x and y are different sizes");
    assert(is_sorted(x) && "Error in interp1: x is not sorted");

    Tensor<T> yi(xi.shape());

    for (size_t i = 0; i < xi.size(); i++) {
        std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);
        size_t x1 = std::get<0>(xbounds);
        size_t x2 = std::get<1>(xbounds);
        //std::cout << x1 << " " << x2 << std::endl;

        T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);
        yi[i] = y(x1) * (1 - xd) + y(x2) * xd;
    }

    return yi;
}

/* 2D linear interpolation, x and y must be sorted
 * extrapolates for values of xi, yi outside of x, y */
template<typename T>
inline constexpr Tensor<T> interp2(const Tensor<T>& x, const Tensor<T>& y,
    const Tensor<T>& xy, const Tensor<T>& xi, const Tensor<T>& yi)
{
    assert(ndims(x) == 1 && "Error in interp2: x is not one-dimensional");
    assert(ndims(y) == 1 && "Error in interp2: y is not one-dimensional");
    assert(ndims(xy) == 2 && "Error in interp2: xy is not two-dimensional");
    assert(ndims(xi) == 1 && "Error in interp2: xi is not one-dimensional");
    assert(ndims(yi) == 1 && "Error in interp2: yi is not one-dimensional");
    assert(x.shape(0) == xy.shape(0) &&
            "Error in interp2: size of x doesn't match zeroth dimension of xy");
    assert(y.shape(0) == xy.shape(1) &&
            "Error in interp2: size of y doesn't match first dimension of xy");
    assert(is_sorted(x) && "Error in interp2: x is not sorted");
    assert(is_sorted(y) && "Error in interp2: y is not sorted");

    Tensor<T> xiyi({xi.size(), yi.size()});

    for (size_t i = 0; i < xi.size(); i++) {
        std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);

        size_t x1 = std::get<0>(xbounds);
        size_t x2 = std::get<1>(xbounds);

        T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);

        for (size_t j = 0; j < yi.size(); j++) {
            std::tuple<size_t,size_t> ybounds = interpolation_bounds(y, yi[j]);

            size_t y1 = std::get<0>(ybounds);
            size_t y2 = std::get<1>(ybounds);
            
            T yd = (yi[j] - y[y1]) / (y[y2] - y[y1]);

            T c00 = xy(x1,y1) * (1 - xd) + xy(x2,y1) * xd;
            T c01 = xy(x1,y2) * (1 - xd) + xy(x2,y2) * xd;
            xiyi(i,j) = c00 * (1 - yd) + c01 * yd;
        }
    }

    return xiyi;
}

/* 3D linear interpolation, x and y must be sorted
 * extrapolates for values of xi, yi outside of x, y */
template<typename T>
inline constexpr Tensor<T> interp3(const Tensor<T>& x, const Tensor<T>& y,
    const Tensor<T>& z, const Tensor<T>& xyz, const Tensor<T>& xi,
    const Tensor<T>& yi, const Tensor<T>& zi)
{
    assert(ndims(x) == 1 && "Error in interp3: x is not one-dimensional");
    assert(ndims(y) == 1 && "Error in interp3: y is not one-dimensional");
    assert(ndims(z) == 1 && "Error in interp3: z is not one-dimensional");
    assert(ndims(xyz) == 3 && "Error in interp3: xyz is not three-dimensional");
    assert(ndims(xi) == 1 && "Error in interp3: xi is not one-dimensional");
    assert(ndims(yi) == 1 && "Error in interp3: yi is not one-dimensional");
    assert(ndims(zi) == 1 && "Error in interp3: zi is not one-dimensional");
    assert(x.shape(0) == xyz.shape(0) && 
        "Error in interp3: size of x doesn't match zeroth dimension of xyz");
    assert(y.shape(1) == xyz.shape(1) && 
        "Error in interp3: size of y doesn't match first dimension of xyz");
    assert(y.shape(2) == xyz.shape(2) && 
        "Error in interp3: size of y doesn't match second dimension of xyz");
    assert(is_sorted(x) && "Error in interp3: x is not sorted");
    assert(is_sorted(y) && "Error in interp3: y is not sorted");
    assert(is_sorted(z) && "Error in interp3: z is not sorted");

    Tensor<T> xiyizi({xi.size(), yi.size(), zi.size()});

    for (size_t i = 0; i < xi.size(); i++) {
        std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);

        size_t x1 = std::get<0>(xbounds);
        size_t x2 = std::get<1>(xbounds);

        T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);

        for (size_t j = 0; j < yi.size(); j++) {
            std::tuple<size_t,size_t> ybounds = interpolation_bounds(y, yi[j]);

            size_t y1 = std::get<0>(ybounds);
            size_t y2 = std::get<1>(ybounds);
            
            T yd = (yi[j] - y[y1]) / (y[y2] - y[y1]);

            for (size_t k = 0; k < zi.size(); k++) {
                std::tuple<size_t,size_t> zbounds = interpolation_bounds(z, zi[k]);

                size_t z1 = std::get<0>(zbounds);
                size_t z2 = std::get<1>(zbounds);
                
                T zd = (zi[k] - z[z1]) / (z[z2] - z[z1]);

                T c000 = xyz(x1,y1,z1);
                T c001 = xyz(x1,y1,z2);
                T c010 = xyz(x1,y2,z1);
                T c011 = xyz(x1,y2,z2);
                T c100 = xyz(x2,y1,z1);
                T c101 = xyz(x2,y1,z2);
                T c110 = xyz(x2,y2,z1);
                T c111 = xyz(x2,y2,z2);

                T c00 = c000 * (1 - xd) + c100 * xd;
                T c01 = c001 * (1 - xd) + c101 * xd;
                T c10 = c010 * (1 - xd) + c110 * xd;
                T c11 = c011 * (1 - xd) + c111 * xd;

                T c0 = c00 * (1 - yd) + c10 * yd;
                T c1 = c01 * (1 - yd) + c11 * yd;
                
                xiyizi(i,j,k) = c0 * (1 - zd) + c1 * zd;
            }
        }
    }

    return zi;
}

} // namespace gt
