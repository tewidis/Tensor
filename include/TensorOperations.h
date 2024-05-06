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

#include <algorithm>
#include <cmath>
#include <complex>
#include <set>

#include "Arithmetic.h"
#include "Enums.h"
#include "Tensor.h"
#include "Trigonometry.h"
#include "Statistics.h"

namespace gt {

template<typename T>
inline constexpr Tensor<T> ones(const std::vector<size_t>& shape)
{
    Tensor<T> output(shape);

    std::fill(output.begin(), output.end(), 1);

    return output;
}

template<typename T>
inline constexpr Tensor<T> zeros(const std::vector<size_t>& shape)
{
    Tensor<T> output(shape);

    std::fill(output.begin(), output.end(), 0);

    return output;
}

template<typename T>
inline constexpr Tensor<T> eye(size_t size)
{
    Tensor<T> output = zeros<T>({size, size});

    for (size_t i = 0; i < output.shape(0); i++) {
        output(i,i) = 1;
    }

    return output;
}

template<typename T>
inline constexpr size_t ndims(const Tensor<T>& input)
{
    return input.shape().size();
}

template<typename T>
inline constexpr Tensor<T> reshape(const Tensor<T>& input, const std::vector<size_t>& shape)
{
    Tensor<T> output(shape);
    std::copy(input.begin(), input.end(), output.begin());
    return output;
}

template<typename T>
inline constexpr Tensor<T> squeeze(const Tensor<T>& input)
{
    std::vector<size_t> shape;
    for (size_t dim : input.shape()) {
        if (dim != 0) {
            shape.push_back(dim);
        }
    }

    return reshape(input, shape);
}

template<typename T>
inline constexpr Tensor<T> flatten(const Tensor<T>& input)
{
    return reshape(input, {input.size()});
}

template<typename T>
inline constexpr Tensor<T> repmat(const Tensor<T>& input, const std::vector<size_t>& reps)
{
    std::vector<size_t> shape(reps.size());
    for (size_t i = 0; i < std::max(reps.size(), ndims(input)); i++) {
        if (i < reps.size()) {
            shape[i] = input.shape(i) * reps[i];
        } else {
            shape[i] = input.shape(i);
        }
    }

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t index = 0;
        for (size_t j = 0; j < shape.size(); j++) {
            index += input.stride(j) * ((i / output.stride(j)) % input.shape(j));
        }
        output[i] = input[index];
    }

    return output;
}

inline size_t calculate_permute_index(const std::vector<size_t>& shape,
    const std::vector<size_t>& stride, const std::vector<size_t>& step, size_t index)
{
    size_t output = 0;
    for (size_t i = 0; i < shape.size(); i++) {
        output += step[i] * ((index / stride[i]) % shape[i]);
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> permute(const Tensor<T>& input, const std::vector<size_t>& order)
{
    assert((ndims(input) == order.size()) &&
        "Error in permute: Size of order does not dimensionality of Tensor");

    std::vector<size_t> permuted_shape(order.size());
    std::vector<size_t> permuted_stride(order.size());
    for (size_t i = 0; i < order.size(); i++) {
        permuted_shape[i] = input.shape(order[i]);
        permuted_stride[i] = input.stride(order[i]);
    }

    Tensor<T> output(permuted_shape);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = input[calculate_permute_index(permuted_shape,
            output.stride(), permuted_stride, i)];
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> ipermute(const Tensor<T>& input, const std::vector<size_t>& order)
{
    assert((ndims(input) == order.size()) &&
        "Error in ipermute: Size of order does not dimensionality of Tensor");

    std::vector<size_t> new_order(order.size());
    for (size_t i = 0; i < new_order.size(); i++) {
        new_order[order[i]] = i;
    }

    return permute(input, new_order);
}

template<typename T>
inline constexpr Tensor<T> circshift(const Tensor<T>& input, int64_t nshift, size_t dim = 0)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    nshift = input.shape(dim) - rem(nshift, input.shape(dim));

    Tensor<T> output(input.shape());
    size_t modulo = input.shape(dim) * input.stride(dim);
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        for (size_t j = 0; j < output.shape(dim); j++) {
            output[offset + j * output.stride(dim)] =
                input[offset + (j + nshift) * output.stride(dim) % modulo];
        }
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> circshift(const Tensor<T>& input,
    const std::vector<int64_t>& nshift)
{
    assert(ndims(input) == nshift.size() &&
        "Error in circshift: Mismatch in number of dimensions");

    Tensor<T> output = input;
    for (size_t dim = 0; dim < ndims(input); dim++) {
        output = circshift(output, nshift[dim], dim);
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> flip(const Tensor<T>& input, size_t dim = 0)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        for (size_t j = 0; j < output.shape(dim); j++) {
            output[offset + j * output.stride(dim)] =
                input[offset + (input.shape(dim) - j - 1) * input.stride(dim)];
        }
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> cat(size_t dim, const Tensor<T>& lhs, const Tensor<T>& rhs)
{
    std::vector<size_t> shape(std::max(ndims(lhs), ndims(rhs)));
    for (size_t i = 0; i < ndims(lhs); i++) {
        if (i != dim) {
            assert(lhs.shape(i) == rhs.shape(i) && "Error in cat: Non-singleton dimensions must agree");
            shape[i] = lhs.shape(i);
        } else {
            shape[i] = lhs.shape(i) + rhs.shape(i);
        }
    }

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t index = 0;
        bool use_lhs = true;
        for (size_t dim = 0; dim < shape.size(); dim++) {
            size_t offset = i / output.stride(dim) % output.shape(dim);
            if (offset >= lhs.shape(dim)) {
                use_lhs = false;
                offset -= lhs.shape(dim);
                index += offset * rhs.stride(dim);
            } else {
                index += offset * lhs.stride(dim);
            }
        }

        if (use_lhs) {
            output[i] = lhs[index];
        } else {
            output[i] = rhs[index];
        }
    }

    return output;
}

template<typename T, typename... Ts>
inline constexpr Tensor<T> cat(size_t dim, const Tensor<T>& arg, const Tensor<Ts>&... args)
{
    return cat(dim, arg, cat(dim, args...));
}

template<typename T>
inline constexpr Tensor<T> unique(const Tensor<T>& input)
{
    std::set<T> unique_set(input.begin(), input.end());

    Tensor<T> output({unique_set.size()});
    for (auto B = unique_set.begin(), E = unique_set.end(), Bo = output.begin();
        B != E; ++B, ++Bo)
    {
        *Bo = *B;
    }

    return output;
}

/* returns true only if input is monotonically increasing or decreasing */
template<typename T>
inline constexpr bool is_sorted(const Tensor<T>& input)
{
    bool is_increasing = true;
    for (size_t i = 1; i < input.size(); i++) {
        if (input[i] <= input[i-1]) {
            is_increasing = false;
            break;
        }
    }

    bool is_decreasing = true;
    for (size_t i = 1; i < input.size(); i++) {
        if (input[i] >= input[i-1]) {
            is_increasing = false;
            break;
        }
    }

    return is_increasing || is_decreasing;
}

template<typename T>
constexpr inline Tensor<T> broadcast(const Tensor<T>& lhs, const Tensor<T>& rhs, OPERATION op)
{
    std::vector<size_t> shape(std::max(ndims(lhs), ndims(rhs)));
    for (size_t i = 0; i < shape.size(); i++) {
        shape[i] = std::max(lhs.shape(i), rhs.shape(i));
    }

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t lidx = 0;
        size_t ridx = 0;
        for (size_t j = 0; j < ndims(output); j++) {
            lidx += lhs.stride(j) * ((i / output.stride(j)) % lhs.shape(j));
            ridx += rhs.stride(j) * ((i / output.stride(j)) % rhs.shape(j));
        }

        switch (op) {
            case PLUS:
                output[i] = lhs[lidx] + rhs[ridx];
                break;
            case MINUS:
                output[i] = lhs[lidx] - rhs[ridx];
                break;
            case TIMES:
                output[i] = lhs[lidx] * rhs[ridx];
                break;
            case DIVIDE:
                output[i] = lhs[lidx] / rhs[ridx];
                break;
            case POWER:
                output[i] = std::pow(lhs[lidx], rhs[ridx]);
                break;
            case MAX:
                output[i] = gt::max(lhs[lidx], rhs[ridx]);
                break;
            case MIN:
                output[i] = gt::min(lhs[lidx], rhs[ridx]);
                break;
            case MOD:
                output[i] = gt::mod(lhs[lidx], rhs[ridx]);
                break;
            case REM:
                output[i] = gt::rem(lhs[lidx], rhs[ridx]);
                break;
            case ATAN2:
                output[i] = std::atan2(lhs[lidx], rhs[ridx]);
                break;
            case ATAN2D:
                output[i] = gt::atan2d(lhs[lidx], rhs[ridx]);
                break;
            case HYPOT:
                output[i] = std::hypot(lhs[lidx], rhs[ridx]);
                break;
        }
    }

    return output;
}

template<typename T>
constexpr inline size_t sub2ind(const Tensor<T>& input, const std::vector<size_t>& subs) {
    size_t index = 0;

    for (size_t i = 0; i < subs.size(); i++) {
        index += input.stride(i) * subs[i];
    }

    return index;
}

template<typename T>
constexpr inline std::vector<size_t> ind2sub(const Tensor<T>& input, size_t index) {
    std::vector<size_t> subs(ndims(input));
    for (size_t i = 0; i < subs.size(); i++) {
        subs[i] = input.stride(i) * ((index / input.stride(i)) % input.shape(i));
    }

    return subs;
}

} // namespace gt
