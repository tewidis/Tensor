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

#include "Tensor.h"
#include "cblas.h"

namespace gt
{
    #define UNARY_EXPRESSION(op) \
    template<typename T> \
    inline constexpr Tensor<T> op(const Tensor<T>& input) \
    { \
        Tensor<T> output(input.shape()); \
        std::transform(input.begin(), input.end(), output.begin(), \
            [] (T value) { return std::op(value); }); \
        return output; \
    }

    //TODO: Need to be smarter about handling type promotion here
    #define BINARY_EXPRESSION(op) \
    template<typename LHS, typename RHS> \
    inline constexpr Tensor<LHS> operator op(const Tensor<LHS>& lhs, const Tensor<RHS>& rhs) \
    { \
        assert((lhs.size() == rhs.size()) && "Error in binary operation: Tensors are different shapes"); \
        Tensor<LHS> output(lhs.shape()); \
        for (size_t i = 0; i < output.size(); i++) { \
            output[i] = lhs[i] op rhs[i]; \
        } \
        return output; \
    } \
    template<typename LHS, typename RHS> requires std::is_arithmetic_v<RHS>\
    inline constexpr Tensor<LHS> operator op(const Tensor<LHS>& lhs, RHS rhs) \
    { \
        Tensor<LHS> output(lhs.shape()); \
        for (size_t i = 0; i < output.size(); i++) { \
            output[i] = lhs[i] op rhs; \
        } \
        return output; \
    }

    UNARY_EXPRESSION(sin);
    UNARY_EXPRESSION(cos);
    UNARY_EXPRESSION(tan);
    UNARY_EXPRESSION(sinh);
    UNARY_EXPRESSION(cosh);
    UNARY_EXPRESSION(tanh);
    UNARY_EXPRESSION(asin);
    UNARY_EXPRESSION(acos);
    UNARY_EXPRESSION(atan);
    UNARY_EXPRESSION(asinh);
    UNARY_EXPRESSION(acosh);
    UNARY_EXPRESSION(atanh);
    UNARY_EXPRESSION(sqrt);
    UNARY_EXPRESSION(log);
    UNARY_EXPRESSION(log1p);
    UNARY_EXPRESSION(log2);
    UNARY_EXPRESSION(log10);
    UNARY_EXPRESSION(exp);
    UNARY_EXPRESSION(expm1);
    UNARY_EXPRESSION(exp2);
    UNARY_EXPRESSION(floor);
    UNARY_EXPRESSION(ceil);
    UNARY_EXPRESSION(abs);
    UNARY_EXPRESSION(hypot);

    BINARY_EXPRESSION(+);
    BINARY_EXPRESSION(-);
    BINARY_EXPRESSION(*);
    BINARY_EXPRESSION(/);
    BINARY_EXPRESSION(>);
    BINARY_EXPRESSION(<);
    BINARY_EXPRESSION(>=);
    BINARY_EXPRESSION(<=);
    BINARY_EXPRESSION(==);
    BINARY_EXPRESSION(!=);

    template<typename T>
    inline constexpr bool all(const Tensor<T>& input)
    {
        bool all = true;
        for (size_t i = 0; i < input.size(); i++) {
            all &= (input[i] != 0);
        }
        return all;
    }

    template<typename T>
    inline constexpr bool any(const Tensor<T>& input)
    {
        bool any = false;
        for (size_t i = 0; i < input.size(); i++) {
            any |= (input[i] != 0);
        }
        return any;
    }

    template<typename T>
    inline constexpr Tensor<T> isinf(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return std::isinf(value); });

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> isnan(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return std::isnan(value); });

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> isfinite(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return !std::isinf(value) && !std::isnan(value); });

        return output;
    }

    /* rounds towards 0 */
    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr T fix(T input)
    {
        return (input < 0) ? std::ceil(input) : std::floor(input);
    }

    template<typename T>
    inline constexpr Tensor<T> fix(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return fix(value); });

        return output;
    }

    /* remainder as computed by MATLAB */
    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr T1 rem(T1 lhs, T2 rhs)
    {
        return lhs - fix(lhs / rhs) * rhs;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr Tensor<T1> rem(const Tensor<T2>& lhs, T1 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return rem(value, rhs); });

        return output;
    }

    /* modulus as computed by MATLAB */
    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr T1 mod(T1 lhs, T2 rhs)
    {
        return lhs - std::floor(lhs / rhs) * rhs;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr Tensor<T1> mod(const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return mod(value, rhs); });

        return output;
    }

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
    inline constexpr Tensor<T> ones(const std::vector<size_t>& shape)
    {
        Tensor<T> output(shape);

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = 1;
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> zeros(const std::vector<size_t>& shape)
    {
        Tensor<T> output(shape);

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = 0;
        }

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
    inline constexpr Tensor<T> pow(const Tensor<T>& input, float scalar)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::pow(input[i], scalar);
        }
        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> pow(float scalar, const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::pow(scalar, input[i]);
        }
        return output;
    }

    inline size_t calculate_offset(const std::vector<size_t>& input_stride,
        const std::vector<size_t>& output_stride, const std::vector<size_t>& shape,
        size_t index)
    {
        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            offset += input_stride[i] * ((index / output_stride[i]) % shape[i]);
        }

        return offset;
    }

    template<typename T>
    inline constexpr Tensor<T> sum(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] += input[offset + j * input.stride(dim)];
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cumsum(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            T running_total = 0;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total += input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> movsum(const Tensor<T>& input, size_t B, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[offset + j * output.stride(dim)] = 0;
                for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                    size_t index = offset + k * input.stride(dim);
                    if (index < input.size()) {
                        output[offset + j * output.stride(dim)] += input[index];
                    }
                }
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> diff(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] -= 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[i] = input[offset + input.stride(dim)] - input[offset];
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> prod(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[i] = 1;
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] *= input[offset + j * input.stride(dim)];
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cumprod(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            T running_total = 1;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total *= input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> movprod(const Tensor<T>& input, size_t B, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[offset + j * output.stride(dim)] = 1;
                for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                    size_t index = offset + k * input.stride(dim);
                    if (index < input.size()) {
                        output[offset + j * output.stride(dim)] *= input[index];
                    }
                }
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> trapz(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 1; j < input.shape(dim); j++) {
                output[i] += (input[offset + j * input.stride(dim)] +
                    input[offset + (j - 1) * input.stride(dim)]) / 2.0f;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cumtrapz(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[offset] = 0;
            for (size_t j = 1; j < output.shape(dim); j++) {
                output[offset + j * input.stride(dim)] = 
                    output[offset + (j - 1) * input.stride(dim)] +
                    (input[offset + j * input.stride(dim)] +
                    input[offset + (j - 1) * input.stride(dim)]) / 2.0f;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> max(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[i] = input[offset];
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] = std::max(output[i], input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cummax(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[offset] = input[offset];
            for (size_t j = 1; j < output.shape(dim); j++) {
                output[offset + j * input.stride(dim)] = std::max(
                    output[offset + (j - 1) * input.stride(dim)],
                    input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> movmax(const Tensor<T>& input, size_t B, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                T local_max = input[offset + j * input.stride(dim)];
                for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                    size_t index = offset + k * input.stride(dim);
                    if (index < input.size()) {
                        local_max = std::max(local_max, input[index]);
                    }
                }
                output[offset + j * output.stride(dim)] = local_max;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> maxval(const Tensor<T>& input, T value)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(), \
            [value] (T data) { return std::max(data, value); }); \

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> min(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[i] = input[offset];
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] = std::min(output[i], input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cummin(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            output[offset] = input[offset];
            for (size_t j = 1; j < output.shape(dim); j++) {
                output[offset + j * input.stride(dim)] = std::min(
                    output[offset + (j - 1) * input.stride(dim)],
                    input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> movmin(const Tensor<T>& input, size_t B, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                T local_min = input[offset + j * input.stride(dim)];
                for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                    size_t index = offset + k * input.stride(dim);
                    if (index < input.size()) {
                        local_min = std::min(local_min, input[index]);
                    }
                }
                output[offset + j * output.stride(dim)] = local_min;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> minval(const Tensor<T>& input, T value)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(), \
            [value] (T data) { return std::min(data, value); }); \

        return output;
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
        size_t ndims = std::max(reps.size(), input.shape().size());
        std::vector<size_t> shape(reps.size());
        for (size_t i = 0; i < ndims; i++) {
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
        assert((input.shape().size() == order.size()) &&
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
        assert((input.shape().size() == order.size()) &&
            "Error in ipermute: Size of order does not dimensionality of Tensor");

        std::vector<size_t> new_order(order.size());
        for (size_t i = 0; i < new_order.size(); i++) {
            new_order[order[i]] = i;
        }

        return permute(input, new_order);
    }

    template<typename T>
    inline constexpr Tensor<T> mean(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] += input[offset + j * input.stride(dim)] / input.shape(dim);
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> movmean(const Tensor<T>& input, size_t B, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[offset + j * output.stride(dim)] = 0;
                T denom = 0;
                for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                    size_t index = offset + k * input.stride(dim);
                    if (index < input.size()) {
                        output[offset + j * output.stride(dim)] += input[index];
                        denom++;
                    }
                }
                output[offset + j * output.stride(dim)] /= denom;
            }
        }

        return output;
    }

    template<typename T>
    inline constexpr T median(Tensor<T>& input)
    {
        if (input.size() % 2 == 0) {
            auto med = input.begin() + input.size() / 2;
            std::nth_element(input.begin(), med, input.end());
            return (input[input.size() / 2 - 1] + input[input.size() / 2]) / 2.0f;
        } else {
            auto med = input.begin() + input.size() / 2;
            std::nth_element(input.begin(), med, input.end());
            return input[input.size() / 2];
        }
    }

    template<typename T>
    inline constexpr Tensor<T> median(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            Tensor<T> temp({input.shape(dim)});
            for (size_t j = 0; j < input.shape(dim); j++) {
                temp[j] = input[offset + j * input.stride(dim)];
            }
            output[i] = median(temp);
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> var(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> reps(input.shape().size());
        std::fill(reps.begin(), reps.end(), 1);
        reps[dim] = input.shape(dim);
        size_t denom = std::max(std::size_t{1}, input.shape(dim) - 1);
        return sum(pow(abs(input - repmat(mean(input, dim), reps)), 2), dim) / denom;
    }

    template<typename T>
    inline constexpr Tensor<T> stddev(const Tensor<T>& input, size_t dim)
    {
        return sqrt(var(input, dim));
    }

    template<typename T>
    inline constexpr Tensor<T> circshift(const Tensor<T>& input, int64_t nshift, size_t dim = 0)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }
        std::vector<size_t> stride = calculate_stride(shape);

        nshift = input.shape(dim) - rem(nshift, input.shape(dim));

        Tensor<T> output(input.shape());
        size_t modulo = input.shape(dim) * input.stride(dim);
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[offset + j * output.stride(dim)] =
                    input[offset + (j + nshift) * output.stride(dim) % modulo];
            }
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
        std::vector<size_t> stride = calculate_stride(shape);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), stride, shape, i);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[offset + j * output.stride(dim)] =
                    input[offset + (input.shape(dim) - j - 1) * input.stride(dim)];
            }
        }

        return output;
    }

    /* input is an Nx3 Tensor where the zeroth dimension is the x-coordinate,
     * the first dimension is the y-coordinate, and the second dimension is the
     * z-coordinate
     * output is an Nx3 Tensor where the zeroth dimension is the azimuth,
     * the first dimension is the elevation, and the second dimension is the
     * radius */
    template<typename T>
    inline constexpr Tensor<T> cart2sph(const Tensor<T>& input)
    {
        assert(input.shape().size() == 2 && input.shape(1) == 3);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape(0); i++) {
            T hypotxy = std::hypot(input(i,0), input(i,1));
            output(i,0) = std::atan2(input(i,1), input(i,0));
            output(i,1) = std::atan2(input(i,2), hypotxy);
            output(i,2) = std::hypot(hypotxy, input(i,2));
        }

        return output;
    }

    /* input is an Nx3 Tensor where the zeroth dimension is the azimuth,
     * the first dimension is the elevation, and the second dimension is the
     * radius
     * output is an Nx3 Tensor where the zeroth dimension is the x-coordinate,
     * the first dimension is the y-coordinate, and the second dimension is the
     * z-coordinate */
    template<typename T>
    inline constexpr Tensor<T> sph2cart(const Tensor<T>& input)
    {
        assert(input.shape().size() == 2 && input.shape(1) == 3);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape(0); i++) {
            T rcoselev = input(i,2) * std::cos(input(i,1));
            output(i,0) = rcoselev * std::cos(input(i,0));
            output(i,1) = rcoselev * std::sin(input(i,0));
            output(i,2) = input(i,2) * std::sin(input(i,1));
        }

        return output;
    }

    /* input is an Nx3 Tensor where the zeroth dimension is the x-coordinate,
     * the first dimension is the y-coordinate, and the second dimension is the
     * z-coordinate
     * output is an Nx3 Tensor where the zeroth dimension is the theta,
     * the first dimension is the radius, and the second dimension is the
     * height */
    template<typename T>
    inline constexpr Tensor<T> cart2pol(const Tensor<T>& input)
    {
        assert(input.shape().size() == 2 && input.shape(1) == 3);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape(0); i++) {
            output(i,0) = std::atan2(input(i,1), input(i,0));
            output(i,1) = std::hypot(input(i,0), input(i,1));
            output(i,2) = input(i,2);
        }

        return output;
    }

    /* input is an Nx3 Tensor where the zeroth dimension is the theta,
     * the first dimension is the radius, and the second dimension is the
     * height
     * output is an Nx3 Tensor where the zeroth dimension is the x-coordinate,
     * the first dimension is the y-coordinate, and the second dimension is the
     * z-coordinate */
    template<typename T>
    inline constexpr Tensor<T> pol2cart(const Tensor<T>& input)
    {
        assert(input.shape().size() == 2 && input.shape(1) == 3);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape(0); i++) {
            output(i,0) = input(i,1) * std::cos(input(i,0));
            output(i,1) = input(i,1) * std::sin(input(i,0));
            output(i,2) = input(i,2);
        }

        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> cat(size_t dim, const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        std::vector<size_t> shape(std::max(lhs.shape().size(), rhs.shape().size()));
        for (size_t i = 0; i < lhs.shape().size(); i++) {
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
    inline constexpr Tensor<T> cat(size_t dim, const Tensor<T>& arg, const Tensor<Ts...>& args)
    {
        return cat(dim, arg, cat(dim, args));
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

        while (low < high) {
            if (high - low <= 1) {
                break;
            }

            size_t mid = (low + high) / 2;

            if (input[mid] == value) {
                return mid;
            } else if (input[mid] < value) {
                low = mid;
            } else {
                high = mid;
            }
        }

        size_t index;
        if (is_increasing) {
            if (value <= input[low] || std::abs(input[low] - value) <= std::abs(input[high] - value)) {
                index = low;
            } else {
                index = high;
            }
        } else {
            if (value >= input[low] || std::abs(input[low] - value) > std::abs(input[high] - value)) {
                index = low;
            } else {
                index = high;
            }
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
                if (x1 == 0 || (xi > x[x1] && x1 != x.size() - 1)) {
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
        assert(x.shape().size() == 1 && "Error in interp1: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp1: y is not one-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp1: xi is not one-dimensional");
        assert(x.size() == y.size() && "Error in interp1: x and y are different sizes");
        assert(is_sorted(x) && "Error in interp1: x is not sorted");

        Tensor<T> yi(xi.shape());

        for (size_t i = 0; i < xi.size(); i++) {
            std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);
            size_t x1 = std::get<0>(xbounds);
            size_t x2 = std::get<0>(xbounds);

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
        assert(x.shape().size() == 1 && "Error in interp2: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp2: y is not one-dimensional");
        assert(xy.shape().size() == 2 && "Error in interp2: xy is not two-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp2: xi is not one-dimensional");
        assert(yi.shape().size() == 1 && "Error in interp2: yi is not one-dimensional");
        assert(x.shape(0) == xy.shape(0) && "Error in interp2: size of x doesn't match zeroth dimension of xy");
        assert(y.shape(0) == xy.shape(1) && "Error in interp2: size of y doesn't match first dimension of xy");
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
        assert(x.shape().size() == 1 && "Error in interp3: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp3: y is not one-dimensional");
        assert(z.shape().size() == 1 && "Error in interp3: z is not one-dimensional");
        assert(xyz.shape().size() == 3 && "Error in interp3: xyz is not three-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp3: xi is not one-dimensional");
        assert(yi.shape().size() == 1 && "Error in interp3: yi is not one-dimensional");
        assert(zi.shape().size() == 1 && "Error in interp3: zi is not one-dimensional");
        assert(x.shape(0) == xyz.shape(0) && "Error in interp3: size of x doesn't match zeroth dimension of xyz");
        assert(y.shape(1) == xyz.shape(1) && "Error in interp3: size of y doesn't match first dimension of xyz");
        assert(y.shape(2) == xyz.shape(2) && "Error in interp3: size of y doesn't match second dimension of xyz");
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
};
