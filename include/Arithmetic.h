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
#include "Trigonometry.h"
#include "Statistics.h"

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
    UNARY_EXPRESSION(round);

    template<typename T>
    inline constexpr Tensor<T> abs(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return std::abs(value); });
        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> abs(const Tensor<std::complex<T>>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (const std::complex<T>& value) { return std::abs(value); });
        return output;
    }

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

    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr bool iseven(T input)
    {
        return (input % 2) == 0;
    }

    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr bool isodd(T input)
    {
        return (input % 2) != 0;
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

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr Tensor<T1> pow(const Tensor<T1>& input, T2 scalar)
    {
        Tensor<T1> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [scalar] (T1 value) { return std::pow(value, scalar); });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    inline constexpr Tensor<T2> pow(T1 scalar, const Tensor<T2>& input)
    {
        Tensor<T2> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [scalar] (T2 value) { return std::pow(scalar, value); });
        return output;
    }

    template<typename T>
    inline constexpr Tensor<T> sum(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
            T running_total = 0;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total += input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
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

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
            output[i] = input[offset + input.stride(dim)] - input[offset];
        }

        return output;
    }

    template<typename T>
    inline constexpr T prod(const Tensor<T>& input)
    {
        //T output = std::accumulate(input.begin(), input.end(), 1, std::multiplies<T>());
        T output = 1;
        for (size_t i = 0; i < input.size(); i++) {
            output *= input(i);
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

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
            T running_total = 1;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total *= input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
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

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input.stride(), shape, dim, i);
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
};
