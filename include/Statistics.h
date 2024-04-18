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
//#include "TensorOperations.h"

namespace gt
{
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

    /* Cumulative Statistics */
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

    /* Moving Statistics */
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

    /* Basic Statistics */
    template<typename T>
    inline constexpr T angle(const std::complex<T>& input)
    {
        return std::hypot(std::imag(input), std::real(input));
    }

    template<typename T>
    inline constexpr Tensor<T> angle(const Tensor<std::complex<T>>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(), \
            [] (T data) { return angle(data); }); \

        return output;
    }

    /* custom max to handle complex numbers */
    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr T max(T lhs, T rhs)
    {
        return std::max(lhs, rhs);
    }

    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr std::complex<T> max(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        T abs_lhs = std::abs(lhs);
        T abs_rhs = std::abs(rhs);
        if (abs_lhs == abs_rhs) {
            return std::max(gt::angle(lhs), gt::angle(rhs));
        } else {
            return std::max(abs_lhs, abs_rhs);
        }
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
    inline constexpr Tensor<T> maxval(const Tensor<T>& input, T value)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(), \
            [value] (T data) { return std::max(data, value); }); \

        return output;
    }

    /* custom min to handle complex numbers */
    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr T min(T lhs, T rhs)
    {
        return std::min(lhs, rhs);
    }

    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr std::complex<T> min(std::complex<T> lhs, std::complex<T> rhs)
    {
        T abs_lhs = std::abs(lhs);
        T abs_rhs = std::abs(rhs);
        if (abs_lhs == abs_rhs) {
            return std::min(gt::angle(lhs), gt::angle(rhs));
        } else {
            return std::min(abs_lhs, abs_rhs);
        }
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
    inline constexpr Tensor<T> minval(const Tensor<T>& input, T value)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(), \
            [value] (T data) { return std::min(data, value); }); \

        return output;
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
};
