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
#include <unordered_map>

#include "Arithmetic.h"
#include "Enums.h"
#include "Tensor.h"

namespace gt {

inline size_t calculate_offset(const std::vector<size_t>& input_stride,
    const std::vector<size_t>& shape, size_t dim, size_t index)
{
    size_t offset = 0;
    size_t stride = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i != dim) {
            offset += input_stride[i] * ((index / stride) % shape[i]);
            stride *= shape[i];
        }
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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

    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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
inline constexpr Tensor<T> movmedian(const Tensor<T>& input, size_t B, size_t dim)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    Tensor<T> temp({B});
    Tensor<T> output(input.shape());
    for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        for (size_t j = 0; j < output.shape(dim); j++) {
            output[offset + j * output.stride(dim)] = 0;
            size_t temp_idx = 0;
            for (size_t k = (j > B / 2) ? (j - B / 2) : 0; k < j + B / 2; k++) {
                size_t index = offset + k * input.stride(dim);
                temp[temp_idx] = input[index];
                temp_idx++;
            }
            output[offset + j * output.stride(dim)] = median(temp.begin(), temp.begin() + temp_idx);
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
inline constexpr T max(const Tensor<T>& input)
{
    T output = input(0);
    for (size_t i = 1; i < input.size(); i++) {
        output = gt::max(output, input(i));
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

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        output[i] = input[offset];
        for (size_t j = 0; j < input.shape(dim); j++) {
            output[i] = std::max(output[i], input[offset + j * input.stride(dim)]);
        }
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> maxk(const Tensor<T>& input, size_t k, size_t dim)
{
    std::vector<size_t> ishape = input.shape();
    if (dim < ishape.size()) {
        ishape[dim] = 1;
    }

    std::vector<size_t> oshape = input.shape();
    if (dim < oshape.size()) {
        oshape[dim] = k;
    }

    Tensor<T> temp({input.shape(dim)});
    Tensor<T> output(oshape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), ishape, dim, i);
        for (size_t j = 0; j < input.shape(dim); j++) {
            temp[j] = input[offset + j * input.stride(dim)];
        }

        offset = calculate_offset(output.stride(), ishape, dim, i);
        for (size_t j = 0; j < k; j++) {
            std::nth_element(temp.begin(), temp.end() - j - 1, temp.end());
            output[offset + j * output.stride(dim)] = temp[input.shape(dim) - j - 1];
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
inline constexpr T min(const Tensor<T>& input)
{
    T output = input(0);
    for (size_t i = 1; i < input.size(); i++) {
        output = gt::min(output, input(i));
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> min(const Tensor<T>& input, size_t dim)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        output[i] = input[offset];
        for (size_t j = 0; j < input.shape(dim); j++) {
            output[i] = std::min(output[i], input[offset + j * input.stride(dim)]);
        }
    }

    return output;
}

template<typename T>
inline constexpr Tensor<T> mink(const Tensor<T>& input, size_t k, size_t dim)
{
    std::vector<size_t> ishape = input.shape();
    if (dim < ishape.size()) {
        ishape[dim] = 1;
    }

    std::vector<size_t> oshape = input.shape();
    if (dim < oshape.size()) {
        oshape[dim] = k;
    }

    Tensor<T> temp({input.shape(dim)});
    Tensor<T> output(oshape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), ishape, dim, i);
        for (size_t j = 0; j < input.shape(dim); j++) {
            temp[j] = input[offset + j * input.stride(dim)];
        }

        offset = calculate_offset(output.stride(), ishape, dim, i);
        for (size_t j = 0; j < k; j++) {
            std::nth_element(temp.begin(), temp.begin() + j, temp.end());
            output[offset + j * output.stride(dim)] = temp[j];
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

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        for (size_t j = 0; j < input.shape(dim); j++) {
            output[i] += input[offset + j * input.stride(dim)] / input.shape(dim);
        }
    }

    return output;
}

template<typename Iterator>
inline constexpr typename std::iterator_traits<Iterator>::value_type
    median(Iterator begin, Iterator end)
{
    size_t size = end - begin;
    if (size % 2 == 0) {
        auto med1 = begin + size / 2;
        auto med2 = begin + size / 2 - 1;
        std::nth_element(begin, med1, end);
        std::nth_element(begin, med2, end);
        return (*med1 + *med2) / 2.0f;
    } else {
        auto med = begin + size / 2;
        std::nth_element(begin, med, end);
        return *med;
    }
}

template<typename T>
inline constexpr T median(Tensor<T>& input)
{
    return median(input.begin(), input.end());
}

template<typename T>
inline constexpr Tensor<T> median(const Tensor<T>& input, size_t dim)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    Tensor<T> output(shape);
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
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
    T denom = std::max(std::size_t{1}, input.shape(dim) - 1);
    return sum(pow(abs(broadcast(input, mean(input, dim), gt::MINUS)), 2), dim) / denom;
}

template<typename T>
inline constexpr Tensor<T> stddev(const Tensor<T>& input, size_t dim)
{
    return sqrt(var(input, dim));
}

/* helper function for mode. similar to using std::max_element, but obeys the
 * convention that the smaller key is returned for values that are identical */
template<typename T>
inline constexpr T find_mode(const std::unordered_map<T,size_t>& map)
{
    T min_value = std::numeric_limits<T>::max();
    size_t max_size = 0;
    for (auto B = map.begin(), E = map.end(); B != E; ++B) {
        if (B->first < min_value && B->second >= max_size) {
            min_value = B->first;
            max_size = B->second;
        }
    }

    return min_value;
}

template<typename T>
inline constexpr T mode(const Tensor<T>& input)
{
    std::unordered_map<T,size_t> map;
    for (auto B = input.begin(), E = input.end(); B != E; ++B) {
        map[*B]++;
    }

    return find_mode(map);
}

template<typename T>
inline constexpr Tensor<T> mode(const Tensor<T>& input, size_t dim)
{
    std::vector<size_t> shape = input.shape();
    if (dim < shape.size()) {
        shape[dim] = 1;
    }

    Tensor<T> output(shape);
    std::unordered_map<T,size_t> map;
    for (size_t i = 0; i < output.size(); i++) {
        size_t offset = calculate_offset(input.stride(), shape, dim, i);
        for (size_t j = 0; j < input.shape(dim); j++) {
            map[input[offset + j * input.stride(dim)]]++;
        }
        output[i] = find_mode(map);
        map.clear();
    }

    return output;

}

} // namespace gt
