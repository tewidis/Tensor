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
    UNARY_EXPRESSION(abs);
    UNARY_EXPRESSION(round);

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
    inline constexpr Tensor<T> reshape(const Tensor<T>& input, const std::vector<size_t>& shape)
    {
        Tensor<T> output(shape);
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = input[i];
        }
        //std::copy(input.begin(), input.end(), output.begin());
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

    enum OPERATION {
        PLUS,
        MINUS,
        TIMES,
        DIVIDE,
        POWER,
        MAX,
        MIN,
        MOD,
        REM,
        ATAN2,
        ATAN2D,
        HYPOT
    };

    template<typename T>
    constexpr inline Tensor<T> broadcast(const Tensor<T>& t1, const Tensor<T>& t2, OPERATION op)
    {
        std::vector<size_t> shape(std::max(t1.shape().size(), t2.shape().size()));
        for (size_t i = 0; i < shape.size(); i++) {
            shape[i] = std::max(t1.shape(i), t2.shape(i));
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t lidx = 0;
            size_t ridx = 0;
            for (size_t j = 0; j < output.shape().size(); j++) {
                lidx += t1.stride(j) * ((i / output.stride(j)) % t1.shape(j));
                ridx += t2.stride(j) * ((i / output.stride(j)) % t2.shape(j));
            }

            switch (op) {
                case PLUS:
                    output[i] = t1[lidx] + t2[ridx];
                    break;
                case MINUS:
                    output[i] = t1[lidx] - t2[ridx];
                    break;
                case TIMES:
                    output[i] = t1[lidx] * t2[ridx];
                    break;
                case DIVIDE:
                    output[i] = t1[lidx] / t2[ridx];
                    break;
                case POWER:
                    output[i] = std::pow(t1[lidx], t2[ridx]);
                    break;
                case MAX:
                    output[i] = gt::max(t1[lidx], t2[ridx]);
                    break;
                case MIN:
                    output[i] = gt::min(t1[lidx], t2[ridx]);
                    break;
                case MOD:
                    output[i] = gt::mod(t1[lidx], t2[ridx]);
                    break;
                case REM:
                    output[i] = gt::rem(t1[lidx], t2[ridx]);
                    break;
                case ATAN2:
                    output[i] = std::atan2(t1[lidx], t2[ridx]);
                    break;
                case ATAN2D:
                    output[i] = gt::atan2d(t1[lidx], t2[ridx]);
                    break;
                case HYPOT:
                    output[i] = std::hypot(t1[lidx], t2[ridx]);
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
        std::vector<size_t> subs(input.shape().size());
        for (size_t i = 0; i < subs.size(); i++) {
            subs[i] = input.stride(i) * ((index / input.stride(i)) % input.shape(i));
        }

        return subs;
    }
};
