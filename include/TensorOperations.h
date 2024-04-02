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
    Tensor<T> op(const Tensor<T>& input) \
    { \
        Tensor<T> output(input.shape()); \
        std::transform(input.begin(), input.end(), output.begin(), \
            [] (T value) { return std::op(value); }); \
        return output; \
    }

    //TODO: Need to be smarter about handling type promotion here
    #define BINARY_EXPRESSION(op) \
    template<typename LHS, typename RHS> \
    Tensor<LHS> operator op(const Tensor<LHS>& lhs, const Tensor<RHS>& rhs) \
    { \
        assert((lhs.size() == rhs.size()) && "Error in binary operation: Tensors are different shapes"); \
        Tensor<LHS> output(lhs.shape()); \
        for (size_t i = 0; i < output.size(); i++) { \
            output[i] = lhs[i] op rhs[i]; \
        } \
        return output; \
    } \
    template<typename LHS, typename RHS> requires std::is_arithmetic_v<RHS>\
    Tensor<LHS> operator op(const Tensor<LHS>& lhs, RHS rhs) \
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
    bool all(const Tensor<T>& input)
    {
        bool all = true;
        for (size_t i = 0; i < input.size(); i++) {
            all &= (input[i] != 0);
        }
        return all;
    }

    template<typename T>
    bool any(const Tensor<T>& input)
    {
        bool any = false;
        for (size_t i = 0; i < input.size(); i++) {
            any |= (input[i] != 0);
        }
        return any;
    }

    /* rounds towards 0 */
    template<typename T> requires std::is_arithmetic_v<T>
    T fix(T input)
    {
        return (input < 0) ? std::ceil(input) : std::floor(input);
    }

    template<typename T>
    Tensor<T> fix(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        std::transform(input.begin(), input.end(), output.begin(),
            [] (T value) { return fix(value); });

        return output;
    }

    /* remainder as computed by MATLAB */
    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    T1 rem(T1 lhs, T2 rhs)
    {
        return lhs - fix(lhs / rhs) * rhs;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> rem(const Tensor<T2>& lhs, T1 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return rem(value, rhs); });

        return output;
    }

    /* modulus as computed by MATLAB */
    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    T1 mod(T1 lhs, T2 rhs)
    {
        return lhs - std::floor(lhs / rhs) * rhs;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> mod(const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return mod(value, rhs); });

        return output;
    }

    /* produces N linearly spaced points between min and max */
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> linspace(T min, T max, size_t N)
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
    Tensor<T> logspace(T min, T max, size_t N)
    {
        assert((N > 0) && "Error in logspace: N must be greater than 0");
        return pow(10.0f, linspace(min, max, N));
    }

    template<typename T>
    Tensor<T> pow(const Tensor<T>& input, float scalar)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::pow(input[i], scalar);
        }
        return output;
    }

    template<typename T>
    Tensor<T> pow(float scalar, const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::pow(scalar, input[i]);
        }
        return output;
    }

    template<typename T>
    size_t calculate_offset(const gt::Tensor<T>& input, size_t index, size_t dim)
    {
        assert((dim < input.shape().size()) && "Error in calculate offset: Dimension exceeds dimensions of input");
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        std::vector<size_t> stride = calculate_stride(shape);

        size_t offset = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            offset += input.stride(i) * ((index / stride[i]) % shape[i]);
        }

        return offset;
    }

    template<typename T>
    Tensor<T> sum(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] += input[offset + j * input.stride(dim)];
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> cumsum(const Tensor<T>& input, size_t dim)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input, i, dim);
            T running_total = 0;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total += input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> diff(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] -= 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            output[i] = input[offset + input.stride(dim)] - input[offset];
        }

        return output;
    }

    template<typename T>
    Tensor<T> prod(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            output[i] = 1;
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] *= input[offset + j * input.stride(dim)];
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> cumprod(const Tensor<T>& input, size_t dim)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input, i, dim);
            T running_total = 1;
            for (size_t j = 0; j < output.shape(dim); j++) {
                running_total *= input[offset + j * input.stride(dim)];
                output[offset + j * input.stride(dim)] = running_total;
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> trapz(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            for (size_t j = 1; j < input.shape(dim); j++) {
                output[i] += (input[offset + j * input.stride(dim)] +
                    input[offset + (j - 1) * input.stride(dim)]) / 2.0f;
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> cumtrapz(const Tensor<T>& input, size_t dim)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input, i, dim);
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
    Tensor<T> max(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            output[i] = input[offset];
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] = std::max(output[i], input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> cummax(const Tensor<T>& input, size_t dim)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input, i, dim);
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
    Tensor<T> min(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            output[i] = input[offset];
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] = std::min(output[i], input[offset + j * input.stride(dim)]);
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> cummin(const Tensor<T>& input, size_t dim)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t offset = calculate_offset(input, i, dim);
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
    Tensor<T> reshape(const Tensor<T>& input, const std::vector<size_t>& shape)
    {
        Tensor<T> output(shape);
        std::copy(input.begin(), input.end(), output.begin());
        return output;
    }

    template<typename T>
    Tensor<T> squeeze(const Tensor<T>& input)
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
    Tensor<T> flatten(const Tensor<T>& input)
    {
        return reshape(input, {input.size()});
    }

    template<typename T>
    Tensor<T> repmat(const Tensor<T>& input, const std::vector<size_t>& reps)
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

    size_t calculate_permute_index(const std::vector<size_t>& shape,
        const std::vector<size_t>& stride, const std::vector<size_t>& step,
        size_t index)
    {
        size_t output = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            output += step[i] * ((index / stride[i]) % shape[i]);
        }

        return output;
    }

    template<typename T>
    Tensor<T> permute(const Tensor<T>& input, const std::vector<size_t>& order)
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
    Tensor<T> ipermute(const Tensor<T>& input, const std::vector<size_t>& order)
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
    Tensor<T> mean(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            for (size_t j = 0; j < input.shape(dim); j++) {
                output[i] += input[offset + j * input.stride(dim)] / input.shape(dim);
            }
        }

        return output;
    }

    template<typename T>
    T median(Tensor<T>& input)
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
    Tensor<T> median(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        if (dim < shape.size()) {
            shape[dim] = 1;
        }

        Tensor<T> output(shape);
        for (size_t i = 0; i < output.size(); i++) {
            size_t offset = calculate_offset(input, i, dim);
            Tensor<T> temp({input.shape(dim)});
            for (size_t j = 0; j < input.shape(dim); j++) {
                temp[j] = input[offset + j * input.stride(dim)];
            }
            output[i] = median(temp);
        }

        return output;
    }

    template<typename T>
    Tensor<T> var(const Tensor<T>& input, size_t dim)
    {
        std::vector<size_t> reps(input.shape().size());
        std::fill(reps.begin(), reps.end(), 1);
        reps[dim] = input.shape(dim);
        size_t denom = std::max(std::size_t{1}, input.shape(dim) - 1);
        return sum(pow(abs(input - repmat(mean(input, dim), reps)), 2), dim) / denom;
    }

    template<typename T>
    Tensor<T> stddev(const Tensor<T>& input, size_t dim)
    {
        return sqrt(var(input, dim));
    }

    template<typename T>
    Tensor<T> circshift(const Tensor<T>& input, int64_t nshift, size_t dim = 0)
    {
        nshift = input.shape(dim) - rem(nshift, input.shape(dim));

        Tensor<T> output(input.shape());
        size_t modulo = input.shape(dim) * input.stride(dim);
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t index = calculate_offset(input, i, dim);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[index + j * output.stride(dim)] =
                    input[index + (j + nshift) * output.stride(dim) % modulo];
            }
        }

        return output;
    }

    template<typename T>
    Tensor<T> flip(const Tensor<T>& input, size_t dim = 0)
    {
        Tensor<T> output(input.shape());
        for (size_t i = 0; i < output.size() / output.shape(dim); i++) {
            size_t index = calculate_offset(input, i, dim);
            for (size_t j = 0; j < output.shape(dim); j++) {
                output[index + j * output.stride(dim)] =
                    input[index + (input.shape(dim) - j - 1) * input.stride(dim)];
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
    Tensor<T> cart2sph(const Tensor<T>& input)
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
    Tensor<T> sph2cart(const Tensor<T>& input)
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
    Tensor<T> cart2pol(const Tensor<T>& input)
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
    Tensor<T> pol2cart(const Tensor<T>& input)
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
    Tensor<T> cat(size_t dim, const Tensor<T>& lhs, const Tensor<T>& rhs)
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
    Tensor<T> cat(size_t dim, const Tensor<T>& arg, const Tensor<Ts...>& args)
    {
        return cat(dim, arg, cat(dim, args));
    }

    template<typename T>
    Tensor<T> unique(const Tensor<T>& input)
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
};
