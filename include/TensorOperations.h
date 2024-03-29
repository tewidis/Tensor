#pragma once

#include <cmath>

#include "Tensor.h"

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
    template<typename LHS> \
    Tensor<LHS> operator op(const Tensor<LHS>& lhs, LHS rhs) \
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
    UNARY_EXPRESSION(sqrt);
    UNARY_EXPRESSION(log);
    UNARY_EXPRESSION(log10);
    UNARY_EXPRESSION(exp);
    UNARY_EXPRESSION(floor);
    UNARY_EXPRESSION(ceil);
    UNARY_EXPRESSION(abs);

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
    template<typename T> requires std::is_arithmetic_v<T>
    T rem(T lhs, T rhs)
    {
        return lhs - fix(lhs / rhs) * rhs;
    }

    template<typename T>
    Tensor<T> rem(const Tensor<T>& lhs, T rhs)
    {
        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T value) { return rem(value, rhs); });

        return output;
    }

    /* modulus as computed by MATLAB */
    template<typename T> requires std::is_arithmetic_v<T>
    T mod(T lhs, T rhs)
    {
        return lhs - std::floor(lhs / rhs) * rhs;
    }

    template<typename T>
    Tensor<T> mod(const Tensor<T>& lhs, T rhs)
    {
        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T value) { return mod(value, rhs); });

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

        //TODO: Fix this, it's close but not quite right
        Tensor<T> output(shape);
        for (size_t dim = 0; dim < shape.size(); dim++) {
            for (size_t i = 0; i < shape[dim]; i++) {
                size_t index = input.stride(dim) * ((i / output.stride(dim)) % input.shape(dim));
                output[i] = input[index];
            }
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
#if 0
    template<typename T>
    auto mean(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                float output = 0;
                for (size_t i = 0; i < input.shape(dim); i++) {
                    output += input[index * input.shape(dim) + i * input.stride(dim)];
                }
                return output / input.shape(dim);
            }, shape
        };
    }

    template<typename T>
    auto var(const T& input, size_t dim)
    {
        std::vector<size_t> reps(input.shape().size());
        std::fill(reps.begin(), reps.end(), 1);
        reps[dim] = input.shape(dim);
        return sum(pow(input - repmat(mean(input, dim), reps), 2), dim) / input.shape(dim);
    }

    template<typename T>
    auto stddev(const T& input, size_t dim)
    {
        return sqrt(var(input, dim));
    }

    template<typename LHS, typename RHS>
    auto cat(size_t dim, const LHS& lhs, const RHS& rhs)
    {
        std::vector<size_t> shape(std::max(lhs.shape().size(), rhs.shape().size()));
        for (size_t i = 0; i < lhs.shape().size(); i++) {
            if (i != dim && lhs.shape(i) != 1 && rhs.shape(i) != 1) {
                assert(lhs.shape(i) == rhs.shape(i) && "Error in cat: Non-singleton dimensions must agree");
                shape[i] = lhs.shape(i);
            } else {
                shape[i] = lhs.shape(i) + rhs.shape(i);
            }
        }

        return BinaryIndexExpression{lhs, rhs, [dim] (const auto& lhs, const auto& rhs, size_t index)
            {
                //TODO: This works when the dimension is greater than 1, but not less than
                if (index < lhs.size()) {
                    return lhs[index];
                } else {
                    return rhs[index - lhs.size()];
                }
            }, shape
        };
    }

    template<typename... Ts>
    auto cat(size_t dim, Ts... tensors)
    {

    }

    template<typename LHS, typename RHS>
    auto matmul(const LHS& lhs, const RHS& rhs)
    {
        assert(lhs.shape().size() <= 2 && rhs.shape().size() <= 2 && 
            "Error in matmul: Matrix multiplication is only defined on vectors or matrices");
        assert(lhs.shape(1) == rhs.shape(0) && "Error in matmul: Dimension 1 of lhs does not match dimension 0 of rhs");

        std::vector<size_t> shape{lhs.shape(0), rhs.shape(1)};

        return BinaryIndexExpression{lhs, rhs, [shape] (const auto& lhs, const auto& rhs, size_t index)
            {
                float output = 0;
                size_t row = index % shape[0];
                size_t col = index / shape[0];
                for (size_t i = 0; i < lhs.shape(1); i++) {
                    output += lhs(row, i) * rhs(i, col);
                }
                return output;
            }, shape
        };
    }
#endif
};
