#pragma once

#include <cmath>

#include "Tensor.h"

namespace gt
{
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> linspace(T min, T max, size_t N)
    {
        Tensor<T> output({N});
        for (size_t i = 0; i < output.size(); i++) {
            output[i] = min + (max - min + 1) / N * i;
        }

        return output;
    }

    template<typename T>
    Tensor<T> logspace(T min, T max, size_t N)
    {
        return pow(10.0f, linspace(min, max, N));
    }

    #define UNARY_EXPRESSION(op) \
    template<typename T> \
    Tensor<T> op(const Tensor<T>& input) \
    { \
        Tensor<T> output(input.shape()); \
        for (size_t i = 0; i < output.size(); i++) { \
            output[i] = op(input[i]); \
        } \
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
        bool all = false;
        for (size_t i = 0; i < input.size(); i++) {
            all |= (input[i] != 0);
        }
        return all;
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

    //size_t calculate_offset(size_t index, const std::vector<size_t>& shape, const std::vector<size_t>& stride)
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
            for (size_t j = 0; j < input.shape(dim); j++) {
                running_total += input[offset + j * input.stride(dim)];
                output[i + j * input.stride(dim)] = running_total;
            }
        }
        return output;
    }

#if 0
    template<typename T>
    auto diff(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] -= 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                size_t offset = (index / input.stride(dim)) * input.stride(dim+1) + index % input.stride(dim);
                return input[offset + input.stride(dim)] - input[offset];
            }, shape
        };
    }

    template<typename T>
    auto prod(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                float output = 1;
                size_t offset = (index / input.stride(dim)) * input.stride(dim+1) + index % input.stride(dim);
                for (size_t i = 0; i < input.shape(dim); i++) {
                    output *= input[offset + i * input.stride(dim)];
                }
                return output;
            }, shape
        };
    }

    template<typename T>
    auto cumprod(const T& input, size_t dim)
    {
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                float output = 1;
                size_t offset = (index / input.stride(dim+1)) * input.stride(dim+1) + index % input.stride(dim);
                while (offset <= index) {
                    output *= input[offset];
                    offset += input.stride(dim);
                }
                return output;
            }, input.shape()
        };
    }

    template<typename T>
    auto trapz(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                float output = 0;
                size_t offset = (index / input.stride(dim)) * input.stride(dim+1) + index % input.stride(dim);
                for (size_t i = 0; i < input.shape(dim) - 1; i++) {
                    output += (input[offset + (i + 1) * input.stride(dim)] +
                        input[offset + i * input.stride(dim)]) / 2.0f;
                }
                return output;
            }, shape
        };
    }

    template<typename T>
    auto cumtrapz(const T& input, size_t dim)
    {
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                float output = 0;
                size_t offset = (index / input.stride(dim+1)) * input.stride(dim+1) + index % input.stride(dim);
                while (offset < index) {
                    output += (input[offset] + input[offset + input.stride(dim)]) / 2.0f;
                    offset += input.stride(dim);
                }
                return output;
            }, input.shape()
        };
    }

    template<typename T>
    auto max(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                size_t offset = (index / input.stride(dim)) * input.stride(dim+1) + index % input.stride(dim);
                float output = input[offset];
                for (size_t i = 0; i < input.shape(dim); i++) {
                    output = std::max(output, input[offset + i * input.stride(dim)]);
                }
                return output;
            }, shape
        };
    }

    template<typename T>
    auto cummax(const T& input, size_t dim)
    {
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                size_t offset = (index / input.stride(dim+1)) * input.stride(dim+1) + index % input.stride(dim);
                float output = input[offset];
                while (offset <= index) {
                    output = std::max(output, input[offset]);
                    offset += input.stride(dim);
                }
                return output;
            }, input.shape()
        };
    }

    template<typename T>
    auto min(const T& input, size_t dim)
    {
        std::vector<size_t> shape = input.shape();
        shape[dim] = 1;
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                size_t offset = (index / input.stride(dim)) * input.stride(dim+1) + index % input.stride(dim);
                float output = input[offset];
                for (size_t i = 0; i < input.shape(dim); i++) {
                    output = std::min(output, input[offset + i * input.stride(dim)]);
                }
                return output;
            }, shape
        };
    }

    template<typename T>
    auto cummin(const T& input, size_t dim)
    {
        return UnaryIndexExpression{input, [dim] (const auto& input, size_t index)
            {
                size_t offset = (index / input.stride(dim+1)) * input.stride(dim+1) + index % input.stride(dim);
                float output = input[offset];
                while (offset <= index) {
                    output = std::min(output, input[offset]);
                    offset += input.stride(dim);
                }
                return output;
            }, input.shape()
        };
    }

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

    template<typename T>
    auto reshape(const T& input, const std::initializer_list<size_t>& shape)
    {
        return UnaryExpression{input, [] (const auto& input)
            {
                return input;
            }, shape
        };
    }

    template<typename T>
    auto repmat(const T& input, const std::vector<size_t>& reps)
    {
        std::vector<size_t> shape(reps.size());
        for (size_t i = 0; i < reps.size(); i++) {
            shape[i] = input.shape(i) * reps[i];
        }
        std::vector<size_t> stride = calculate_stride(shape);

        return UnaryIndexExpression{input, [shape, stride] (const auto& input, size_t index)
            {
                size_t output = 0;
                for (size_t dim = 0; dim < input.shape().size(); dim++) {
                    output += ((index / stride[dim]) % shape[dim]) % input.shape(dim) * input.stride(dim);
                }
                return input[output];
            }, shape
        };
    }

    template<typename T>
    auto permute(const T& input, const std::vector<size_t>& order)
    {
        assert((input.shape().size() == order.size()) && "Error in permute: Size of order does not dimensionality of Tensor");
        std::vector<size_t> permuted_shape(order.size());
        std::vector<size_t> permuted_stride(order.size());
        for (size_t i = 0; i < order.size(); i++) {
            permuted_shape[i] = input.shape(order[i]);
            permuted_stride[i] = input.stride(order[i]);
        }
        std::vector<size_t> new_stride = calculate_stride(permuted_shape);

        return UnaryIndexExpression{input, [permuted_shape, permuted_stride, new_stride] (const auto& input, size_t index)
            {
                size_t output = 0;
                for (size_t dim = 0; dim < permuted_shape.size(); dim++) {
                    output += ((index / new_stride[dim]) % permuted_shape[dim]) * permuted_stride[dim];
                }
                return input[output];
            }, permuted_shape
        };
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
