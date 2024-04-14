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

#include "cblas.h"

#include "Tensor.h"

namespace gt
{
    enum CONVOLUTION {
        FULL,
        SAME,
        VALID
    };

    namespace sp
    {
        template<typename T>
        inline constexpr auto convolution_parameters(const Tensor<T>& t1,
            const Tensor<T>& t2, CONVOLUTION type)
        {
            std::vector<size_t> shape(t1.shape().size());
            std::vector<size_t> offset(t1.shape().size());

            for (size_t i = 0; i < shape.size(); i++) {
                switch (type) {
                    case FULL:
                    {
                        shape[i] = t1.shape(i) + t2.shape(i) - 1;
                        offset[i] = 0;
                        break;
                    }
                    case SAME:
                    {
                        shape[i] = t1.shape(i);
                        offset[i] = t2.shape(i) / 2;
                        break;
                    }
                    case VALID:
                    {
                        assert(t1.shape(i) >= t2.shape(i) && "Error in convolution: result will be empty");
                        shape[i] = t1.shape(i) - t2.shape(i) + 1;
                        offset[i] = t2.shape(i) - 1;
                        break;
                    }
                }
            }

            return std::make_pair(shape, offset);
        }

        inline constexpr size_t convolution_limit(size_t i, size_t t1_shape, size_t t2_shape)
        {
            return std::min(
                std::min(t1_shape, t2_shape),
                std::min(i + 1, t1_shape + t2_shape - 1 - i)
            );
        }

        template<typename T>
        inline constexpr Tensor<T> conv1(const Tensor<T>& t1, const Tensor<T>& t2, CONVOLUTION type)
        {
            assert(t1.shape().size() == 1 && t2.shape().size() == 1 && 
                "Error in conv1: Tensors are not one-dimensional");
            auto parameters = convolution_parameters(t1, t2, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= t2.shape(0) - 1 ? i - (t2.shape(0) - 1) : 0;
                size_t i2 = std::min(i, t2.shape(0) - 1);
                size_t limit = convolution_limit(i, t1.shape(0), t2.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    output(i - offset[0]) += t1(i1 + j) * t2(i2 - j);
                }
            }

            return output;
        }

        template<typename T>
        inline constexpr Tensor<T> conv2(const Tensor<T>& t1, const Tensor<T>& t2, CONVOLUTION type)
        {
            assert(t1.shape().size() == 2 && t2.shape().size() == 2 && 
                "Error in conv2: Tensors are not two-dimensional");
            auto parameters = convolution_parameters(t1, t2, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= t2.shape(0) - 1 ? i - (t2.shape(0) - 1) : 0;
                size_t i2 = std::min(i, t2.shape(0) - 1);
                size_t limit = convolution_limit(i, t1.shape(0), t2.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    for (size_t k = offset[1]; k < offset[1] + output.shape(1); k++) {
                        size_t k1 = k >= t2.shape(1) - 1 ? k - (t2.shape(1) - 1) : 0;
                        size_t k2 = std::min(k, t2.shape(1) - 1);
                        size_t limit = convolution_limit(k, t1.shape(1), t2.shape(1));
                        for (size_t l = 0; l < limit; l++) {
                            output(i - offset[0],k - offset[1]) += t1(i1 + j,k1 + l) * t2(i2 - j,k2 - l);
                        }
                    }
                }
            }

            return output;
        }

        template<typename T>
        inline constexpr Tensor<T> conv3(const Tensor<T>& t1, const Tensor<T>& t2, CONVOLUTION type)
        {
            assert(t1.shape().size() == 3 && t2.shape().size() == 3 && 
                "Error in conv3: Tensors are not three-dimensional");
            auto parameters = convolution_parameters(t1, t2, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= t2.shape(0) - 1 ? i - (t2.shape(0) - 1) : 0;
                size_t i2 = std::min(i, t2.shape(0) - 1);
                size_t limit = convolution_limit(i, t1.shape(0), t2.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    for (size_t k = offset[1]; k < offset[1] + output.shape(1); k++) {
                        size_t k1 = k >= t2.shape(1) - 1 ? k - (t2.shape(1) - 1) : 0;
                        size_t k2 = std::min(k, t2.shape(1) - 1);
                        size_t limit = convolution_limit(k, t1.shape(1), t2.shape(1));
                        for (size_t l = 0; l < limit; l++) {
                            for (size_t m = offset[2]; m < offset[2] + output.shape(2); m++) {
                                size_t m1 = m >= t2.shape(2) - 1 ? m - (t2.shape(2) - 1) : 0;
                                size_t m2 = std::min(m, t2.shape(2) - 1);
                                size_t limit = convolution_limit(m, t1.shape(2), t2.shape(2));
                                for (size_t n = 0; n < limit; n++) {
                                    output(i - offset[0],k - offset[1],m - offset[2]) += t1(i1 + j,k1 + l,m1 + n) * t2(i2 - j,k2 - l,m2 - n);
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }
    }
}
