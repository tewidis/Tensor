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

        inline Tensor<float> gencoswin(size_t N)
        {
            float half = std::ceil(N / 2.0f);
            Tensor<float> output = gt::cat(0,
                gt::linspace(0.0f, half - 1.0f, half),
                gt::linspace(std::floor(N / 2.0f) - 1.0f, 0.0f, N - half)) / (N - 1);

            return output;
        }

        inline Tensor<float> rect(size_t N)
        {
            return gt::ones<float>({N});
        }

        inline Tensor<float> bartlett(size_t N)
        {
            Tensor<float> output = 2.0f * gencoswin(N);

            return output;
        }

        inline Tensor<float> triang(size_t N)
        {
            float half = std::ceil(N / 2.0f);
            Tensor<float> output({N});
            if (iseven(N)) {
                output = (2 * gt::cat(0,
                    gt::linspace(1.0f, half, half),
                    gt::linspace(half, 1.0f, N - half)) - 1) / N;
            } else {
                output = (2 * gt::cat(0,
                    gt::linspace(1.0f, half, half),
                    gt::linspace(half - 1.0f, 1.0f, N - half))) / (N + 1);
            }

            return output;
        }

        inline Tensor<float> barthann(size_t N)
        {
            Tensor<float> output = gt::linspace(0.0f, 1.0f, N) - 0.5f;
            output = 0.62 - 0.48 * gt::abs(output) + 0.38 * gt::cos(2 * PI * output);

            return output;
        }

        inline Tensor<float> blackman(size_t N)
        {
            Tensor<float> output = gencoswin(N);
            output = 0.42 - 0.5 * gt::cos(2 * PI * output) + 0.08 * gt::cos(4 * PI * output);

            return output;
        }

        inline Tensor<float> blackmanharris(size_t N)
        {
            assert(N >= 4 && "Error in blackmanharris: N must be greater than or equal to 4");

            Tensor<float> coeff({4});
            coeff = {0.35875, -0.48829, 0.14128, -0.01168};
            Tensor<float> increment({1, 4});
            increment = {0.0f, 1.0f, 2.0f, 3.0f};

            Tensor<float> output = 2 * PI * gt::linspace(0.0f, N - 1.0f, N) / (N - 1);
            output = gt::linalg::matmul(gt::cos(gt::broadcast(output, increment, gt::TIMES)), coeff);

            return output;
        }

        inline Tensor<float> bohman(size_t N)
        {
            Tensor<float> output = gt::abs(gt::linspace(-1.0f, 1.0f, N));
            output = (1.0f - output) * gt::cos(PI * output) + (1 / PI) * gt::sin(PI * output);
            output(0) = 0.0f;
            output(N - 1) = 0.0f;

            return output;
        }

        inline Tensor<float> chebyshev(size_t N)
        {
            Tensor<float> output({N});

            return output;
        }

        inline Tensor<float> flattop(size_t N)
        {
            const float a0 = 0.21557895;
            const float a1 = 0.41663158;
            const float a2 = 0.277263158;
            const float a3 = 0.083578947;
            const float a4 = 0.006947368;

            Tensor<float> output = gencoswin(N);
            output = a0 - a1 * gt::cos(2 * PI * output)
                + a2 * gt::cos(4 * PI * output)
                - a3 * gt::cos(6 * PI * output)
                + a4 * gt::cos(8 * PI * output);

            return output;
        }

        inline Tensor<float> gaussian(size_t N)
        {
            const float alpha = 2.5f;

            float L = N - 1.0f;
            Tensor<float> output = gt::linspace(0.0f, L, N) - (L / 2.0f);
            output = gt::exp(-0.5f * gt::pow(alpha * output / (L / 2), 2.0f));

            return output;
        }
    }
}
