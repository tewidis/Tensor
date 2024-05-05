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
#include "fftw3.h"

#include "Enums.h"
#include "Statistics.h"
#include "Tensor.h"

namespace gt
{
    namespace sp
    {
        inline Tensor<std::complex<float>> fft(const Tensor<float>& input,
            size_t N, size_t dim = 0)
        {
            fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N/2+1));
            float* in = (float*) fftwf_malloc(sizeof(float) * N);
            std::vector<size_t> shape = input.shape();
            shape[dim] = N;
            Tensor<std::complex<float>> output(shape);
            fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
            assert(plan && "Error in fft: plan creation failed");

            for (size_t i = 0; i < input.size() / input.shape(dim); i++) {
                size_t offset_i = calculate_offset(input.stride(), input.shape(), dim, i);
                size_t offset_o = calculate_offset(output.stride(), output.shape(), dim, i);
                for (size_t j = 0; j < N; j++) {
                    in[j] = input[offset_i + j * input.stride(dim)];
                }

                fftwf_execute(plan);

                for (size_t j = 0; j < output.shape(dim); j++) {
                    if (j < N / 2 + 1) {
                        output[offset_o + j * output.stride(dim)] = {out[j][0], out[j][1]};
                    } else {
                        output[offset_o + j * output.stride(dim)] = {out[N-j][0], -out[N-j][1]};
                    }
                }
            }

            fftwf_free(in);
            fftwf_free(out);
            fftwf_destroy_plan(plan);

            return output;
        }

        inline Tensor<std::complex<float>> fft(const Tensor<std::complex<float>>& input,
            size_t N, size_t dim = 0)
        {
            fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
            fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N);
            std::vector<size_t> shape = input.shape();
            shape[dim] = N;
            Tensor<std::complex<float>> output(shape);
            fftwf_plan plan = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
            assert(plan && "Error in fft: plan creation failed");

            for (size_t i = 0; i < input.size() / input.shape(dim); i++) {
                size_t offset_i = calculate_offset(input.stride(), input.shape(), dim, i);
                size_t offset_o = calculate_offset(output.stride(), output.shape(), dim, i);
                for (size_t j = 0; j < N; j++) {
                    in[j][0] = std::real(input[offset_i + j * input.stride(dim)]);
                    in[j][1] = std::imag(input[offset_i + j * input.stride(dim)]);
                }

                fftwf_execute(plan);

                for (size_t j = 0; j < output.shape(dim); j++) {
                    output[offset_o + j * output.stride(dim)] = {out[j][0], out[j][1]};
                }
            }

            fftwf_free(in);
            fftwf_free(out);
            fftwf_destroy_plan(plan);

            return output;
        }
        inline Tensor<float> ifft(const Tensor<std::complex<float>>& input,
            size_t N, size_t dim = 0)
        {
            fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N/2+1));
            float* out = (float*) fftwf_malloc(sizeof(float) * N);
            std::vector<size_t> shape = input.shape();
            shape[dim] = N;
            Tensor<float> output(shape);
            fftwf_plan plan = fftwf_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
            assert(plan && "Error in ifft: plan creation failed");

            for (size_t i = 0; i < input.size() / input.shape(dim); i++) {
                size_t offset_i = calculate_offset(input.stride(), input.shape(), dim, i);
                size_t offset_o = calculate_offset(output.stride(), output.shape(), dim, i);
                for (size_t j = 0; j < N / 2 + 1; j++) {
                    in[j][0] = std::real(input[offset_i + j * input.stride(dim)]);
                    in[j][1] = std::imag(input[offset_i + j * input.stride(dim)]);
                }

                fftwf_execute(plan);

                for (size_t j = 0; j < output.shape(dim); j++) {
                    if (j < N / 2 + 1) {
                        output[offset_o + j * output.stride(dim)] = out[j] / N;
                    } else {
                        output[offset_o + j * output.stride(dim)] = out[j] / N;
                    }
                }
            }

            fftwf_free(in);
            fftwf_free(out);
            fftwf_destroy_plan(plan);

            return output;
        }

        template<typename T>
        inline constexpr Tensor<T> fftshift(const Tensor<T>& input, size_t dim)
        {
            return gt::circshift(input, input.shape(dim) / 2, dim);
        }

        template<typename T>
        inline constexpr Tensor<T> fftshift(const Tensor<T>& input)
        {
            std::vector<int64_t> nshift(ndims(input));
            for (size_t i = 0; i < nshift.size(); i++) {
                nshift[i] = input.shape(i) / 2;
            }

            return gt::circshift(input, nshift);
        }

        template<typename T>
        inline constexpr auto convolution_parameters(const Tensor<T>& lhs,
            const Tensor<T>& rhs, CONVOLUTION type)
        {
            std::vector<size_t> shape(ndims(lhs));
            std::vector<size_t> offset(ndims(lhs));

            for (size_t i = 0; i < shape.size(); i++) {
                switch (type) {
                    case FULL:
                    {
                        shape[i] = lhs.shape(i) + rhs.shape(i) - 1;
                        offset[i] = 0;
                        break;
                    }
                    case SAME:
                    {
                        shape[i] = lhs.shape(i);
                        offset[i] = rhs.shape(i) / 2;
                        break;
                    }
                    case VALID:
                    {
                        assert(lhs.shape(i) >= rhs.shape(i) && "Error in convolution: result will be empty");
                        shape[i] = lhs.shape(i) - rhs.shape(i) + 1;
                        offset[i] = rhs.shape(i) - 1;
                        break;
                    }
                }
            }

            return std::make_pair(shape, offset);
        }

        inline constexpr size_t convolution_limit(size_t i, size_t lhs_shape, size_t rhs_shape)
        {
            return std::min(
                std::min(lhs_shape, rhs_shape),
                std::min(i + 1, lhs_shape + rhs_shape - 1 - i)
            );
        }

        template<typename T>
        inline constexpr Tensor<T> conv1(const Tensor<T>& lhs, const Tensor<T>& rhs, CONVOLUTION type)
        {
            assert(ndims(lhs) == 1 && ndims(rhs) == 1 && 
                "Error in conv1: Tensors are not one-dimensional");
            auto parameters = convolution_parameters(lhs, rhs, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= rhs.shape(0) - 1 ? i - (rhs.shape(0) - 1) : 0;
                size_t i2 = std::min(i, rhs.shape(0) - 1);
                size_t limit = convolution_limit(i, lhs.shape(0), rhs.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    output(i - offset[0]) += lhs(i1 + j) * rhs(i2 - j);
                }
            }

            return output;
        }

        template<typename T>
        inline constexpr Tensor<T> conv2(const Tensor<T>& lhs, const Tensor<T>& rhs, CONVOLUTION type)
        {
            assert(ndims(lhs) == 2 && ndims(rhs) == 2 && 
                "Error in conv2: Tensors are not two-dimensional");
            auto parameters = convolution_parameters(lhs, rhs, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= rhs.shape(0) - 1 ? i - (rhs.shape(0) - 1) : 0;
                size_t i2 = std::min(i, rhs.shape(0) - 1);
                size_t limit = convolution_limit(i, lhs.shape(0), rhs.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    for (size_t k = offset[1]; k < offset[1] + output.shape(1); k++) {
                        size_t k1 = k >= rhs.shape(1) - 1 ? k - (rhs.shape(1) - 1) : 0;
                        size_t k2 = std::min(k, rhs.shape(1) - 1);
                        size_t limit = convolution_limit(k, lhs.shape(1), rhs.shape(1));
                        for (size_t l = 0; l < limit; l++) {
                            output(i - offset[0],k - offset[1]) += lhs(i1 + j,k1 + l) * rhs(i2 - j,k2 - l);
                        }
                    }
                }
            }

            return output;
        }

        template<typename T>
        inline constexpr Tensor<T> conv3(const Tensor<T>& lhs, const Tensor<T>& rhs, CONVOLUTION type)
        {
            assert(ndims(lhs) == 3 && ndims(rhs) == 3 && 
                "Error in conv3: Tensors are not three-dimensional");
            auto parameters = convolution_parameters(lhs, rhs, type);
            std::vector<size_t> shape = std::get<0>(parameters);
            std::vector<size_t> offset = std::get<1>(parameters);
            Tensor<T> output = gt::zeros<T>(shape);

            for (size_t i = offset[0]; i < offset[0] + output.shape(0); i++) {
                size_t i1 = i >= rhs.shape(0) - 1 ? i - (rhs.shape(0) - 1) : 0;
                size_t i2 = std::min(i, rhs.shape(0) - 1);
                size_t limit = convolution_limit(i, lhs.shape(0), rhs.shape(0));
                for (size_t j = 0; j < limit; j++) {
                    for (size_t k = offset[1]; k < offset[1] + output.shape(1); k++) {
                        size_t k1 = k >= rhs.shape(1) - 1 ? k - (rhs.shape(1) - 1) : 0;
                        size_t k2 = std::min(k, rhs.shape(1) - 1);
                        size_t limit = convolution_limit(k, lhs.shape(1), rhs.shape(1));
                        for (size_t l = 0; l < limit; l++) {
                            for (size_t m = offset[2]; m < offset[2] + output.shape(2); m++) {
                                size_t m1 = m >= rhs.shape(2) - 1 ? m - (rhs.shape(2) - 1) : 0;
                                size_t m2 = std::min(m, rhs.shape(2) - 1);
                                size_t limit = convolution_limit(m, lhs.shape(2), rhs.shape(2));
                                for (size_t n = 0; n < limit; n++) {
                                    output(i - offset[0],k - offset[1],m - offset[2]) += lhs(i1 + j,k1 + l,m1 + n) * rhs(i2 - j,k2 - l,m2 - n);
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

        inline Tensor<float> bartlett(size_t N)
        {
            Tensor<float> output = 2.0f * gencoswin(N);

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

        inline Tensor<float> cheb(size_t N, const Tensor<float>& input)
        {
            Tensor<float> output(input.shape());

            for (size_t i = 0; i < output.size(); i++) {
                if (std::abs(input(i)) <= 1) {
                    output(i) = std::cos(N * std::acos(input(i)));
                } else {
                    output(i) = std::pow(gt::sign(input(i)), N % 2)  *
                        std::cosh(N * std::acosh(std::abs(input(i))));
                }
            }

            return output;
        }

        inline Tensor<float> chebyshev(size_t N, float r = 100.0f)
        {
            Tensor<float> output({N});

            /* N == 1 is a special case */
            if (N == 1) {
                output(0) = 1.0f;
                return output;
            }

            float gamma = std::pow(10, -r / 20);
            float beta = std::cosh(1.0f / (N - 1) * std::acosh(1.0f / gamma));
            Tensor<float> k = gt::linspace(0.0f, N - 1.0f, N);
            Tensor<float> x = beta * gt::cos(PI * k / N);
            Tensor<float> p = cheb(N - 1, x);

            std::complex<float> j{0, 1};
            if (iseven(N)) {
                Tensor<std::complex<float>> sample({N});
                for (size_t i = 0; i < sample.size(); i++) {
                    sample(i) = p(i) * std::exp(j * PI /
                        static_cast<float>(N) * static_cast<float>(i));
                }
                Tensor<float> w = gt::real(gt::sp::fft(sample, sample.size()));
                w = w / w(1);

                size_t limit = N / 2;
                for (size_t i = 0; i < output.size(); i++) {
                    if (i < limit) {
                        output[i] = w[limit - i];
                    } else {
                        output[i] = w[i - limit + 1];
                    }
                }
            } else {
                Tensor<float> w = gt::real(gt::sp::fft(p, p.size()));
                w = w / w(0);

                size_t limit = (N + 1) / 2 - 1;
                for (size_t i = 0; i < output.size(); i++) {
                    if (i < limit) {
                        output[i] = w[limit - i];
                    } else {
                        output[i] = w[i - limit];
                    }
                }
            }
            output = output / gt::max(output);

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

        inline Tensor<float> gaussian(size_t N, float alpha = 2.5f)
        {
            float L = N - 1.0f;
            Tensor<float> output = gt::linspace(0.0f, L, N) - (L / 2.0f);
            output = gt::exp(-0.5f * gt::pow(alpha * output / (L / 2), 2.0f));

            return output;
        }

        inline Tensor<float> hamming(size_t N)
        {
            Tensor<float> output = 0.54f - 0.46f * gt::cos(2 * PI * gencoswin(N));

            return output;
        }

        inline Tensor<float> hann(size_t N)
        {
            Tensor<float> output = 0.5f - 0.5f * gt::cos(2 * PI * gencoswin(N));

            return output;
        }

        inline Tensor<float> hanning(size_t N)
        {
            float half = std::ceil(N / 2.0f);
            Tensor<float> output = gt::cat(0,
                gt::linspace(1.0f, half, half),
                gt::linspace(std::floor(N / 2.0f), 1.0f, N - half)) / (N + 1);
            output = 0.5f * (1 - gt::cos(2 * PI * output));

            return output;
        }

        template<typename T> requires std::is_arithmetic_v<T>
        inline Tensor<T> besseli(T nu, const Tensor<T>& beta)
        {
            Tensor<T> output(beta.shape());
            std::transform(beta.begin(), beta.end(), output.begin(),
                [nu] (const T value) { return std::cyl_bessel_i(nu, value); });
            return output;
        }

        inline Tensor<float> kaiser(size_t N, float beta = 0.5f)
        {
            float bes = std::abs(std::cyl_bessel_i(0.0f, beta));
            float odd = gt::rem(N, 2);
            float xind = std::pow(N - 1.0f, 2.0f);
            float n = gt::fix((N + 1.0f) / 2.0f);
            Tensor<float> xi = 4 * gt::pow(gt::linspace(0.0f, n - 1.0f, n) + 0.5f * (1 - odd), 2.0f);
            Tensor<float> w = besseli(0.0f, beta * gt::sqrt(1.0f - xi / xind)) / bes;

            Tensor<float> output({N});
            if (iseven(N)) {
                output = gt::abs(gt::cat(0, gt::flip(w), w));
            } else {
                size_t half = N / 2;
                for (size_t i = 0; i < N; i++) {
                    if (i < half) {
                        output(i) = std::abs(w(half - i));
                    } else {
                        output(i) = std::abs(w(i - half));
                    }
                }
            }

            return output;
        }

        inline Tensor<float> nuttall(size_t N)
        {
            assert(N >= 4 && "Error in nuttall: N must be greater than or equal to 4");

            Tensor<float> coeff({4});
            coeff = {0.3635819, -0.4891775, 0.1365995, -0.0106411};
            Tensor<float> increment({1, 4});
            increment = {0.0f, 1.0f, 2.0f, 3.0f};

            Tensor<float> output = 2 * PI * gt::linspace(0.0f, N - 1.0f, N) / (N - 1);
            output = gt::linalg::matmul(gt::cos(gt::broadcast(output, increment, gt::TIMES)), coeff);

            return output;
        }

        inline Tensor<float> parzen(int64_t N)
        {
            Tensor<float> k = gt::linspace(-(N - 1) / 2.0f, (N - 1) / 2.0f, N);
            Tensor<float> k1 = k(k < -(N - 1) / 4.0f);
            Tensor<float> k2 = k(gt::abs(k) <= (N - 1) / 4.0f);
            Tensor<float> w1 = 2 * gt::pow((1 - gt::abs(k1) / (N / 2.0f)), 3.0f);
            Tensor<float> w2 = 1 - 6 * gt::pow((gt::abs(k2) / (N / 2.0f)), 2.0f)
                + 6 * gt::pow(gt::abs(k2) / (N / 2.0f), 3.0f);
            Tensor<float> output = gt::cat(0, w1, w2, gt::flip(w1));

            return output;
        }

        inline Tensor<float> rect(size_t N)
        {
            return gt::ones<float>({N});
        }

        inline Tensor<float> taylor(size_t N, size_t nbar = 4, float sll = -30.0f)
        {
            assert(sll < 0 && "Error in taylor: sll must be less than 0");

            float A = std::acosh(std::pow(10, -sll / 20.0f)) / PI;
            float sp2 = std::pow(nbar, 2) / (std::pow(A, 2) + std::pow(nbar - 0.5f, 2));
            Tensor<float> k = gt::linspace(0.0f, N - 1.0f, N);
            Tensor<float> xi = (k - 0.5f * N + 0.5f) / N;
            Tensor<float> n = gt::linspace(1.0f, nbar - 1.0f, nbar - 1);

            Tensor<float> summation = gt::zeros<float>({N});
            for (size_t m = 1; m <= nbar - 1; m++) {
                Tensor<float> p = n(n != m);

                float num = gt::prod(1.0f - (std::pow(m, 2) / sp2) / (std::pow(A, 2) + gt::pow(n - 0.5f, 2)));
                float den = gt::prod(1.0f - std::pow(m, 2) / gt::pow(p, 2));
                float Fm = (std::pow(-1.0f, m + 1) * num) / (2.0f * den);

                summation = Fm * gt::cos(2 * PI * m * xi) + summation;
            }

            Tensor<float> output = gt::ones<float>({N});
            output = output + 2 * summation;
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

        inline Tensor<float> tukey(size_t N, float ratio = 0.5f)
        {
            Tensor<float> output({N});
            if (ratio <= 0) {
                output = gt::ones<float>({N});
            } else if (ratio >= 1) {
                output = gt::sp::hann(N);
            } else {
                Tensor<float> t = gt::linspace(0.0f, 1.0f, N);
                float per = ratio / 2.0f;
                float tl = std::floor(per * (N - 1)) + 1.0f;
                float th = N - tl;

                for (size_t i = 0; i < output.size(); i++) {
                    if (i < tl) {
                        output(i) = (1 + std::cos(PI / per * (t(i) - per))) / 2.0f;
                    } else if (i >= tl && i < th) {
                        output(i) = 1.0f;
                    } else {
                        output(i) = (1 + std::cos(PI / per * (t(i) - 1 + per))) / 2.0f;
                    }
                }
            }

            return output;
        }
    }
}
