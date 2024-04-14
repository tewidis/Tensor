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
    namespace linalg
    {
        inline Tensor<float> matmul(const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            assert(lhs.shape().size() <= 2 && rhs.shape().size() <= 2 && 
                "Error in matmul: Matrix multiplication is only defined on vectors or matrices");
            assert(lhs.shape(1) == rhs.shape(0) && "Error in matmul: Dimension 1 of lhs does not match dimension 0 of rhs");

            const size_t M = lhs.shape(0);
            const size_t N = rhs.shape(1);
            const size_t K = lhs.shape(1);
            const float alpha = 1.0f;
            const float beta = 0.0f;

            Tensor<float> output({lhs.shape(0), rhs.shape(1)});
            if (N == 1) {
                cblas_sgemv(CblasColMajor, CblasNoTrans, M, K, alpha, lhs.data(),
                    lhs.shape(0), rhs.data(), 1, beta, output.data(), 1);
            } else {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, lhs.data(), lhs.shape(0), rhs.data(), rhs.shape(0), beta,
                    output.data(), output.shape(0));
            }

            return output;
        }

        inline Tensor<double> matmul(const Tensor<double>& lhs, const Tensor<double>& rhs)
        {
            assert(lhs.shape().size() <= 2 && rhs.shape().size() <= 2 && 
                "Error in matmul: Matrix multiplication is only defined on vectors or matrices");
            assert(lhs.shape(1) == rhs.shape(0) && "Error in matmul: Dimension 1 of lhs does not match dimension 0 of rhs");

            const size_t M = lhs.shape(0);
            const size_t N = rhs.shape(1);
            const size_t K = lhs.shape(1);
            const double alpha = 1.0;
            const double beta = 0.0;

            Tensor<double> output({lhs.shape(0), rhs.shape(1)});
            if (N == 1) {
                cblas_dgemv(CblasColMajor, CblasNoTrans, M, K, alpha, lhs.data(),
                    lhs.shape(0), rhs.data(), 1, beta, output.data(), 1);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    alpha, lhs.data(), lhs.shape(0), rhs.data(), rhs.shape(0), beta,
                    output.data(), output.shape(0));
            }

            return output;
        }

        inline Tensor<std::complex<float>> matmul(const Tensor<std::complex<float>>& lhs, const Tensor<std::complex<float>>& rhs)
        {
            assert(lhs.shape().size() <= 2 && rhs.shape().size() <= 2 && 
                "Error in matmul: Matrix multiplication is only defined on vectors or matrices");
            assert(lhs.shape(1) == rhs.shape(0) && "Error in matmul: Dimension 1 of lhs does not match dimension 0 of rhs");

            const size_t M = lhs.shape(0);
            const size_t N = rhs.shape(1);
            const size_t K = lhs.shape(1);
            const std::complex<float> alpha{1.0f, 0.0f};
            const std::complex<float> beta{0.0f, 0.0f};

            Tensor<std::complex<float>> output({lhs.shape(0), rhs.shape(1)});
            if (N == 1) {
                cblas_cgemv(CblasColMajor, CblasNoTrans, M, K, &alpha, lhs.data(),
                    lhs.shape(0), rhs.data(), 1, &beta, output.data(), 1);
            } else {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    &alpha, lhs.data(), lhs.shape(0), rhs.data(), rhs.shape(0), &beta,
                    output.data(), output.shape(0));
            }

            return output;
        }

        inline Tensor<std::complex<double>> matmul(const Tensor<std::complex<double>>& lhs, const Tensor<std::complex<double>>& rhs)
        {
            assert(lhs.shape().size() <= 2 && rhs.shape().size() <= 2 && 
                "Error in matmul: Matrix multiplication is only defined on vectors or matrices");
            assert(lhs.shape(1) == rhs.shape(0) && "Error in matmul: Dimension 1 of lhs does not match dimension 0 of rhs");

            const size_t M = lhs.shape(0);
            const size_t N = rhs.shape(1);
            const size_t K = lhs.shape(1);
            const std::complex<double> alpha{1.0f, 0.0f};
            const std::complex<double> beta{0.0f, 0.0f};

            Tensor<std::complex<double>> output({lhs.shape(0), rhs.shape(1)});
            if (N == 1) {
                cblas_zgemv(CblasColMajor, CblasNoTrans, M, K, &alpha, lhs.data(),
                    lhs.shape(0), rhs.data(), 1, &beta, output.data(), 1);
            } else {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    &alpha, lhs.data(), lhs.shape(0), rhs.data(), rhs.shape(0), &beta,
                    output.data(), output.shape(0));
            }

            return output;
        }
    }
}
