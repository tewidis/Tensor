#pragma once

#include "cblas.h"

#include "Tensor.h"

namespace gt
{
    namespace linalg
    {
        Tensor<float> matmul(const Tensor<float>& lhs, const Tensor<float>& rhs)
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

        Tensor<double> matmul(const Tensor<double>& lhs, const Tensor<double>& rhs)
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

        Tensor<std::complex<float>> matmul(const Tensor<std::complex<float>>& lhs, const Tensor<std::complex<float>>& rhs)
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

        Tensor<std::complex<double>> matmul(const Tensor<std::complex<double>>& lhs, const Tensor<std::complex<double>>& rhs)
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
