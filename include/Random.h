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

#include <random>

#include "Tensor.h"

namespace gt
{
    namespace rand
    {
        /* uniform distribtions */
        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> rand(const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_real_distribution distribution(0.0, 1.0);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_integral_v<T>
        inline constexpr Tensor<T> randi(T imax, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_int_distribution distribution(1, imax);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_integral_v<T>
        inline constexpr Tensor<T> randi(const std::tuple<T,T>& limits, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_int_distribution distribution(std::get<0>(limits), std::get<1>(limits));
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        /* bernoulli distributions */
        template<typename T> requires std::is_floating_point_v<T>
        inline Tensor<bool> bernoulli(T p, const std::vector<size_t>& shape)
        {
            assert(0.0 <= p && p <= 1.0 &&
                "Error in bernoulli: p must be between 0 and 1");

            Tensor<bool> output(shape);

            std::mt19937 engine;
            std::bernoulli_distribution distribution(p);
            std::for_each(output.begin(), output.end(),
                [&] (bool& value) { value = distribution(engine); });

            return output;
        }

        template<typename T1, typename T2>
            requires std::is_integral_v<T1> && std::is_floating_point_v<T2>
        inline Tensor<T1> binomial(T1 t, T2 p, const std::vector<size_t>& shape)
        {
            assert(0.0 <= p && p <= 1.0 &&
                "Error in binomial: p must be between 0 and 1");
            assert(t > 0 && "Error in binomial: t must be greater than 0");

            Tensor<T1> output(shape);

            std::mt19937 engine;
            std::binomial_distribution distribution(t, p);
            std::for_each(output.begin(), output.end(),
                [&] (T1& value) { value = distribution(engine); });

            return output;
        }

        template<typename T1, typename T2>
            requires std::is_integral_v<T1> && std::is_floating_point_v<T2>
        inline Tensor<T1> negative_binomial(T1 k, T2 p, const std::vector<size_t>& shape)
        {
            assert(0.0 <= p && p <= 1.0 &&
                "Error in negative_binomial: p must be between 0 and 1");
            assert(k > 0 && "Error in negative_binomial: k must be greater than 0");

            Tensor<T1> output(shape);

            std::mt19937 engine;
            std::negative_binomial_distribution distribution(k, p);
            std::for_each(output.begin(), output.end(),
                [&] (T1& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline Tensor<T> geometric(T p, const std::vector<size_t>& shape)
        {
            assert(0.0 <= p && p <= 1.0 &&
                "Error in geometric: p must be between 0 and 1");

            Tensor<T> output(shape);

            std::mt19937 engine;
            std::geometric_distribution distribution(p);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        /* poisson distributions */
        template<typename T> requires std::is_integral_v<T>
        inline constexpr Tensor<T> poisson(T n, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::poisson_distribution distribution(n);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> exponential(T n, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::exponential_distribution distribution(n);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> gamma(T alpha, T beta, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::gamma_distribution distribution(alpha, beta);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> weibull(T a, T b, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::weibull_distribution distribution(a, b);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> extreme_value(T a, T b, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::extreme_value_distribution distribution(a, b);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        /* normal distributions */
        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> randn(T mean, T stddev, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::normal_distribution distribution(mean, stddev);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> lognormal(T mean, T stddev, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::lognormal_distribution distribution(mean, stddev);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> chi_squared(T dof, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::chi_squared_distribution distribution(dof);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> cauchy(T a, T b, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::cauchy_distribution distribution(a, b);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> fisher_f(T dof1, T dof2, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::fisher_f_distribution distribution(dof1, dof2);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> student_t(T dof, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::student_t_distribution distribution(dof);
            std::for_each(output.begin(), output.end(),
                [&] (T& value) { value = distribution(engine); });

            return output;
        }
    }
}
