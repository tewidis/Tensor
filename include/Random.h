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
        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> rand(const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_real_distribution distribution(0.0, 1.0);
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

            return output;
        }

        template<typename T> requires std::is_integral_v<T>
        inline constexpr Tensor<T> randi(T imax, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_int_distribution distribution(1, imax);
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

            return output;
        }

        template<typename T> requires std::is_integral_v<T>
        inline constexpr Tensor<T> randi(const std::tuple<T,T>& limits, const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::uniform_int_distribution distribution(std::get<0>(limits), std::get<1>(limits));
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline constexpr Tensor<T> randn(const std::vector<size_t>& shape)
        {
            Tensor<T> output(shape);

            std::mt19937 engine;
            std::normal_distribution distribution(0.0, 1.0);
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

            return output;
        }

        template<typename T> requires std::is_floating_point_v<T>
        inline Tensor<bool> bernoulli(T p, const std::vector<size_t>& shape)
        {
            assert(0.0 <= p && p <= 1.0 &&
                "Error in bernoulli: p must be between 0 and 1");

            Tensor<bool> output(shape);

            std::mt19937 engine;
            std::bernoulli_distribution distribution(p);
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

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
            std::binomial_distribution<T1> distribution(p);
            for (size_t i = 0; i < output.size(); i++) {
                output[i] = distribution(engine);
            }

            return output;
        }
    }
}
