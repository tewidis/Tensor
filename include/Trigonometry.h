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
    constexpr float PI = std::acos(-1.0f);

    /* Conversions */
    template<typename T> requires std::is_floating_point_v<T>
    constexpr inline T deg2rad(T input)
    {
        return input * PI / 180.0f;
    }

    template<typename T> requires std::is_floating_point_v<T>
    constexpr inline Tensor<T> deg2rad(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return deg2rad(x); });

        return output;
    }

    template<typename T> requires std::is_floating_point_v<T>
    constexpr inline T rad2deg(T input)
    {
        return input / PI * 180.0f;
    }

    template<typename T> requires std::is_floating_point_v<T>
    constexpr inline Tensor<T> rad2deg(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return rad2deg(x); });

        return output;
    }

    /* input is an Nx3 Tensor where the zeroth dimension is the x-coordinate,
     * the first dimension is the y-coordinate, and the second dimension is the
     * z-coordinate
     * output is an Nx3 Tensor where the zeroth dimension is the azimuth,
     * the first dimension is the elevation, and the second dimension is the
     * radius */
    template<typename T>
    inline constexpr Tensor<T> cart2sph(const Tensor<T>& input)
    {
        assert(ndims(input) == 2 && input.shape(1) == 3);

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
    inline constexpr Tensor<T> sph2cart(const Tensor<T>& input)
    {
        assert(ndims(input) == 2 && input.shape(1) == 3);

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
    inline constexpr Tensor<T> cart2pol(const Tensor<T>& input)
    {
        assert(ndims(input) == 2 && input.shape(1) == 3);

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
    inline constexpr Tensor<T> pol2cart(const Tensor<T>& input)
    {
        assert(ndims(input) == 2 && input.shape(1) == 3);

        Tensor<T> output(input.shape());
        for (size_t i = 0; i < input.shape(0); i++) {
            output(i,0) = input(i,1) * std::cos(input(i,0));
            output(i,1) = input(i,1) * std::sin(input(i,0));
            output(i,2) = input(i,2);
        }

        return output;
    }


    /* Sine */
    template<typename T>
    constexpr inline Tensor<T> sin(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::sin(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> sind(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::sin(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> asin(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::asin(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> asind(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::asin(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> sinh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::sinh(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> asinh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::asinh(x); });

        return output;
    }

    /* Cosine */
    template<typename T>
    constexpr inline Tensor<T> cos(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::cos(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> cosd(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::cos(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> acos(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::acos(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> acosd(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::acos(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> cosh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::cosh(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> acosh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::acosh(x); });

        return output;
    }

    /* Tangent */
    template<typename T>
    constexpr inline Tensor<T> tan(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::tan(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> tand(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::tan(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> atan(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::atan(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> atand(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::atan(deg2rad(x)); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> atan2(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(lhs.shape() == rhs.shape() && "Error in atan2: Shapes are different");

        Tensor<T> output(lhs.shape());

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::atan2(lhs[i], rhs[i]);
        }

        return output;
    }

    template<typename T>
    constexpr inline T atan2d(T lhs, T rhs)
    {
        return std::atan2(deg2rad(lhs), deg2rad(rhs));
    }

    template<typename T>
    constexpr inline Tensor<T> atan2d(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(lhs.shape() == rhs.shape() && "Error in atan2: Shapes are different");

        Tensor<T> output(lhs.shape());

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = gt::atan2d(lhs, rhs);
        }

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> tanh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::tanh(x); });

        return output;
    }

    template<typename T>
    constexpr inline Tensor<T> atanh(const Tensor<T>& input)
    {
        Tensor<T> output(input.shape());

        std::transform(input.begin(), input.end(), output.begin(),
            [] (T x) { return std::atanh(x); });

        return output;
    }

    /* Cosecant */
    template<typename T>
    constexpr inline Tensor<T> csc(const Tensor<T>& input)
    {
        return 1.0f / gt::sin(input);
    }

    template<typename T>
    constexpr inline Tensor<T> cscd(const Tensor<T>& input)
    {
        return 1.0f / gt::sind(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acsc(const Tensor<T>& input)
    {
        return 1.0f / gt::asin(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acscd(const Tensor<T>& input)
    {
        return 1.0f / gt::asind(input);
    }

    template<typename T>
    constexpr inline Tensor<T> csch(const Tensor<T>& input)
    {
        return 1.0f / gt::sinh(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acsch(const Tensor<T>& input)
    {
        return 1.0f / gt::asinh(input);
    }

    /* Secant */
    template<typename T>
    constexpr inline Tensor<T> sec(const Tensor<T>& input)
    {
        return 1.0f / gt::cos(input);
    }

    template<typename T>
    constexpr inline Tensor<T> secd(const Tensor<T>& input)
    {
        return 1.0f / gt::cosd(input);
    }

    template<typename T>
    constexpr inline Tensor<T> asec(const Tensor<T>& input)
    {
        return 1.0f / gt::acos(input);
    }

    template<typename T>
    constexpr inline Tensor<T> asecd(const Tensor<T>& input)
    {
        return 1.0f / gt::acosd(input);
    }

    template<typename T>
    constexpr inline Tensor<T> sech(const Tensor<T>& input)
    {
        return 1.0f / gt::cosh(input);
    }

    template<typename T>
    constexpr inline Tensor<T> asech(const Tensor<T>& input)
    {
        return 1.0f / gt::acosh(input);
    }

    /* Cotangent */
    template<typename T>
    constexpr inline Tensor<T> cot(const Tensor<T>& input)
    {
        return 1.0f / gt::tan(input);
    }

    template<typename T>
    constexpr inline Tensor<T> cotd(const Tensor<T>& input)
    {
        return 1.0f / gt::tand(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acot(const Tensor<T>& input)
    {
        return 1.0f / gt::atan(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acotd(const Tensor<T>& input)
    {
        return 1.0f / gt::atand(input);
    }

    template<typename T>
    constexpr inline Tensor<T> coth(const Tensor<T>& input)
    {
        return 1.0f / gt::tanh(input);
    }

    template<typename T>
    constexpr inline Tensor<T> acoth(const Tensor<T>& input)
    {
        return 1.0f / gt::atanh(input);
    }

    template<typename T>
    constexpr inline Tensor<T> hypot(const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(lhs.shape() == rhs.shape() && "Error in hypot: Shapes are different");

        Tensor<T> output(lhs.shape());

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = std::hypot(lhs, rhs);
        }

        return output;
    }
}
