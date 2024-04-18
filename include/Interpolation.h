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

namespace gt
{
    /* produces N linearly spaced points between min and max */
    template<typename T> requires std::is_arithmetic_v<T>
    inline constexpr Tensor<T> linspace(T min, T max, size_t N)
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
    inline constexpr Tensor<T> logspace(T min, T max, size_t N)
    {
        assert((N > 0) && "Error in logspace: N must be greater than 0");
        return pow(10.0f, linspace(min, max, N));
    }

    template<typename T>
    inline constexpr std::tuple<Tensor<T>,Tensor<T>> meshgrid(const Tensor<T>& t1, const Tensor<T>& t2)
    {
        throw std::runtime_error{"meshgrid is not yet implemented."};
    }

    /* 1D linear interpolation; x must be sorted
     * extrapolates for values of xi outside of x */
    template<typename T>
    inline constexpr Tensor<T> interp1(const Tensor<T>& x, const Tensor<T>& y, const Tensor<T>& xi)
    {
        assert(x.shape().size() == 1 && "Error in interp1: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp1: y is not one-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp1: xi is not one-dimensional");
        assert(x.size() == y.size() && "Error in interp1: x and y are different sizes");
        assert(is_sorted(x) && "Error in interp1: x is not sorted");

        Tensor<T> yi(xi.shape());

        for (size_t i = 0; i < xi.size(); i++) {
            std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);
            size_t x1 = std::get<0>(xbounds);
            size_t x2 = std::get<0>(xbounds);

            T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);
            yi[i] = y(x1) * (1 - xd) + y(x2) * xd;
        }

        return yi;
    }

    /* 2D linear interpolation, x and y must be sorted
     * extrapolates for values of xi, yi outside of x, y */
    template<typename T>
    inline constexpr Tensor<T> interp2(const Tensor<T>& x, const Tensor<T>& y,
        const Tensor<T>& xy, const Tensor<T>& xi, const Tensor<T>& yi)
    {
        assert(x.shape().size() == 1 && "Error in interp2: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp2: y is not one-dimensional");
        assert(xy.shape().size() == 2 && "Error in interp2: xy is not two-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp2: xi is not one-dimensional");
        assert(yi.shape().size() == 1 && "Error in interp2: yi is not one-dimensional");
        assert(x.shape(0) == xy.shape(0) && "Error in interp2: size of x doesn't match zeroth dimension of xy");
        assert(y.shape(0) == xy.shape(1) && "Error in interp2: size of y doesn't match first dimension of xy");
        assert(is_sorted(x) && "Error in interp2: x is not sorted");
        assert(is_sorted(y) && "Error in interp2: y is not sorted");

        Tensor<T> xiyi({xi.size(), yi.size()});

        for (size_t i = 0; i < xi.size(); i++) {
            std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);

            size_t x1 = std::get<0>(xbounds);
            size_t x2 = std::get<1>(xbounds);

            T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);

            for (size_t j = 0; j < yi.size(); j++) {
                std::tuple<size_t,size_t> ybounds = interpolation_bounds(y, yi[j]);

                size_t y1 = std::get<0>(ybounds);
                size_t y2 = std::get<1>(ybounds);
                
                T yd = (yi[j] - y[y1]) / (y[y2] - y[y1]);

                T c00 = xy(x1,y1) * (1 - xd) + xy(x2,y1) * xd;
                T c01 = xy(x1,y2) * (1 - xd) + xy(x2,y2) * xd;
                xiyi(i,j) = c00 * (1 - yd) + c01 * yd;
            }
        }

        return xiyi;
    }

    /* 3D linear interpolation, x and y must be sorted
     * extrapolates for values of xi, yi outside of x, y */
    template<typename T>
    inline constexpr Tensor<T> interp3(const Tensor<T>& x, const Tensor<T>& y,
        const Tensor<T>& z, const Tensor<T>& xyz, const Tensor<T>& xi,
        const Tensor<T>& yi, const Tensor<T>& zi)
    {
        assert(x.shape().size() == 1 && "Error in interp3: x is not one-dimensional");
        assert(y.shape().size() == 1 && "Error in interp3: y is not one-dimensional");
        assert(z.shape().size() == 1 && "Error in interp3: z is not one-dimensional");
        assert(xyz.shape().size() == 3 && "Error in interp3: xyz is not three-dimensional");
        assert(xi.shape().size() == 1 && "Error in interp3: xi is not one-dimensional");
        assert(yi.shape().size() == 1 && "Error in interp3: yi is not one-dimensional");
        assert(zi.shape().size() == 1 && "Error in interp3: zi is not one-dimensional");
        assert(x.shape(0) == xyz.shape(0) && "Error in interp3: size of x doesn't match zeroth dimension of xyz");
        assert(y.shape(1) == xyz.shape(1) && "Error in interp3: size of y doesn't match first dimension of xyz");
        assert(y.shape(2) == xyz.shape(2) && "Error in interp3: size of y doesn't match second dimension of xyz");
        assert(is_sorted(x) && "Error in interp3: x is not sorted");
        assert(is_sorted(y) && "Error in interp3: y is not sorted");
        assert(is_sorted(z) && "Error in interp3: z is not sorted");

        Tensor<T> xiyizi({xi.size(), yi.size(), zi.size()});

        for (size_t i = 0; i < xi.size(); i++) {
            std::tuple<size_t,size_t> xbounds = interpolation_bounds(x, xi[i]);

            size_t x1 = std::get<0>(xbounds);
            size_t x2 = std::get<1>(xbounds);

            T xd = (xi[i] - x[x1]) / (x[x2] - x[x1]);

            for (size_t j = 0; j < yi.size(); j++) {
                std::tuple<size_t,size_t> ybounds = interpolation_bounds(y, yi[j]);

                size_t y1 = std::get<0>(ybounds);
                size_t y2 = std::get<1>(ybounds);
                
                T yd = (yi[j] - y[y1]) / (y[y2] - y[y1]);

                for (size_t k = 0; k < zi.size(); k++) {
                    std::tuple<size_t,size_t> zbounds = interpolation_bounds(z, zi[k]);

                    size_t z1 = std::get<0>(zbounds);
                    size_t z2 = std::get<1>(zbounds);
                    
                    T zd = (zi[k] - z[z1]) / (z[z2] - z[z1]);

                    T c000 = xyz(x1,y1,z1);
                    T c001 = xyz(x1,y1,z2);
                    T c010 = xyz(x1,y2,z1);
                    T c011 = xyz(x1,y2,z2);
                    T c100 = xyz(x2,y1,z1);
                    T c101 = xyz(x2,y1,z2);
                    T c110 = xyz(x2,y2,z1);
                    T c111 = xyz(x2,y2,z2);

                    T c00 = c000 * (1 - xd) + c100 * xd;
                    T c01 = c001 * (1 - xd) + c101 * xd;
                    T c10 = c010 * (1 - xd) + c110 * xd;
                    T c11 = c011 * (1 - xd) + c111 * xd;

                    T c0 = c00 * (1 - yd) + c10 * yd;
                    T c1 = c01 * (1 - yd) + c11 * yd;
                    
                    xiyizi(i,j,k) = c0 * (1 - zd) + c1 * zd;
                }
            }
        }

        return zi;
    }
}
