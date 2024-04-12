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
        inline std::vector<size_t> convolution_shape(const Tensor<T>& t1,
            const Tensor<T>& t2, CONVOLUTION type)
        {
            assert(t1.shape().size() == t2.shape().size() && "Error in convolution: Tensors have different ranks");
            std::vector<size_t> shape(t1.shape().size());

            for (size_t i = 0; i < shape.size(); i++) {
                switch (type) {
                    case FULL:
                    {
                        shape[i] = t1.shape(i) + t2.shape(i) - 1;
                        break;
                    }
                    case SAME:
                    {
                        shape[i] = t1.shape(i);
                        break;
                    }
                    case VALID:
                    {
                        assert(t1.shape(i) >= t2.shape(i) && "Error in convolution: result will be empty");
                        shape[i] = t1.shape(i) - t2.shape(i) + 1;
                        break;
                    }
                }
            }

            return shape;
        }

        template<typename T>
        inline Tensor<T> conv1(const Tensor<T>& t1, const Tensor<T>& t2, CONVOLUTION type)
        {
            Tensor<T> output(convolution_shape(t1, t2, type));
            //Tensor<T> output = gt::zeros(convolution_shape(t1, t2, type));

            size_t offset = 0;
            switch (type) {
                case FULL:
                {
                    offset = 0;
                    break;
                }
                case SAME:
                {
                    offset = t2.shape(0) / 2;
                    break;
                }
                case VALID:
                {
                    offset = t2.shape(0) - 1;
                    break;
                }
            }

            for (size_t i = offset; i < offset + output.shape(0); i++) {
                size_t limit = std::min(std::min(i + 1, t1.shape(0)),
                    t1.shape(0) + t2.shape(0) - 1 - i);
                size_t i1 = i >= t2.shape(0) - 1 ? i - (t2.shape(0) - 1) : 0;
                size_t i2 = std::min(i, t2.shape(0) - 1);
                for (size_t j = 0; j < limit; j++) {
                    output(i - offset) += t1(i1 + j) * t2(i2 - j);
                }
            }

            return output;
        }
    }
}
