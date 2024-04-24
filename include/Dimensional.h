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

#include <numeric>
#include <cassert>
#include <vector>

/* Calculates the stride for a Tensor arranged in column-major order */
std::vector<size_t> calculate_stride(const std::vector<size_t>& shape) {
    assert(shape.size() > 0 && "Error in calculate_stride: Size of shape must be greater than 0");

    std::vector<size_t> stride(shape.size());
    stride[0] = 1;

    size_t prod = shape[0];
    for (size_t i = 1; i < shape.size(); i++) {
        stride[i] = prod;
        prod *= shape[i];
    }

    return stride;
}

/* Calculates the total number of elements in a Tensor */
size_t calculate_size(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
}

/* Contains the shape, stride, and size information for Tensors or Expressions
 * Both will inherit from it */
class Dimensional {
    public:
        Dimensional(const std::vector<size_t>& shape) :
            m_shape(shape), 
            m_stride(calculate_stride(shape)),
            m_size(calculate_size(shape)) {}

        std::vector<size_t> shape() const {
            return this->m_shape;
        }

        size_t shape(size_t dim) const {
            if (dim < this->m_shape.size()) {
                return this->m_shape[dim];
            } else {
                return 1;
            }
        }

        std::vector<size_t> stride() const {
            return this->m_stride;
        }

        size_t stride(size_t dim) const {
            if (dim > this->m_stride.size()) {
                return this->m_size;
            } else {
                return this->m_stride[dim];
            }
        }

        size_t size() const {
            return this->m_size;
        }

    protected:
        std::vector<size_t> m_shape;
        std::vector<size_t> m_stride;
        size_t m_size;
};
