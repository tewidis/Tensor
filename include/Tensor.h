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

#include <algorithm>
#include <complex>
#include <iomanip>
#include <memory>

#include "Dimensional.h"

namespace gt {
    /* A multidimensional array object 
     * Wrapping a vector for now to get iterators for free, might manually manage the memory later */
    template<typename T>
    class Tensor : public Dimensional {
        public:
        typedef T& reference;
        typedef const T& const_reference;
        typedef T* pointer;
        typedef const T* const_pointer;

        class iterator {
            public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = T*;
            using reference         = T&;

            iterator(pointer ptr) : m_ptr(ptr) {}
            reference operator * () { return *m_ptr; }
            pointer operator -> () { return m_ptr; }
            reference operator [] (const difference_type& value) const { return m_ptr[value]; }
            iterator& operator ++ () { ++m_ptr; return *this; }
            iterator operator ++ (int) { iterator tmp = *this; ++(*this); return tmp; }
            iterator& operator -- () { --m_ptr; return *this; }
            iterator operator -- (int) { iterator tmp = *this; --(*this); return tmp; }
            iterator operator + (difference_type value) const { return iterator(m_ptr + value); }
            iterator operator - (difference_type value) const { return iterator(m_ptr - value); }
            iterator operator + (const iterator& other) const { return iterator(m_ptr + other); }
            difference_type operator - (const iterator& other) const { return m_ptr - other.m_ptr; }
            iterator& operator += (difference_type value) const { m_ptr += value; return *this; }
            iterator& operator -= (difference_type value) const { m_ptr -= value; return *this; }
            bool operator == (const iterator& other) { return this->m_ptr == other.m_ptr; }
            bool operator != (const iterator& other) { return this->m_ptr != other.m_ptr; }
            bool operator > (const iterator& other) { return this->m_ptr > other.m_ptr; }
            bool operator < (const iterator& other) { return this->m_ptr < other.m_ptr; }
            bool operator >= (const iterator& other) { return this->m_ptr >= other.m_ptr; }
            bool operator <= (const iterator& other) { return this->m_ptr <= other.m_ptr; }

            private:
            pointer m_ptr;
        };

        class const_iterator {
            public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = T;
            using pointer           = const T*;
            using reference         = const T&;

            const_iterator(pointer ptr) : m_ptr(ptr) {}
            reference operator * () const { return *m_ptr; }
            pointer operator -> () const { return m_ptr; }
            reference operator [] (const difference_type& value) const { return m_ptr[value]; }
            const_iterator& operator ++ () { ++m_ptr; return *this; }
            const_iterator operator ++ (int) { const_iterator tmp = *this; ++(*this); return tmp; }
            const_iterator& operator -- () { --m_ptr; return *this; }
            const_iterator operator -- (int) { const_iterator tmp = *this; --(*this); return tmp; }
            const_iterator operator + (difference_type value) const { return const_iterator(m_ptr + value); }
            const_iterator operator - (difference_type value) const { return const_iterator(m_ptr - value); }
            const_iterator operator + (const const_iterator& other) const { return const_iterator(m_ptr + other); }
            difference_type operator - (const const_iterator& other) const { return m_ptr - other.m_ptr; }
            const_iterator& operator += (difference_type value) const { m_ptr += value; return *this; }
            const_iterator& operator -= (difference_type value) const { m_ptr -= value; return *this; }
            bool operator == (const const_iterator& other) { return this->m_ptr == other.m_ptr; }
            bool operator != (const const_iterator& other) { return this->m_ptr != other.m_ptr; }
            bool operator > (const const_iterator& other) { return this->m_ptr > other.m_ptr; }
            bool operator < (const const_iterator& other) { return this->m_ptr < other.m_ptr; }
            bool operator >= (const const_iterator& other) { return this->m_ptr >= other.m_ptr; }
            bool operator <= (const const_iterator& other) { return this->m_ptr <= other.m_ptr; }

            private:
            pointer m_ptr;
        };

        /* Constructor */
        Tensor(const std::vector<size_t>& dims)
            : Dimensional(dims)
            , m_data(std::shared_ptr<T>(new T[this->m_size]))
        {
            std::fill(this->begin(), this->end(), static_cast<T>(0));
        }

        /* Destructor */
        ~Tensor() {}

        /* Copy assignment operator */
        Tensor& operator = (const Tensor& other)
        {
            if (this != &other) {
                this->m_data = other.m_data;
                this->m_shape = other.shape();
                this->m_size = other.size();
                this->m_stride = other.stride();
                std::copy(other.begin(), other.end(), this->begin());
            }
            return *this;
        }

        /* Move assignment operator */
        Tensor& operator = (Tensor<T>&& other) noexcept
        {
            if (this != &other) {
                this->m_data = other.m_data;
                this->m_shape = other.m_shape;
                this->m_size = other.m_size;
                this->m_stride = other.m_stride;

                other.m_size = 0;
            }
            return *this;
        }

        Tensor& operator = (std::vector<T>&& other) noexcept
        {
            assert(this->size() == other.size() && "Error in move assignment operator: Size mismatch");
            std::copy(other.begin(), other.end(), this->begin());
            return *this;
        }

        /* Copy constructor */
        Tensor(const Tensor<T>& other)
            : Dimensional(other.shape())
            , m_data(other.m_data)
        {}

        /* Move constructor */
        Tensor(Tensor<T>&& other) noexcept
            : Dimensional(other.shape())
            , m_data(other.m_data)
        {
            other.m_size = 0;
            other.m_shape = {};
            other.m_stride = {};
        }

        template<typename... Ts>
        const_reference operator () (Ts... dims) const {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->m_data.get()[sub2ind(*this, subs)];
        }

        template<typename... Ts>
        reference operator () (Ts... dims) {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->m_data.get()[sub2ind(*this, subs)];
        }

        Tensor<T> operator () (const Tensor<bool>& indices)
        {
            assert(indices.size() == this->size() && "Error in logical indexing");

            size_t count = std::count_if(indices.begin(), indices.end(), [] (bool b) { return b; });
            Tensor<T> output({count});
            size_t index = 0;
            for (size_t i = 0; i < indices.size(); i++) {
                if (indices[i]) {
                    output[index] = this->m_data.get()[i];
                    index += 1;
                }
            }
            
            return output;
        }

        reference operator [] (size_t index) {
            return this->m_data.get()[index];
        }

        const_reference operator [] (size_t index) const {
            return this->m_data.get()[index];
        }

        friend std::ostream& operator << (std::ostream& output, const Tensor<T>& input) {
            size_t nloops = input.size() / input.shape(0) / input.shape(1);

            output << std::fixed << std::setprecision(8);
            for (size_t i = 0; i < nloops; i++) {
                output << "ans(:,:," << i << ")" << std::endl;

                for (size_t j = 0; j < input.shape(1); j++) {
                    for (size_t k = 0; k < input.shape(0); k++) {
                        size_t index = i * input.stride(2) + j * input.stride(1) + k * input.stride(0);
                        output << "\t" << std::setw(8) << input[index] << " ";
                    }
                    output << std::endl;
                }
                output << std::endl;
            }

            return output;
        }

        /* Iterators */
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        iterator begin() { return iterator(&m_data.get()[0]); }
        iterator end() { return iterator(&m_data.get()[this->size()]); }
        const_iterator begin() const { return const_iterator(&m_data.get()[0]); }
        const_iterator end() const { return const_iterator(&m_data.get()[this->size()]); }

        reverse_iterator rbegin() { return reverse_iterator(&m_data.get()[this->size()]); }
        reverse_iterator rend() { return reverse_iterator(&m_data.get()[0]); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(&m_data.get()[this->size()]); }
        const_reverse_iterator rend() const { return const_reverse_iterator(&m_data.get()[0]); }

        const_iterator cbegin() const { return const_iterator(&m_data.get()[0]); }
        const_iterator cend() const { return const_iterator(&m_data.get()[this->size()]); }

        const_reverse_iterator crbegin() const { return const_reverse_iterator(&m_data.get()[this->size()]); }
        const_reverse_iterator crend() const { return const_reverse_iterator(&m_data.get()[0]); }

        /* data accessors */
        pointer data() { return this->m_data.get(); }
        const_pointer data() const { return this->m_data.get(); }
        reference front() { return this->m_data.get()[0]; }
        const_reference front() const { return this->m_data.get()[0]; }
        reference back() { return this->m_data.get()[this->size()-1]; }
        const_reference back() const { return this->m_data.get()[this->size()-1]; }

        private:
        std::shared_ptr<T> m_data;
    };

    template<typename T>
    struct is_complex_t : public std::false_type {};

    template<typename T>
    struct is_complex_t<std::complex<T>> : public std::true_type {};

    template<typename T>
    concept TensorType = requires(T param)
    {
        requires std::is_integral_v<T> || std::is_floating_point_v<T>
            || is_complex_t<T>::value || std::is_same_v<bool, T>;
        requires !std::is_pointer_v<T>;
    };

    template<typename T1, typename T2>
    bool compare_shape(const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        for (size_t i = 0; i < std::max(ndims(lhs), ndims(rhs)); i++) {
            if (lhs.shape(i) != rhs.shape(i)) {
                return false;
            }
        }
        return true;
    }

    /* addition operators */
    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() + T2())> operator + (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in + operator: Tensors are different shapes");

        Tensor<decltype(T1() + T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), std::plus<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() + T2())> operator + (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<decltype(T1() + T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value + rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() + T2())> operator + (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs + lhs;
    }

    /* subtraction operators */
    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() - T2())> operator - (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in - operator: Tensors are different shapes");

        Tensor<decltype(T1() - T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(),
            [] (T1 lhs, T2 rhs) { return lhs - rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() - T2())> operator - (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<decltype(T1() - T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value - rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() - T2())> operator - (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<decltype(T1() - T2())> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs - value; });
        return output;
    }

    /* multiplication operators */
    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() * T2())> operator * (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in * operator: Tensors are different shapes");

        Tensor<decltype(T1() * T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(),
            [] (T1 lhs, T2 rhs) { return lhs * rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() * T2())> operator * (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<decltype(T1() * T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value * rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() * T2())> operator * (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs * lhs;
    }

    /* division operators */
    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() / T2())> operator / (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in / operator: Tensors are different shapes");

        Tensor<decltype(T1() / T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(),
            [] (T1 lhs, T2 rhs) { return lhs / rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() / T2())> operator / (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<decltype(T1() / T2())> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value / rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<decltype(T1() / T2())> operator / (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<decltype(T1() / T2())> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs / value; });
        return output;
    }

    /* equal operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator == (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in == operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::equal_to<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator == (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value == rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator == (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs == lhs;
    }

    /* not equal operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator != (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in != operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::not_equal_to<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator != (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value != rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator != (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs != lhs;
    }

    /* greater operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator > (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in > operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::greater<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator > (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value > rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator > (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs > value; });
        return output;
    }

    /* less operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator < (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in < operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::less<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator < (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value < rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator < (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs < value; });
        return output;
    }

    /* greater/equal operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator >= (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in >= operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::greater_equal<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator >= (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value >= rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator >= (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs >= value; });
        return output;
    }

    /* less/equal operators */
    template<TensorType T1, TensorType T2>
    Tensor<bool> operator <= (const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in <= operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::less_equal<>{});
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator <= (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value <= rhs; });
        return output;
    }

    template<TensorType T1, TensorType T2>
    Tensor<bool> operator <= (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs <= value; });
        return output;
    }
}
