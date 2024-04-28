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

#include <iomanip>

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
            iterator& operator ++ () { m_ptr++; return *this; }
            iterator operator ++ (int) { iterator tmp = *this; ++(*this); return tmp; }
            iterator& operator -- () { m_ptr--; return *this; }
            iterator operator -- (int) { iterator tmp = *this; --(*this); return tmp; }
            iterator operator + (const difference_type& value) { m_ptr += value; return *this; }
            iterator operator - (const difference_type& value) { m_ptr -= value; return *this; }
            friend bool operator == (const iterator& a, const iterator& b) { return a.m_ptr == b.m_ptr; }
            friend bool operator != (const iterator& a, const iterator& b) { return a.m_ptr != b.m_ptr; }
            friend bool operator > (const iterator& a, const iterator& b) { return a.m_ptr > b.m_ptr; }
            friend bool operator < (const iterator& a, const iterator& b) { return a.m_ptr < b.m_ptr; }
            friend bool operator >= (const iterator& a, const iterator& b) { return a.m_ptr >= b.m_ptr; }
            friend bool operator <= (const iterator& a, const iterator& b) { return a.m_ptr <= b.m_ptr; }
            friend bool operator - (const iterator& a, const iterator& b) { return a.m_ptr - b.m_ptr; }

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
            const_iterator& operator ++ () { m_ptr++; return *this; }
            const_iterator operator ++ (int) { const_iterator tmp = *this; ++(*this); return tmp; }
            const_iterator& operator -- () { m_ptr--; return *this; }
            const_iterator operator -- (int) { const_iterator tmp = *this; --(*this); return tmp; }
            const_iterator operator + (const difference_type& value) { m_ptr += value; return *this; }
            const_iterator operator - (const difference_type& value) { m_ptr -= value; return *this; }
            friend bool operator == (const const_iterator& a, const const_iterator& b) { return a.m_ptr == b.m_ptr; }
            friend bool operator != (const const_iterator& a, const const_iterator& b) { return a.m_ptr != b.m_ptr; }
            friend bool operator > (const const_iterator& a, const const_iterator& b) { return a.m_ptr > b.m_ptr; }
            friend bool operator < (const const_iterator& a, const const_iterator& b) { return a.m_ptr < b.m_ptr; }
            friend bool operator >= (const const_iterator& a, const const_iterator& b) { return a.m_ptr >= b.m_ptr; }
            friend bool operator <= (const const_iterator& a, const const_iterator& b) { return a.m_ptr <= b.m_ptr; }
            friend bool operator - (const const_iterator& a, const const_iterator& b) { return a.m_ptr - b.m_ptr; }

            private:
            pointer m_ptr;
        };

        /* Constructor */
        Tensor(const std::vector<size_t>& dims)
            : Dimensional(dims)
            , m_data(new T[this->m_size])
        {
            std::fill(this->begin(), this->end(), static_cast<T>(0));
        }

        /* Destructor */
        ~Tensor()
        {
            delete[] this->m_data;
        }

        /* Copy assignment operator */
        Tensor& operator = (const Tensor& other)
        {
            if (this != &other) {
                delete[] this->m_data;
                this->m_data = new T[other.size()];
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
                delete[] this->m_data;
                this->m_data = other.m_data;
                this->m_shape = other.m_shape;
                this->m_size = other.m_size;
                this->m_stride = other.m_stride;

                other.m_size = 0;
                other.m_data = nullptr;
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
            , m_data(new T[this->m_size])
        {
            std::copy(other.begin(), other.end(), this->begin());
        }

        /* Move constructor */
        Tensor(Tensor<T>&& other) noexcept
            : Dimensional(other.shape())
            , m_data(other.m_data)
        {
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_shape = {};
            other.m_stride = {};
        }

        template<typename... Ts>
        const_reference operator () (Ts... dims) const {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->m_data[sub2ind(*this, subs)];
        }

        template<typename... Ts>
        reference operator () (Ts... dims) {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->m_data[sub2ind(*this, subs)];
        }

        reference operator [] (size_t index) {
            return this->m_data[index];
        }

        const_reference operator [] (size_t index) const {
            return this->m_data[index];
        }

        friend std::ostream& operator << (std::ostream& output, const Tensor<T>& input) {
            size_t rest = input.size() / input.shape(0) / input.shape(1);

            output << std::fixed << std::setprecision(8);
            for (size_t i = 0; i < rest; i++) {
                output << "ans(:,:," << i << ")" << std::endl;
                for (size_t j = 0; j < input.shape(0); j++) {
                    for (size_t k = 0; k < input.shape(1); k++) {
                        size_t index = i * input.stride(2) + k * input.stride(1) + j;
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

        iterator begin() { return iterator(&m_data[0]); }
        iterator end() { return iterator(&m_data[this->size()]); }
        const_iterator begin() const { return const_iterator(&m_data[0]); }
        const_iterator end() const { return const_iterator(&m_data[this->size()]); }

        reverse_iterator rbegin() { return iterator(&m_data[this->size()]); }
        reverse_iterator rend() { return iterator(&m_data[0]); }
        const_reverse_iterator rbegin() const { return const_iterator(&m_data[this->size()]); }
        const_reverse_iterator rend() const { return const_iterator(&m_data[0]); }

        const_iterator cbegin() const { return const_iterator(&m_data[0]); }
        const_iterator cend() const { return const_iterator(&m_data[this->size()]); }

        const_reverse_iterator crbegin() const { return const_iterator(&m_data[this->size()]); }
        const_reverse_iterator crend() const { return const_iterator(&m_data[0]); }

        /* data accessors */
        pointer data() { return this->m_data; }
        const_pointer data() const { return this->m_data; }
        reference front() { return this->m_data[0]; }
        const_reference front() const { return this->m_data[0]; }
        reference back() { return this->m_data[this->size()-1]; }
        const_reference back() const { return this->m_data[this->size()-1]; }

        private:
        T* m_data;
    };

    template<typename T1, typename T2>
    bool compare_shape(const Tensor<T1>& lhs, const Tensor<T2>& rhs)
    {
        for (size_t i = 0; i < std::max(lhs.shape().size(), rhs.shape().size()); i++) {
            if (lhs.shape(i) != rhs.shape(i)) {
                return false;
            }
        }
        return true;
    }

    /* addition operators */
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> operator + (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in + operator: Tensors are different shapes");

        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(), std::plus<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> operator + (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value + rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T2> operator + (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs + lhs;
    }

    /* subtraction operators */
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> operator - (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in - operator: Tensors are different shapes");

        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), output.begin(),
            [] (T lhs, T rhs) { return lhs - rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> operator - (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value - rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T2> operator - (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<T2> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs - value; });
        return output;
    }

    /* multiplication operators */
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> operator * (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in * operator: Tensors are different shapes");

        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T value) { return value * rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> operator * (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value * rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T2> operator * (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs * lhs;
    }

    /* division operators */
    template<typename T> requires std::is_arithmetic_v<T>
    Tensor<T> operator / (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in / operator: Tensors are different shapes");

        Tensor<T> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T value) { return value / rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T1> operator / (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<T1> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value / rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<T2> operator / (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<T2> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs / value; });
        return output;
    }

    /* equal operators */
    template<typename T>
    Tensor<bool> operator == (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in == operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::equal_to<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator == (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value == rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator == (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs == lhs;
    }

    /* not equal operators */
    template<typename T>
    Tensor<bool> operator != (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in != operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::not_equal_to<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator != (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value != rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator != (T1 lhs, const Tensor<T2>& rhs)
    {
        return rhs != lhs;
    }

    /* greater operators */
    template<typename T>
    Tensor<bool> operator > (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in > operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::greater<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator > (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value > rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator > (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs > value; });
        return output;
    }

    /* less operators */
    template<typename T>
    Tensor<bool> operator < (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in < operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::less<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator < (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value < rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator < (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs < value; });
        return output;
    }

    /* greater/equal operators */
    template<typename T>
    Tensor<bool> operator >= (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in >= operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::greater_equal<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator >= (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value >= rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator >= (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs >= value; });
        return output;
    }

    /* less/equal operators */
    template<typename T>
    Tensor<bool> operator <= (const Tensor<T>& lhs, const Tensor<T>& rhs)
    {
        assert(compare_shape(lhs, rhs) && "Error in <= operator: Tensors are different shapes");

        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(),
            output.begin(), std::less_equal<>{});
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator <= (const Tensor<T1>& lhs, T2 rhs)
    {
        Tensor<bool> output(lhs.shape());
        std::transform(lhs.begin(), lhs.end(), output.begin(),
            [rhs] (T1 value) { return value <= rhs; });
        return output;
    }

    template<typename T1, typename T2>
        requires std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
    Tensor<bool> operator <= (T1 lhs, const Tensor<T2>& rhs)
    {
        Tensor<bool> output(rhs.shape());
        std::transform(rhs.begin(), rhs.end(), output.begin(),
            [lhs] (T2 value) { return lhs <= value; });
        return output;
    }
}
