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

        Tensor(const std::vector<size_t>& dims) : Dimensional(dims), data_ptr(this->_size) {}

        //TODO: Figure out the right way to do this with concepts
        //template<typename... Ts>// requires std::is_same_v<T1, size_t>
        //Tensor(Ts... dims) : Dimensional({static_cast<size_t>(dims)...}), data_ptr(this->_size) {}

        template<typename... Ts>
        const_reference operator () (Ts... dims) const {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->data_ptr[sub2ind(subs)];
        }

        template<typename... Ts>
        reference operator () (Ts... dims) {
            std::vector<size_t> subs = {static_cast<size_t>(dims)...};
            return this->data_ptr[sub2ind(subs)];
        }

        Tensor& operator = (const auto& input) {
            for (size_t i = 0; i < input.size(); i++) {
                this->data_ptr[i] = input[i];
            }
            return *this;
        }

        reference operator [] (size_t index) {
            return this->data_ptr[index];
        }

        const_reference operator [] (size_t index) const {
            return this->data_ptr[index];
        }

        //template<typename T1>
        //Tensor(Tensor<T1>&& input) : Dimensional(input.shape()) {
        //    this->data_ptr.resize(input.size());
        //    for (size_t i = 0; i < input.size(); i++) {
        //        this->data_ptr[i] = input[i];
        //    }
        //}

        Tensor& operator = (std::vector<T>&& input) {
            //TODO: Make this a true move assignment operator instead of copying
            std::copy(input.begin(), input.end(), this->data_ptr.begin());
            return *this;
        }

        friend std::ostream& operator << (std::ostream& output, const Tensor<T>& input) {
            size_t rest = input.size() / input.shape(0) / input.shape(1);

            for (size_t i = 0; i < rest; i++) {
                output << "ans(:,:," << i << ")" << std::endl;
                for (size_t j = 0; j < input.shape(0); j++) {
                    for (size_t k = 0; k < input.shape(1); k++) {
                        size_t index = i * input.stride(2) + k * input.stride(1) + j;
                        output << "\t" << std::fixed << std::setprecision(4) << input[index] << " ";
                    }
                    output << std::endl;
                }
                output << std::endl;
            }

            return output;
        }

        /* Iterators */
        std::vector<T>::iterator begin() {
            return this->data_ptr.begin();
        }

        std::vector<T>::iterator end() {
            return this->data_ptr.end();
        }

        std::vector<T>::const_iterator begin() const {
            return this->data_ptr.begin();
        }

        std::vector<T>::const_iterator end() const {
            return this->data_ptr.end();
        }

        pointer data() {
            return this->data_ptr.data();
        }

        const_pointer data() const {
            return this->data_ptr.data();
        }

        private:
        std::vector<T> data_ptr;

        size_t sub2ind(const std::vector<size_t>& subs) const {
            size_t index = 0;

            for (size_t i = 0; i < subs.size(); i++) {
                index += this->stride(i) * subs[i];
            }

            return index;
        }
    };
};
