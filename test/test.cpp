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

#include <cassert>
#include <iostream>

#include "Interpolation.h"
#include "LinearAlgebra.h"
#include "Statistics.h"
#include "Tensor.h"
#include "TensorOperations.h"
#include "Trigonometry.h"
#include "SignalProcessing.h"

/* Tests that the () operator overload is indexing correctly */
void access_test()
{
    gt::Tensor<size_t> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    size_t compare = 0;
    for (size_t k = 0; k < input.shape(2); k++) {
        for (size_t j = 0; j < input.shape(1); j++) {
            for (size_t i = 0; i < input.shape(0); i++) {
                assert(input(i,j,k) == compare);
                compare++;
            }
        }
    }
}

void mod_test()
{
    gt::Tensor<float> actual = gt::mod(gt::linspace(-1.0f, 1.0f, 21), 2.0f);

    gt::Tensor<float> correct({21});
    correct = {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
        1.9f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(actual - correct) > 1e-4f));
}

void rem_test()
{
    gt::Tensor<float> actual = gt::rem(gt::linspace(-1.0f, 1.0f, 21), 2.0f);

    gt::Tensor<float> correct({21});
    correct = {-1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f,
        -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(actual - correct) > 1e-4f));
}

void linspace_test()
{
    gt::Tensor<float> correct({21});
    correct = {-1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f,
        -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(gt::linspace(-1.0f, 1.0f, 21) - correct) > 1e-4f));
}

void logspace_test()
{
    gt::Tensor<float> correct({11});
    correct = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000};
    assert(gt::all(gt::logspace(-5.0f, 5.0f, 11) == correct));
}

void sum_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 5, 9,
               13, 17, 21,
               25, 29, 33,
               37, 41, 45};
    assert(gt::all(gt::sum(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {6, 9,
                24, 27,
                42, 45,
                60, 63};
    assert(gt::all(gt::sum(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {36, 40,
                44, 48,
                52, 56};
    assert(gt::all(gt::sum(input, 2) == correct2));
}

void cumsum_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 5, 4, 9, 6, 13, 8, 17, 10, 21, 12, 25, 14, 29, 16, 33, 18, 37, 20, 41, 22, 45};
    assert(gt::all(gt::cumsum(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 4, 6, 9, 6, 7, 14, 16, 24, 27, 12, 13, 26, 28, 42, 45, 18, 19, 38, 40, 60, 63};
    assert(gt::all(gt::cumsum(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 40, 44, 48, 52, 56};
    assert(gt::all(gt::cumsum(input, 2) == correct2));
}

void movsum_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 5, 4, 9, 6, 13, 8, 17, 10, 21, 12, 25, 14, 29, 16, 33, 18, 37, 20, 41, 22, 45};
    assert(gt::all(gt::movsum(input, 2, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 4, 6, 8, 6, 7, 14, 16, 18, 20, 12, 13, 26, 28, 30, 32, 18, 19, 38, 40, 42, 44};
    assert(gt::all(gt::movsum(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40};
    assert(gt::all(gt::movsum(input, 2, 2) == correct2));
}

void diff_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1};
    assert(gt::all(gt::diff(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 2, 4});
    correct1 = {2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2};
    assert(gt::all(gt::diff(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 3});
    correct2 = {6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6};
    assert(gt::all(gt::diff(input, 2) == correct2));
}

void prod_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0, 6, 20, 42, 72, 110, 156, 210, 272, 342, 420, 506};
    assert(gt::all(gt::prod(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {0, 15, 480, 693, 2688, 3315, 7920, 9177};
    assert(gt::all(gt::prod(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {0, 1729, 4480, 8505, 14080, 21505};
    assert(gt::all(gt::prod(input, 2) == correct2));
}

void cumprod_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 6, 4, 20, 
                6, 42, 8, 72, 10, 110, 
                12, 156, 14, 210, 16, 272, 
                18, 342, 20, 420, 22, 506};
    assert(gt::all(gt::cumprod(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 3, 0, 15,
                6, 7, 48, 63, 480, 693,
                12, 13, 168, 195, 2688, 3315,
                18, 19, 360, 399, 7920, 9177};
    assert(gt::all(gt::cumprod(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5,
                0, 7, 16, 27, 40, 55,
                0, 91, 224, 405, 640, 935,
                0, 1729, 4480, 8505, 14080, 21505};
    assert(gt::all(gt::cumprod(input, 2) == correct2));
}

void movprod_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 6, 4, 20, 
                6, 42, 8, 72, 10, 110, 
                12, 156, 14, 210, 16, 272, 
                18, 342, 20, 420, 22, 506};
    assert(gt::all(gt::movprod(input, 2, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 3, 8, 15,
                6, 7, 48, 63, 80, 99,
                12, 13, 168, 195, 224, 255,
                18, 19, 360, 399, 440, 483};
    assert(gt::all(gt::movprod(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5,
                0, 7, 16, 27, 40, 55,
                72, 91, 112, 135, 160, 187,
                216, 247, 280, 315, 352, 391};
    assert(gt::all(gt::movprod(input, 2, 2) == correct2));
}

void trapz_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(gt::trapz(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {4, 6, 16, 18, 28, 30, 40, 42};
    assert(gt::all(gt::trapz(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {27, 30, 33, 36, 39, 42};
    assert(gt::all(gt::trapz(input, 2) == correct2));
}

void cumtrapz_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0.5, 0, 2.5, 0, 4.5, 0, 6.5, 0, 8.5, 0, 10.5, 0, 12.5, 0, 14.5, 0, 16.5, 0, 18.5, 0, 20.5, 0, 22.5};
    assert(gt::all(gt::cumtrapz(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 0, 1, 2, 4, 6, 0, 0, 7, 8, 16, 18, 0, 0, 13, 14, 28, 30, 0, 0, 19, 20, 40, 42};
    assert(gt::all(gt::cumtrapz(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36, 39, 42};
    assert(gt::all(gt::cumtrapz(input, 2) == correct2));
}

void max_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    assert(gt::all(gt::max(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {4, 5, 10, 11, 16, 17, 22, 23};
    assert(gt::all(gt::max(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::max(input, 2) == correct2));
}

void cummax_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(input, 2) == correct2));
}

void movmax_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::movmax(input, 2, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::movmax(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::movmax(input, 2, 2) == correct2));
}

void min_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    assert(gt::all(gt::min(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {0, 1, 6, 7, 12, 13, 18, 19};
    assert(gt::all(gt::min(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::min(input, 2) == correct2));
}

void cummin_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22};
    assert(gt::all(gt::cummin(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 12, 13, 12, 13, 12, 13, 18, 19, 18, 19, 18, 19};
    assert(gt::all(gt::cummin(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::cummin(input, 2) == correct2));
}

void movmin_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22};
    assert(gt::all(gt::movmin(input, 2, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 1, 2, 3, 6, 7, 6, 7, 8, 9, 12, 13, 12, 13, 14, 15, 18, 19, 18, 19, 20, 21};
    assert(gt::all(gt::movmin(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    assert(gt::all(gt::movmin(input, 2, 2) == correct2));
}

void mean_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(gt::mean(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {2, 3, 8, 9, 14, 15, 20, 21};
    assert(gt::all(gt::mean(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {9, 10, 11, 12, 13, 14};
    assert(gt::all(gt::mean(input, 2) == correct2));
}

void movmean_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0.5, 2, 2.5, 4, 4.5, 6, 6.5, 8, 8.5, 10, 10.5, 12, 12.5, 14, 14.5, 16, 16.5, 18, 18.5, 20, 20.5, 22, 22.5};
    assert(gt::all(gt::movmean(input, 2, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 1, 2, 3, 4, 6, 7, 7, 8, 9, 10, 12, 13, 13, 14, 15, 16, 18, 19, 19, 20, 21, 22};
    assert(gt::all(gt::movmean(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    assert(gt::all(gt::movmean(input, 2, 2) == correct2));
}

void median_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(gt::median(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {2, 3, 8, 9, 14, 15, 20, 21};
    assert(gt::all(gt::median(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {9, 10, 11, 12, 13, 14};
    assert(gt::all(gt::median(input, 2) == correct2));
}

void var_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    assert(gt::all(gt::var(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {4, 4, 4, 4, 4, 4, 4, 4};
    assert(gt::all(gt::var(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {60, 60, 60, 60, 60, 60};
    assert(gt::all(gt::var(input, 2) == correct2));
}

void stddev_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.707107, 0.707107, 0.707107,
                0.707107, 0.707107, 0.707107,
                0.707107, 0.707107, 0.707107,
                0.707107, 0.707107, 0.707107};
    assert(!gt::any(gt::abs(gt::stddev(input, 0) - correct0) > 1e-4f));

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {2, 2, 2, 2, 2, 2, 2, 2};
    assert(!gt::any(gt::abs(gt::stddev(input, 1) - correct1) > 1e-4f));

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {7.7460, 7.7460, 7.7460,  7.7460,  7.7460,  7.7460};
    assert(!gt::any(gt::abs(gt::stddev(input, 2) - correct2) > 1e-4f));
}

void reshape_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct({1, 24});
    std::iota(correct.begin(), correct.end(), 0);

    assert(gt::all(gt::reshape(input, {1, 24}) == correct));
}

void squeeze_test()
{
    gt::Tensor<float> input({1, 2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    assert(gt::all(gt::squeeze(input) == input));
}

void flatten_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct({24});
    std::iota(correct.begin(), correct.end(), 0);

    assert(gt::all(gt::flatten(input) == correct));
}

void permute_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct012({2, 3, 4});
    correct012 = {0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11,
                  12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::permute(input, {0, 1, 2}) == correct012));

    gt::Tensor<float> correct021({2, 4, 3});
    correct021 = {0, 1, 6, 7, 12, 13, 18, 19,
                  2, 3, 8, 9, 14, 15, 20, 21,
                  4, 5, 10, 11, 16, 17, 22, 23};
    assert(gt::all(gt::permute(input, {0, 2, 1}) == correct021));

    gt::Tensor<float> correct102({3, 2, 4});
    correct102 = {0, 2, 4, 1, 3, 5,
                  6, 8, 10, 7, 9, 11, 
                  12, 14, 16, 13, 15, 17, 
                  18, 20, 22, 19, 21, 23};
    assert(gt::all(gt::permute(input, {1, 0, 2}) == correct102));

    gt::Tensor<float> correct120({3, 4, 2});
    correct120 = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    assert(gt::all(gt::permute(input, {1, 2, 0}) == correct120));

    gt::Tensor<float> correct201({4, 2, 3});
    correct201 = {0, 6, 12, 18, 1, 7, 13, 19,
                  2, 8, 14, 20, 3, 9, 15, 21,
                  4, 10, 16, 22, 5, 11, 17, 23};
    assert(gt::all(gt::permute(input, {2, 0, 1}) == correct201));

    gt::Tensor<float> correct210({4, 3, 2});
    correct210 = {0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22,
                  1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23};
    assert(gt::all(gt::permute(input, {2, 1, 0}) == correct210));
}

void ipermute_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    assert(gt::all(gt::ipermute(gt::permute(input, {0, 1, 2}), {0, 1, 2}) == input));
    assert(gt::all(gt::ipermute(gt::permute(input, {0, 2, 1}), {0, 2, 1}) == input));
    assert(gt::all(gt::ipermute(gt::permute(input, {1, 0, 2}), {1, 0, 2}) == input));
    assert(gt::all(gt::ipermute(gt::permute(input, {1, 2, 0}), {1, 2, 0}) == input));
    assert(gt::all(gt::ipermute(gt::permute(input, {2, 0, 1}), {2, 0, 1}) == input));
    assert(gt::all(gt::ipermute(gt::permute(input, {2, 1, 0}), {2, 1, 0}) == input));
}

void repmat_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct({4, 6, 8});
    correct = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 
               6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 
               6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 
               12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 
               12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 
               18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23, 
               18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23,
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 
               0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 
               6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 
               6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 
               12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 
               12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 
               18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23, 
               18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23};
    assert(gt::all(gt::repmat(input, {2, 2, 2})  == correct));
}

void circshift_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {1, 0, 3, 2, 5, 4,
                7, 6, 9, 8, 11, 10,
                13, 12, 15, 14, 17, 16,
                19, 18, 21, 20, 23, 22};
    assert(gt::all(gt::circshift(input, 1, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {2, 3, 4, 5, 0, 1,
                8, 9, 10, 11, 6, 7,
                14, 15, 16, 17, 12, 13,
                20, 21, 22, 23, 18, 19};
    assert(gt::all(gt::circshift(input, 2, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::circshift(input, 3, 2) == correct2));
}

void flip_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {1, 0, 3, 2, 5, 4,
                7, 6, 9, 8, 11, 10,
                13, 12, 15, 14, 17, 16,
                19, 18, 21, 20, 23, 22};
    assert(gt::all(gt::flip(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {4, 5, 2, 3, 0, 1,
                10, 11, 8, 9, 6, 7,
                16, 17, 14, 15, 12, 13,
                22, 23, 20, 21, 18, 19};
    assert(gt::all(gt::flip(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {18, 19, 20, 21, 22, 23,
                12, 13, 14, 15, 16, 17,
                6, 7, 8, 9, 10, 11,
                0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::flip(input, 2) == correct2));
}

void cart2sph_test()
{
    gt::Tensor<float> input({8, 3});
    input = {0, 0, 0, 0, 1, 1, 1, 1,
             0, 0, 1, 1, 0, 0, 1, 1,
             0, 1, 0, 1, 0, 1, 0, 1};

    gt::Tensor<float> correct({8, 3});
    correct = {0, 0, 1.5708, 1.5708, 0, 0, 0.7854, 0.7854,
               0, 1.5708, 0, 0.7854, 0, 0.7854, 0, 0.6155,
               0, 1, 1, 1.4142, 1, 1.4142, 1.4142, 1.7321};
    assert(!gt::any(gt::abs(gt::cart2sph(input) - correct) > 1e-4f));
}

void sph2cart_test()
{
    gt::Tensor<float> input({8, 3});
    input = {0, 0, 1.5708, 1.5708, 0, 0, 0.7854, 0.7854,
             0, 1.5708, 0, 0.7854, 0, 0.7854, 0, 0.6155,
             0, 1, 1, 1.4142, 1, 1.4142, 1.4142, 1.7321};

    gt::Tensor<float> correct({8, 3});
    correct = {0, 0, 0, 0, 1, 1, 1, 1,
               0, 0, 1, 1, 0, 0, 1, 1,
               0, 1, 0, 1, 0, 1, 0, 1};
    assert(!gt::any(gt::abs(gt::sph2cart(input) - correct) > 1e-4f));
}

void cart2pol_test()
{
    gt::Tensor<float> input({8, 3});
    input = {0, 0, 0, 0, 1, 1, 1, 1,
             0, 0, 1, 1, 0, 0, 1, 1,
             0, 1, 0, 1, 0, 1, 0, 1};

    gt::Tensor<float> correct({8, 3});
    correct = {0, 0, 1.5708, 1.5708, 0, 0, 0.7854, 0.7854,
               0, 0, 1, 1, 1, 1, 1.4142, 1.4142,
               0, 1, 0, 1, 0, 1, 0, 1};
    assert(!gt::any(gt::abs(gt::cart2pol(input) - correct) > 1e-4f));
}

void pol2cart_test()
{
    gt::Tensor<float> input({8, 3});
    input = {0, 0, 1.5708, 1.5708, 0, 0, 0.7854, 0.7854,
             0, 0, 1, 1, 1, 1, 1.4142, 1.4142,
             0, 1, 0, 1, 0, 1, 0, 1};

    gt::Tensor<float> correct({8, 3});
    correct = {0, 0, 0, 0, 1, 1, 1, 1,
               0, 0, 1, 1, 0, 0, 1, 1,
               0, 1, 0, 1, 0, 1, 0, 1};
    assert(!gt::any(gt::abs(gt::pol2cart(input) - correct) > 1e-4f));
}

void cat_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({4, 3, 4});
    correct0 = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5,
                6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11,
                12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17,
                18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23};
    assert(gt::all(gt::cat(0, input, input) == correct0));

    gt::Tensor<float> correct1({2, 6, 4});
    correct1 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cat(1, input, input) == correct1));

    gt::Tensor<float> correct2({2, 3, 8});
    correct2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cat(2, input, input) == correct2));
}

void unique_test()
{
    gt::Tensor<float> input({3, 4});
    input = {2, 3, 1, 5, 4, 3, 1, 8, 5, 4, 7, 6};

    gt::Tensor<float> correct({8});
    correct = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(gt::all(gt::unique(input) == correct));
}

void matmul_test()
{
    gt::Tensor<float> mat1({2, 3});
    std::iota(mat1.begin(), mat1.end(), 0);
    gt::Tensor<float> mat2({3, 2});
    std::iota(mat2.begin(), mat2.end(), 0);
    gt::Tensor<float> vec1({3, 1});
    std::iota(vec1.begin(), vec1.end(), 0);

    gt::Tensor<float> correct({2, 2});
    correct = {10, 13, 28, 40};
    assert(gt::all(gt::linalg::matmul(mat1, mat2) == correct));

    gt::Tensor<float> correct_vec({2, 1});
    correct_vec = {10, 13};
    assert(gt::all(gt::linalg::matmul(mat1, vec1) == correct_vec));
}

void interp1_test()
{
    gt::Tensor<float> x({10});
    x = {-5.6, -4.5, -3.2, -2.7, -1.1, 0.0, 1.3, 2.8, 3.1, 4.4};
    gt::Tensor<float> y({10});
    y = {2.3, 5.4, -4.5, -7.6, 5.1, 2.8, 7.2, 1.9, 2.0, 6.6};
    gt::Tensor<float> xi({10});
    xi = {-6.0, -4.5, -4.4, -3.3, -3.0, -0.5, 0.5, 2.0, 2.9, 5.0};

    gt::Tensor<float> increasing_correct({10});
    increasing_correct = {1.1727, 5.4000, 4.6385, -3.7385, -5.7400, 3.8455, 4.4923, 4.7267, 1.9333, 8.7231};
    assert(!gt::any(gt::abs(gt::interp1(x, y, xi) - increasing_correct) > 1e-4f));

    gt::Tensor<float> decreasing_correct({10});
    decreasing_correct = {8.2727, 2.0000, 1.9923, 1.9077, 4.0200, 4.0545, 0.2154, -6.1533, -1.2000, 0.8692};
    assert(!gt::any(gt::abs(gt::interp1(gt::flip(x), y, xi) - decreasing_correct) > 1e-4f));
}

void interp2_test()
{
    gt::Tensor<float> x({5});
    x = {-2.7, -1.1, 0.0, 1.3, 2.8};
    gt::Tensor<float> y({6});
    y = {-1.3, 0.2, 1.8, 2.1, 3.7, 4.3};
    gt::Tensor<float> z({5, 6});
    z = {3.2, -3.4, 4.5, 1.8, 9.3,
         5.1, -6.5, -2.8, -4.7, 0.2,
         1.9, 2.0, 3.3, 4.2, -6.7,
         5.0, -2.4, -1.7, 7.2, 8.1,
         7.6, 8.2, 9.3, 0.6, 4.5,
         3.2, -6.4, 4.7, 2.2, 3.9};
    gt::Tensor<float> xi({4});
    xi = {-1.5, -0.5, 0.5, 1.5};
    gt::Tensor<float> yi({3});
    yi = {-1.5, -0.5, 0.5};

    gt::Tensor<float> correct({4, 3});
    correct = {-1.5033, 1.6279, 4.3938, 3.7129,
               -2.7367, -1.9661, -0.2677, -0.8516,
               -2.5547, -3.1335, -2.1851, -2.7729};
    assert(!gt::any(gt::abs(gt::interp2(x, y, z, xi, yi) - correct) > 1e-4f));
}

void conv1_test()
{
    gt::Tensor<float> t1({5});
    t1 = {1, 5, 6, 7, 8};
    gt::Tensor<float> t2({3});
    t2 = {2, 3, 4};

    gt::Tensor<float> correct_full({7});
    correct_full = {2, 13, 31, 52, 61, 52, 32};

    gt::Tensor<float> correct_same({5});
    correct_same = {13, 31, 52, 61, 52};

    gt::Tensor<float> correct_valid({3});
    correct_valid = {31, 52, 61};

    assert(gt::all(gt::sp::conv1(t1, t2, gt::FULL) == correct_full));
    assert(gt::all(gt::sp::conv1(t1, t2, gt::SAME) == correct_same));
    assert(gt::all(gt::sp::conv1(t1, t2, gt::VALID) == correct_valid));
}

void conv2_test()
{
    gt::Tensor<float> t1({3, 4});
    t1 = {2, 5, 7, 3, 4, 9, 6, 1, 0, 8, 4, 8};
    gt::Tensor<float> t2({2, 3});
    t2 = {5, 9, 8, 2, 1, 3};

    gt::Tensor<float> correct_full({4, 6});
    correct_full = {10, 43, 80, 63,
                    31, 91, 147, 95,
                    56, 108, 111, 39,
                    91, 125, 99, 99,
                    70, 67, 75, 16,
                    8, 28, 20, 24};

    gt::Tensor<float> correct_same({3, 4});
    correct_same = {91, 147, 95, 108, 111, 39, 125, 99, 99, 67, 75, 16};

    gt::Tensor<float> correct_valid({2, 2});
    correct_valid = {108, 111, 125, 99};

    assert(gt::all(gt::sp::conv2(t1, t2, gt::FULL) == correct_full));
    assert(gt::all(gt::sp::conv2(t1, t2, gt::SAME) == correct_same));
    assert(gt::all(gt::sp::conv2(t1, t2, gt::VALID) == correct_valid));
}

void conv3_test()
{
    gt::Tensor<float> t1({2, 3, 4});
    t1 = {2, 5, 7, 3, 4, 9, 6, 1, 0, 8, 4, 8,
          3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8};
    gt::Tensor<float> t2({1, 2, 3});
    t2 = {5, 9, 8, 2, 1, 3};

    gt::Tensor<float> correct_full({2, 4, 6});
    correct_full = {10, 25, 53, 60, 83, 72, 36, 81,
                    46, 45, 114, 83, 66, 190, 44, 90,
                    65, 18, 72, 98, 118, 152, 65, 124,
                    40, 39, 99, 90, 122, 173, 67, 114,
                    19, 49, 57, 40, 67, 82, 25, 43,
                    2, 6, 11, 21, 20, 17, 15, 24};

    gt::Tensor<float> correct_same({2, 3, 4});
    correct_same = {114, 83, 66, 190, 44, 90,
                    72, 98, 118, 152, 65, 124,
                    99, 90, 122, 173, 67, 114,
                    57, 40, 67, 82, 25, 43};

    gt::Tensor<float> correct_valid({2, 2, 2});
    correct_valid = {72, 98, 118, 152, 99, 90, 122, 173};

    assert(gt::all(gt::sp::conv3(t1, t2, gt::FULL) == correct_full));
    assert(gt::all(gt::sp::conv3(t1, t2, gt::SAME) == correct_same));
    assert(gt::all(gt::sp::conv3(t1, t2, gt::VALID) == correct_valid));
}

void broadcast_test()
{
    gt::Tensor<float> t1({1, 4});
    std::iota(t1.begin(), t1.end(), 0);
    gt::Tensor<float> t2({5, 1});
    std::iota(t2.begin(), t2.end(), 0);

    gt::Tensor<float> correct({5, 4});
    correct = {0, 1, 2, 3, 4,
               1, 2, 3, 4, 5,
               2, 3, 4, 5, 6,
               3, 4, 5, 6, 7};

    assert(gt::all(gt::broadcast(t1, t2, gt::PLUS) == correct));
}

void rect_test()
{
    gt::Tensor<float> correct({8});
    correct = {1, 1, 1, 1, 1, 1, 1, 1};

    assert(gt::all(gt::sp::rect(8) == correct));
}

void bartlett_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.2857, 0.5714, 0.8571, 0.8571, 0.5714, 0.2857, 0};
    assert(!gt::any(gt::abs(gt::sp::bartlett(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0};
    assert(!gt::any(gt::abs(gt::sp::bartlett(9) - correct_odd) > 1e-4f));
}

void triang_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.125, 0.375, 0.625, 0.875, 0.875, 0.625, 0.375, 0.125};
    assert(!gt::any(gt::abs(gt::sp::triang(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2};
    assert(!gt::any(gt::abs(gt::sp::triang(9) - correct_odd) > 1e-4f));
}

void barthann_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.2116, 0.6017, 0.9281, 0.9281, 0.6017, 0.2116, 0};
    assert(!gt::any(gt::abs(gt::sp::barthann(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.1713, 0.5000, 0.8287, 1.0000, 0.8287, 0.5000, 0.1713, 0};
    assert(!gt::any(gt::abs(gt::sp::barthann(9) - correct_odd) > 1e-4f));
}

void blackman_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.0905, 0.4592, 0.9204, 0.9204, 0.4592, 0.0905, 0};
    assert(!gt::any(gt::abs(gt::sp::blackman(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.0664, 0.3400, 0.7736,1.0000, 0.7736, 0.3400, 0.0664, 0};
    assert(!gt::any(gt::abs(gt::sp::blackman(9) - correct_odd) > 1e-4f));
}

int main()
{
    access_test();
    mod_test();
    rem_test();
    linspace_test();
    logspace_test();
    sum_test();
    cumsum_test();
    movsum_test();
    diff_test();
    prod_test();
    cumprod_test();
    movprod_test();
    trapz_test();
    cumtrapz_test();
    max_test();
    cummax_test();
    movmax_test();
    min_test();
    cummin_test();
    movmin_test();
    mean_test();
    movmean_test();
    median_test();
    var_test();
    stddev_test();
    reshape_test();
    squeeze_test();
    flatten_test();
    permute_test();
    ipermute_test();
    repmat_test();
    circshift_test();
    flip_test();
    cart2sph_test();
    sph2cart_test();
    cart2pol_test();
    pol2cart_test();
    cat_test();
    unique_test();
    matmul_test();
    interp1_test();
    interp2_test();
    conv1_test();
    conv2_test();
    conv3_test();
    broadcast_test();
    rect_test();
    bartlett_test();
    triang_test();
    barthann_test();
    blackman_test();
}
