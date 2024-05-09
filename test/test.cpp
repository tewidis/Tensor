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
#include "Random.h"
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

    gt::Tensor<float> correct3({2, 3, 4});
    correct3 = {23, 22, 19, 18, 21, 20,
                5, 4, 1, 0, 3, 2,
                11, 10, 7, 6, 9, 8,
                17, 16, 13, 12, 15, 14};
    assert(gt::all(gt::circshift(input, {1, 1, 1}) == correct3));
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

void fft_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<std::complex<float>> correct0({2, 3, 4});
    correct0 = {{1, 0}, {-1, 0}, {5, 0}, {-1, 0}, {9, 0}, {-1, 0},
        {13, 0}, {-1, 0}, {17, 0}, {-1, 0}, {21, 0}, {-1, 0},
        {25, 0}, {-1, 0}, {29, 0}, {-1, 0}, {33, 0}, {-1, 0},
        {37, 0}, {-1, 0}, {41, 0}, {-1, 0}, {45, 0}, {-1, 0}};

    gt::Tensor<std::complex<float>> correct1({2, 3, 4});
    correct1 = {{6, 0}, {9, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {24, 0}, {27, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {42, 0}, {45, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {60, 0}, {63, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732}};

    gt::Tensor<std::complex<float>> correct2({2, 3, 4});
    correct2 = {{36, 0}, {40, 0}, {44, 0}, {48, 0}, {52, 0}, {56, 0},
        {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12},
        {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0},
        {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}};

    assert(!gt::any(gt::abs(gt::sp::fft(input, 2, 0) - correct0) > 1e-4f));
    assert(!gt::any(gt::abs(gt::sp::fft(input, 3, 1) - correct1) > 1e-4f));
    assert(!gt::any(gt::abs(gt::sp::fft(input, 4, 2) - correct2) > 1e-4f));
}

void ifft_test()
{
    gt::Tensor<std::complex<float>> input0({2, 3, 4});
    input0 = {{1, 0}, {-1, 0}, {5, 0}, {-1, 0}, {9, 0}, {-1, 0},
        {13, 0}, {-1, 0}, {17, 0}, {-1, 0}, {21, 0}, {-1, 0},
        {25, 0}, {-1, 0}, {29, 0}, {-1, 0}, {33, 0}, {-1, 0},
        {37, 0}, {-1, 0}, {41, 0}, {-1, 0}, {45, 0}, {-1, 0}};

    gt::Tensor<std::complex<float>> input1({2, 3, 4});
    input1 = {{6, 0}, {9, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {24, 0}, {27, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {42, 0}, {45, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732},
        {60, 0}, {63, 0}, {-3, 1.732}, {-3, 1.732}, {-3, -1.732}, {-3, -1.732}};

    gt::Tensor<std::complex<float>> input2({2, 3, 4});
    input2 = {{36, 0}, {40, 0}, {44, 0}, {48, 0}, {52, 0}, {56, 0},
        {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12}, {-12, 12},
        {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0}, {-12, 0},
        {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}, {-12, -12}};

    gt::Tensor<float> correct({2, 3, 4});
    std::iota(correct.begin(), correct.end(), 0);

    assert(!gt::any(gt::abs(gt::sp::ifft(input0, 2, 0) - correct) > 1e-4f));
    assert(!gt::any(gt::abs(gt::sp::ifft(input1, 3, 1) - correct) > 1e-4f));
    assert(!gt::any(gt::abs(gt::sp::ifft(input2, 4, 2) - correct) > 1e-4f));
}

void fftshift_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {1, 0, 3, 2, 5, 4,
                7, 6, 9, 8, 11, 10,
                13, 12, 15, 14, 17, 16,
                19, 18, 21, 20, 23, 22};
    assert(gt::all(gt::sp::fftshift(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {4, 5, 0, 1, 2, 3,
                10, 11, 6, 7, 8, 9,
                16, 17, 12, 13, 14, 15,
                22, 23, 18, 19, 20, 21};
    assert(gt::all(gt::sp::fftshift(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11};
    assert(gt::all(gt::sp::fftshift(input, 2) == correct2));

    gt::Tensor<float> correct3({2, 3, 4});
    correct3 = {17, 16, 13, 12, 15, 14,
                23, 22, 19, 18, 21, 20,
                5, 4, 1, 0, 3, 2,
                11, 10, 7, 6, 9, 8};
    assert(gt::all(gt::sp::fftshift(input) == correct3));
}

void ifftshift_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {1, 0, 3, 2, 5, 4,
                7, 6, 9, 8, 11, 10,
                13, 12, 15, 14, 17, 16,
                19, 18, 21, 20, 23, 22};
    assert(gt::all(gt::sp::ifftshift(input, 0) == correct0));

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {2, 3, 4, 5, 0, 1,
                8, 9, 10, 11, 6, 7,
                14, 15, 16, 17, 12, 13,
                20, 21, 22, 23, 18, 19};
    assert(gt::all(gt::sp::ifftshift(input, 1) == correct1));

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11};
    assert(gt::all(gt::sp::ifftshift(input, 2) == correct2));

    gt::Tensor<float> correct3({2, 3, 4});
    correct3 = {15, 14, 17, 16, 13, 12,
                21, 20, 23, 22, 19, 18,
                3, 2, 5, 4, 1, 0,
                9, 8, 11, 10, 7, 6};
    assert(gt::all(gt::sp::ifftshift(input) == correct3));
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

void bartlett_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.2857, 0.5714, 0.8571, 0.8571, 0.5714, 0.2857, 0};
    assert(!gt::any(gt::abs(gt::sp::bartlett<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0};
    assert(!gt::any(gt::abs(gt::sp::bartlett<float>(9) - correct_odd) > 1e-4f));
}

void barthann_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.2116, 0.6017, 0.9281, 0.9281, 0.6017, 0.2116, 0};
    assert(!gt::any(gt::abs(gt::sp::barthann<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.1713, 0.5000, 0.8287, 1.0000, 0.8287, 0.5000, 0.1713, 0};
    assert(!gt::any(gt::abs(gt::sp::barthann<float>(9) - correct_odd) > 1e-4f));
}

void blackman_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.0905, 0.4592, 0.9204, 0.9204, 0.4592, 0.0905, 0};
    assert(!gt::any(gt::abs(gt::sp::blackman<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.0664, 0.3400, 0.7736,1.0000, 0.7736, 0.3400, 0.0664, 0};
    assert(!gt::any(gt::abs(gt::sp::blackman<float>(9) - correct_odd) > 1e-4f));
}

void blackmanharris_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0001, 0.0334, 0.3328, 0.8894, 0.8894, 0.3328, 0.0334, 0.0001};
    assert(!gt::any(gt::abs(gt::sp::blackmanharris<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0001, 0.0217, 0.2175, 0.6958, 1.0000, 0.6958, 0.2175, 0.0217, 0.0001};
    assert(!gt::any(gt::abs(gt::sp::blackmanharris<float>(9) - correct_odd) > 1e-4f));
}

void bohman_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.0707, 0.4375, 0.9104, 0.9104, 0.4375, 0.0707, 0};
    assert(!gt::any(gt::abs(gt::sp::bohman<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.0483, 0.3183, 0.7554, 1.0000, 0.7554, 0.3183, 0.0483, 0};
    assert(!gt::any(gt::abs(gt::sp::bohman<float>(9) - correct_odd) > 1e-4f));
}

void chebyshev_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0364, 0.2254, 0.6242, 1.0000, 1.0000, 0.6242, 0.2254, 0.0364};
    assert(!gt::any(gt::abs(gt::sp::chebyshev<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0218, 0.1445, 0.4435, 0.8208, 1.0000, 0.8208, 0.4435, 0.1445, 0.0218};
    assert(!gt::any(gt::abs(gt::sp::chebyshev<float>(9) - correct_odd) > 1e-4f));
}

void flattop_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {-0.0004, -0.0368, 0.0107, 0.7809, 0.7809, 0.0107, -0.0368, -0.0004};
    assert(!gt::any(gt::abs(gt::sp::flattop<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {-0.0004, -0.0269, -0.0547, 0.4441, 1.0000, 0.4441, -0.0547, -0.0269, -0.0004};
    assert(!gt::any(gt::abs(gt::sp::flattop<float>(9) - correct_odd) > 1e-4f));
}

void gaussian_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0439, 0.2030, 0.5633, 0.9382, 0.9382, 0.5633, 0.2030, 0.0439};
    assert(!gt::any(gt::abs(gt::sp::gaussian<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0439, 0.1724, 0.4578, 0.8226, 1.0000, 0.8226, 0.4578, 0.1724, 0.0439};
    assert(!gt::any(gt::abs(gt::sp::gaussian<float>(9) - correct_odd) > 1e-4f));
}

void hamming_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0800, 0.2532, 0.6424, 0.9544, 0.9544, 0.6424, 0.2532, 0.0800};
    assert(!gt::any(gt::abs(gt::sp::hamming<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0800, 0.2147, 0.5400, 0.8653, 1.0000, 0.8653, 0.5400, 0.2147, 0.0800};
    assert(!gt::any(gt::abs(gt::sp::hamming<float>(9) - correct_odd) > 1e-4f));
}

void hann_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.1883, 0.6113, 0.9505, 0.9505, 0.6113, 0.1883, 0};
    assert(!gt::any(gt::abs(gt::sp::hann<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.1464, 0.5000, 0.8536, 1.0000, 0.8536, 0.5000, 0.1464, 0};
    assert(!gt::any(gt::abs(gt::sp::hann<float>(9) - correct_odd) > 1e-4f));
}

void hanning_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170};
    assert(!gt::any(gt::abs(gt::sp::hanning<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955};
    assert(!gt::any(gt::abs(gt::sp::hanning<float>(9) - correct_odd) > 1e-4f));
}

void kaiser_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.9403, 0.9693, 0.9889, 0.9988, 0.9988, 0.9889, 0.9693, 0.9403};
    assert(!gt::any(gt::abs(gt::sp::kaiser<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.9403, 0.9662, 0.9849, 0.9962, 1.0000, 0.9962, 0.9849, 0.9662, 0.9403};
    assert(!gt::any(gt::abs(gt::sp::kaiser<float>(9) - correct_odd) > 1e-4f));
}

void nuttall_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0004, 0.0378, 0.3427, 0.8919, 0.8919, 0.3427, 0.0378, 0.0004};
    assert(!gt::any(gt::abs(gt::sp::nuttall<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0004, 0.0252, 0.2270, 0.7020, 1.0000, 0.7020, 0.2270, 0.0252, 0.0004};
    assert(!gt::any(gt::abs(gt::sp::nuttall<float>(9) - correct_odd) > 1e-4f));
}

void parzen_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.0039, 0.1055, 0.4727, 0.9180, 0.9180, 0.4727, 0.1055, 0.0039};
    assert(!gt::any(gt::abs(gt::sp::parzen<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.0027, 0.0741, 0.3416, 0.7695, 1.0000, 0.7695, 0.3416, 0.0741, 0.0027};
    assert(!gt::any(gt::abs(gt::sp::parzen<float>(9) - correct_odd) > 1e-4f));
}

void rect_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {1, 1, 1, 1, 1, 1, 1, 1};
    assert(!gt::any(gt::abs(gt::sp::rect<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    assert(!gt::any(gt::abs(gt::sp::rect<float>(9) - correct_odd) > 1e-4f));
}

void taylor_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.4353, 0.8024, 1.2423, 1.5201, 1.5201, 1.2423, 0.8024, 0.4353};
    assert(!gt::any(gt::abs(gt::sp::taylor<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.4236, 0.7275, 1.1291, 1.4407, 1.5581, 1.4407, 1.1291, 0.7275, 0.4236};
    assert(!gt::any(gt::abs(gt::sp::taylor<float>(9) - correct_odd) > 1e-4f));
}

void triang_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0.125, 0.375, 0.625, 0.875, 0.875, 0.625, 0.375, 0.125};
    assert(!gt::any(gt::abs(gt::sp::triang<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2};
    assert(!gt::any(gt::abs(gt::sp::triang<float>(9) - correct_odd) > 1e-4f));
}

void tukey_test()
{
    gt::Tensor<float> correct_even({8});
    correct_even = {0, 0.6113, 1.0000, 1.0000, 1.0000, 1.0000, 0.6113, 0};
    assert(!gt::any(gt::abs(gt::sp::tukey<float>(8) - correct_even) > 1e-4f));

    gt::Tensor<float> correct_odd({9});
    correct_odd = {0, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0};
    assert(!gt::any(gt::abs(gt::sp::tukey<float>(9) - correct_odd) > 1e-4f));
}

void random_test()
{
    /* uniform distributions */
    gt::Tensor<float> rand = gt::rand::rand<float>({2, 3, 4});
    gt::Tensor<int32_t> randi = gt::rand::randi<int32_t>({-10, 10}, {2, 3, 4});

    /* bernoulli distributions */
    gt::Tensor<bool> bern = gt::rand::bernoulli(0.5f, {2, 3, 4});
    gt::Tensor<int32_t> binomial = gt::rand::binomial(4, 0.5, {2, 3, 4});
    gt::Tensor<int32_t> neg_binomial = gt::rand::negative_binomial(4, 0.5, {2, 3, 4});
    gt::Tensor<float> geometric = gt::rand::geometric(0.5f, {2, 3, 4});

    /* poisson distributions */
    gt::Tensor<int32_t> poisson = gt::rand::poisson(1, {2, 3, 4});
    gt::Tensor<float> exponential = gt::rand::exponential(1.0f, {2, 3, 4});
    gt::Tensor<float> gamma = gt::rand::gamma(1.0f, 2.0f, {2, 3, 4});
    gt::Tensor<float> weibull = gt::rand::weibull(1.0f, 2.0f, {2, 3, 4});
    gt::Tensor<float> extreme_value = gt::rand::extreme_value(-1.618f, 1.618f, {2, 3, 4});

    /* normal distributions */
    gt::Tensor<float> randn = gt::rand::randn(1.6f, 0.25f, {2, 3, 4});
    gt::Tensor<float> logn = gt::rand::lognormal(1.6f, 0.25f, {2, 3, 4});
    gt::Tensor<float> chi_squared = gt::rand::chi_squared(2.0f, {2, 3, 4});
    gt::Tensor<float> cauchy = gt::rand::cauchy(-2.0f, 0.5f, {2, 3, 4});
    gt::Tensor<float> fisher_f = gt::rand::fisher_f(1.0f, 5.0f, {2, 3, 4});
    gt::Tensor<float> student_t = gt::rand::student_t(10.0f, {2, 3, 4});
}

void meshgrid_test()
{
    gt::Tensor<float> t0({3, 4});
    std::iota(t0.begin(), t0.end(), 0);
    gt::Tensor<float> t1({2, 3});
    std::iota(t1.begin(), t1.end(), 0);
    gt::Tensor<float> t2({1, 2});
    std::iota(t2.begin(), t2.end(), 0);

    gt::Tensor<float> correct0({6, 12, 2});
    correct0 = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
                8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
                10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
                0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
                8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
                10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11};

    gt::Tensor<float> correct1({6, 12, 2});
    correct1 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};

    gt::Tensor<float> correct2({6, 12, 2});
    correct2 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::tuple<gt::Tensor<float>,gt::Tensor<float>,gt::Tensor<float>> output = meshgrid(t0, t1, t2);

    assert(!gt::any(gt::abs(std::get<0>(output) - correct0) > 1e-4f));
    assert(!gt::any(gt::abs(std::get<1>(output) - correct1) > 1e-4f));
    assert(!gt::any(gt::abs(std::get<2>(output) - correct2) > 1e-4f));
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
    fft_test();
    ifft_test();
    fftshift_test();
    ifftshift_test();
    broadcast_test();
    bartlett_test();
    barthann_test();
    blackman_test();
    blackmanharris_test();
    bohman_test();
    chebyshev_test();
    flattop_test();
    gaussian_test();
    hamming_test();
    hann_test();
    hanning_test();
    kaiser_test();
    nuttall_test();
    parzen_test();
    rect_test();
    taylor_test();
    triang_test();
    tukey_test();
    random_test();
    meshgrid_test();
}
