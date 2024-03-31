#include <cassert>
#include <iostream>

#include "Tensor.h"
#include "TensorOperations.h"

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

    gt::Tensor<float> correct210({3, 4, 2});
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

#if 0
void cat_test()
{
    gt::Tensor<float> input({2, 3, 4});
    std::iota(input.begin(), input.end(), 0);

    gt::Tensor<float> actual = cat(2, input, input);
    std::cout << actual << std::endl;
}

void matmul_test()
{
    gt::Tensor<float> lhs({2, 3});
    std::iota(lhs.begin(), lhs.end(), 0);
    gt::Tensor<float> rhs({3, 2});
    std::iota(rhs.begin(), rhs.end(), 0);

    gt::Tensor<float> actual = matmul(lhs, rhs);
    gt::Tensor<float> correct({2, 2});
    correct = {10, 13, 28, 40};
    assert(gt::all(actual == correct) && "Failed matmul input");
}
#endif

int main()
{
    access_test();
    mod_test();
    rem_test();
    linspace_test();
    logspace_test();
    sum_test();
    cumsum_test();
    diff_test();
    prod_test();
    cumprod_test();
    trapz_test();
    cumtrapz_test();
    max_test();
    cummax_test();
    min_test();
    cummin_test();
    mean_test();
    var_test();
    stddev_test();
    reshape_test();
    squeeze_test();
    permute_test();
    ipermute_test();
    repmat_test();
    circshift_test();
    flip_test();
    cart2sph_test();
    sph2cart_test();
    //cat_test();
    //matmul_test();
}
