#include <cassert>
#include <iostream>

#include "Tensor.h"
#include "TensorOperations.h"

/* Tests that the () operator overload is indexing correctly */
void access_test()
{
    gt::Tensor<size_t> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    size_t compare = 0;
    for (size_t k = 0; k < test.shape(2); k++) {
        for (size_t j = 0; j < test.shape(1); j++) {
            for (size_t i = 0; i < test.shape(0); i++) {
                assert(test(i,j,k) == compare && "Failed array access test");
                compare++;
            }
        }
    }
}

void linspace_test()
{
    gt::Tensor<float> actual = gt::linspace(-1.0f, 1.0f, 21);

    gt::Tensor<float> correct({21});
    correct = {-1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f,
        -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(actual - correct) > 1e-4f)  && "Failed linspace test");
}

void logspace_test()
{
    gt::Tensor<float> actual = gt::logspace(-5.0f, 5.0f, 11);
    gt::Tensor<float> correct({11});
    correct = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000};
    assert(gt::all(actual == correct) && "Failed logspace test");
}

void sum_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 5, 9,
               13, 17, 21,
               25, 29, 33,
               37, 41, 45};
    assert(gt::all(gt::sum(test, 0) == correct0) && "Failed sum dimension 0 test");

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {6, 9,
                24, 27,
                42, 45,
                60, 63};
    assert(gt::all(gt::sum(test, 1) == correct1) && "Failed sum dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {36, 40,
                44, 48,
                52, 56};
    assert(gt::all(gt::sum(test, 2) == correct2) && "Failed sum dimension 2 test");
}

void cumsum_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 5, 4, 9, 6, 13, 8, 17, 10, 21, 12, 25, 14, 29, 16, 33, 18, 37, 20, 41, 22, 45};
    assert(gt::all(gt::cumsum(test, 0) == correct0) && "Failed cumsum dimension 0 test");

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 4, 6, 9, 6, 7, 14, 16, 24, 27, 12, 13, 26, 28, 42, 45, 18, 19, 38, 40, 60, 63};
    assert(gt::all(gt::cumsum(test, 1) == correct1) && "Failed cumsum dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30, 33, 36, 40, 44, 48, 52, 56};
    assert(gt::all(gt::cumsum(test, 2) == correct2) && "Failed cumsum dimension 2 test");
}

void diff_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 1, 1,
                1, 1, 1,
                1, 1, 1,
                1, 1, 1};
    assert(gt::all(gt::diff(test, 0) == correct0) && "Failed diff dimension 0 test");

    gt::Tensor<float> correct1({2, 2, 4});
    correct1 = {2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2,
                2, 2, 2, 2};
    assert(gt::all(gt::diff(test, 1) == correct1) && "Failed diff dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 3});
    correct2 = {6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6};
    assert(gt::all(gt::diff(test, 2) == correct2) && "Failed diff dimension 2 test");
}

void prod_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0, 6, 20, 42, 72, 110, 156, 210, 272, 342, 420, 506};
    assert(gt::all(gt::prod(test, 0) == correct0) && "Failed prod dimension 0 test");

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {0, 15, 480, 693, 2688, 3315, 7920, 9177};
    assert(gt::all(gt::prod(test, 1) == correct1) && "Failed prod dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {0, 1729, 4480, 8505, 14080, 21505};
    assert(gt::all(gt::prod(test, 2) == correct2) && "Failed prod dimension 2 test");
}

void cumprod_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 6, 4, 20, 
                6, 42, 8, 72, 10, 110, 
                12, 156, 14, 210, 16, 272, 
                18, 342, 20, 420, 22, 506};
    assert(gt::all(gt::cumprod(test, 0) == correct0) && "Failed cumprod dimension 0 test");

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 3, 0, 15,
                6, 7, 48, 63, 480, 693,
                12, 13, 168, 195, 2688, 3315,
                18, 19, 360, 399, 7920, 9177};
    assert(gt::all(gt::cumprod(test, 1) == correct1) && "Failed cumprod dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5,
                0, 7, 16, 27, 40, 55,
                0, 91, 224, 405, 640, 935,
                0, 1729, 4480, 8505, 14080, 21505};
    assert(gt::all(gt::cumprod(test, 2) == correct2) && "Failed cumprod dimension 2 test");
}

void trapz_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(gt::trapz(test, 0) == correct0) && "Failed trapz dimension 0 test");

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {4, 6, 16, 18, 28, 30, 40, 42};
    assert(gt::all(gt::trapz(test, 1) == correct1) && "Failed trapz dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {27, 30, 33, 36, 39, 42};
    assert(gt::all(gt::trapz(test, 2) == correct2) && "Failed trapz dimension 2 test");
}

void cumtrapz_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0.5, 0, 2.5, 0, 4.5, 0, 6.5, 0, 8.5, 0, 10.5, 0, 12.5, 0, 14.5, 0, 16.5, 0, 18.5, 0, 20.5, 0, 22.5};
    assert(gt::all(gt::cumtrapz(test, 0) == correct0) && "Failed cumtrapz dimension 0 test");

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 0, 1, 2, 4, 6, 0, 0, 7, 8, 16, 18, 0, 0, 13, 14, 28, 30, 0, 0, 19, 20, 40, 42};
    assert(gt::all(gt::cumtrapz(test, 1) == correct1) && "Failed cumtrapz dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36, 39, 42};
    assert(gt::all(gt::cumtrapz(test, 2) == correct2) && "Failed cumtrapz dimension 2 test");
}

void max_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    assert(gt::all(gt::max(test, 0) == correct0) && "Failed max dimension 0 test");

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {4, 5, 10, 11, 16, 17, 22, 23};
    assert(gt::all(gt::max(test, 1) == correct1) && "Failed max dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::max(test, 2) == correct2) && "Failed max dimension 2 test");
}

void cummax_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(test, 0) == correct0) && "Failed cummax dimension 0 test");

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(test, 1) == correct1) && "Failed cummax dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 11, 
                12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::cummax(test, 2) == correct2) && "Failed cummax dimension 2 test");
}

void min_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({1, 3, 4});
    correct0 = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    assert(gt::all(gt::min(test, 0) == correct0) && "Failed min dimension 0 test");

    gt::Tensor<float> correct1({2, 1, 4});
    correct1 = {0, 1, 6, 7, 12, 13, 18, 19};
    assert(gt::all(gt::min(test, 1) == correct1) && "Failed min dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 1});
    correct2 = {0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::min(test, 2) == correct2) && "Failed min dimension 2 test");
}

void cummin_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct0({2, 3, 4});
    correct0 = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22};
    assert(gt::all(gt::cummin(test, 0) == correct0) && "Failed cummin dimension 0 test");

    gt::Tensor<float> correct1({2, 3, 4});
    correct1 = {0, 1, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 12, 13, 12, 13, 12, 13, 18, 19, 18, 19, 18, 19};
    assert(gt::all(gt::cummin(test, 1) == correct1) && "Failed cummin dimension 1 test");

    gt::Tensor<float> correct2({2, 3, 4});
    correct2 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    assert(gt::all(gt::cummin(test, 2) == correct2) && "Failed cummin dimension 2 test");
}

void reshape_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct({1, 24});
    std::iota(correct.begin(), correct.end(), 0);
    assert(gt::all(gt::reshape(test, {1, 24}) == correct) && "Failed reshape test");
}

void permute_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct012({2, 3, 4});
    correct012 = {0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11,
                  12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23};
    assert(gt::all(gt::permute(test, {0, 1, 2}) == correct012) && "Failed permute {0, 1, 2} test");

    gt::Tensor<float> correct021({2, 4, 3});
    correct021 = {0, 1, 6, 7, 12, 13, 18, 19,
                  2, 3, 8, 9, 14, 15, 20, 21,
                  4, 5, 10, 11, 16, 17, 22, 23};
    assert(gt::all(gt::permute(test, {0, 2, 1}) == correct021) && "Failed permute {0, 2, 1} test");

    gt::Tensor<float> correct102({3, 2, 4});
    correct102 = {0, 2, 4, 1, 3, 5,
                  6, 8, 10, 7, 9, 11, 
                  12, 14, 16, 13, 15, 17, 
                  18, 20, 22, 19, 21, 23};
    assert(gt::all(gt::permute(test, {1, 0, 2}) == correct102) && "Failed permute {1, 0, 2} test");

    gt::Tensor<float> correct120({3, 4, 2});
    correct120 = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    assert(gt::all(gt::permute(test, {1, 2, 0}) == correct120) && "Failed permute {1, 2, 0} test");

    gt::Tensor<float> correct201({4, 2, 3});
    correct201 = {0, 6, 12, 18, 1, 7, 13, 19,
                  2, 8, 14, 20, 3, 9, 15, 21,
                  4, 10, 16, 22, 5, 11, 17, 23};
    assert(gt::all(gt::permute(test, {2, 0, 1}) == correct201) && "Failed permute {2, 0, 1} test");

    gt::Tensor<float> correct210({3, 4, 2});
    correct210 = {0, 6, 12, 18, 2, 8, 14, 20, 4, 10, 16, 22,
                  1, 7, 13, 19, 3, 9, 15, 21, 5, 11, 17, 23};
    assert(gt::all(gt::permute(test, {2, 1, 0}) == correct210) && "Failed permute {2, 1, 0} test");
}

void repmat_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::repmat(test, {2, 2, 2});
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
    assert(gt::all(actual == correct) && "Failed repmat test");
}

void mod_test()
{
    gt::Tensor<float> actual = gt::mod(gt::linspace(-1.0f, 1.0f, 21), 2.0f);

    gt::Tensor<float> correct({21});
    correct = {1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
        1.9f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(actual - correct) > 1e-4f)  && "Failed mod test");
}

void rem_test()
{
    gt::Tensor<float> actual = gt::rem(gt::linspace(-1.0f, 1.0f, 21), 2.0f);

    gt::Tensor<float> correct({21});
    correct = {-1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f,
        -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    assert(!gt::any(gt::abs(actual - correct) > 1e-4f)  && "Failed rem test");
}

void mean_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(gt::mean(test, 0) == correct) && "Failed mean test");
}

void var_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    assert(gt::all(gt::var(test, 0) == correct) && "Failed var test");
}

void stddev_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    assert(gt::all(gt::stddev(test, 0) == correct) && "Failed stddev test");
}

#if 0
void cat_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = cat(2, test, test);
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
    assert(gt::all(actual == correct) && "Failed matmul test");
}
#endif

int main()
{
    access_test();
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
    repmat_test();
    permute_test();
    //cat_test();
    //matmul_test();
    mod_test();
    rem_test();
}
