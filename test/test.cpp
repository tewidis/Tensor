#include <cassert>
#include <iostream>

#include "Tensor.h"
#include "TensorOperations.h"

/* Tests that the () operator overload is indexing correctly */
void access_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    float compare = 0;
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
    gt::Tensor<float> actual = gt::linspace(-5.0f, 5.0f, 11);
    gt::Tensor<float> correct({11});
    correct = {-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    assert(gt::all(actual == correct) && "Failed linspace test");
}

void logspace_test()
{
    gt::Tensor<float> actual = gt::logspace(-5.0f, 5.0f, 11);
    gt::Tensor<float> correct({11});
    correct = {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000};
    assert(gt::all(actual == correct) && "Failed logspace test");
}

#if 0
void sum_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::sum(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {1, 5, 9,
               13, 17, 21,
               25, 29, 33,
               37, 41, 45};
    assert(gt::all(actual == correct) && "Failed sum test");
}

void cumsum_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::cumsum(test, 0);
    gt::Tensor<float> correct({2, 3, 4});
    correct = {0, 1, 2, 5, 4, 9, 6, 13, 8, 17, 10, 21, 12, 25, 14, 29, 16, 33, 18, 37, 20, 41, 22, 45};
    assert(gt::all(actual == correct) && "Failed cumsum test");
}

void diff_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::diff(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {1, 1, 1,
               1, 1, 1,
               1, 1, 1,
               1, 1, 1};
    assert(gt::all(actual == correct) && "Failed diff test");
}

void prod_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::prod(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0, 6, 20, 42, 72, 110, 156, 210, 272, 342, 420, 506};
    assert(gt::all(actual == correct) && "Failed prod test");
}

void cumprod_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::cumprod(test, 0);
    gt::Tensor<float> correct({2, 3, 4});
    correct = {0, 0, 2, 6, 4, 20, 6, 42, 8, 72, 10, 110, 12, 156, 14, 210, 16, 272, 18, 342, 20, 420, 22, 506};
    assert(gt::all(actual == correct) && "Failed cumprod test");
}

void trapz_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::trapz(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(actual == correct) && "Failed trapz test");
}

void cumtrapz_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::cumtrapz(test, 0);
    gt::Tensor<float> correct({2, 3, 4});
    correct = {0, 0.5, 0, 2.5, 0, 4.5, 0, 6.5, 0, 8.5, 0, 10.5, 0, 12.5, 0, 14.5, 0, 16.5, 0, 18.5, 0, 20.5, 0, 22.5};
    assert(gt::all(actual == correct) && "Failed cumtrapz test");
}

void max_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::max(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
    assert(gt::all(actual == correct) && "Failed max test");
}

void cummax_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::cummax(test, 0);
    gt::Tensor<float> correct({2, 3, 4});
    correct = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    assert(gt::all(actual == correct) && "Failed cummax test");
}

void min_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::min(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    assert(gt::all(actual == correct) && "Failed min test");
}

void cummin_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::cummin(test, 0);
    gt::Tensor<float> correct({2, 3, 4});
    correct = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22};
    assert(gt::all(actual == correct) && "Failed cummin test");
}

void mean_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::mean(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5};
    assert(gt::all(actual == correct) && "Failed mean test");
}

void var_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::var(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    assert(gt::all(actual == correct) && "Failed var test");
}

void stddev_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::stddev(test, 0);
    gt::Tensor<float> correct({1, 3, 4});
    correct = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    assert(gt::all(actual == correct) && "Failed stddev test");
}

void reshape_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::reshape(test, {1, 24});
    gt::Tensor<float> correct({1, 24});
    std::iota(correct.begin(), correct.end(), 0);
    assert(gt::all(actual == correct) && "Failed reshape test");
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

void permute_test()
{
    gt::Tensor<float> test({2, 3, 4});
    std::iota(test.begin(), test.end(), 0);

    gt::Tensor<float> actual = gt::permute(test, {1, 0, 2});
    gt::Tensor<float> correct({3, 2, 4});
    correct = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19, 21, 23};
    assert(gt::all(actual == correct) && "Failed permute test");
}

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
    //sum_test();
    //cumsum_test();
    //diff_test();
    //prod_test();
    //cumprod_test();
    //trapz_test();
    //cumtrapz_test();
    //max_test();
    //cummax_test();
    //min_test();
    //cummin_test();
    //mean_test();
    //var_test();
    //stddev_test();
    //reshape_test();
    //repmat_test();
    //permute_test();
    //cat_test();
    //matmul_test();
}
