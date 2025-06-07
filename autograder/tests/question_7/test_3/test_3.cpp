//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<int, 3> t1(2, 5, 3);
    utec::algebra::Tensor<int, 3> t2(2, 3, 7);
    std::iota(t1.begin(), t1.end(), 10);
    std::iota(t2.begin(), t2.end(), 5);
    const auto r = utec::algebra::matrix_product(t1, t2);
    std::cout << r;
}

TEST_CASE("Question #7.3") {
    execute_test("question_7_test_3.in", test_3);
}