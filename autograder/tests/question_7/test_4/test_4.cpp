//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<int, 3> t1(2, 3, 2);
    utec::algebra::Tensor<int, 3> t2(2, 2, 4);
    std::iota(t1.begin(), t1.end(), 1);
    std::iota(t2.begin(), t2.end(), 3);
    const auto r = utec::algebra::matrix_product(t1, t2);
    std::cout << t1 << '\n';
    std::cout << t2 << '\n';
    std::cout << r;
}

TEST_CASE("Question #7.4") {
    execute_test("question_7_test_4.in", test_4);
}