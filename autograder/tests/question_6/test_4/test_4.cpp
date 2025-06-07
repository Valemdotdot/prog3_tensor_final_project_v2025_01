//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<double, 4> t1(2, 2, 2, 3);
    std::iota(t1.begin(), t1.end(), 10.5);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
}

TEST_CASE("Question #6.4") {
    execute_test("question_6_test_4.in", test_4);
}