//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    utec::algebra::Tensor<int, 3> t1(2, 4, 3);
    std::iota(t1.begin(), t1.end(), 10);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
}

TEST_CASE("Question #6.1") {
    execute_test("question_6_test_1.in", test_1);
}