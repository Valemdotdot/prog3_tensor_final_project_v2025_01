//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    utec::algebra::Tensor<int, 2> t1(19, 24);
    std::iota(t1.begin(), t1.end(), 1);
    std::cout << t1 << '\n';
    const auto r = transpose_2d(t1);
    std::cout << r;
}

TEST_CASE("Question #6.2") {
    execute_test("question_6_test_2.in", test_2);
}