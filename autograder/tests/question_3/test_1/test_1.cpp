//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    utec::algebra::Tensor<double, 2> a(8, 6), b(8, 6);
    a(1, 2) = 2.5;
    a(7, 5) = 10.5;
    b.fill(21.1);
    const auto sum = a + b;
    std::cout << sum << std::endl;
}

TEST_CASE("Question #3.1") {
    execute_test("question_3_test_1.in", test_1);
}