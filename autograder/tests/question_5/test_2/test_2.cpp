//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    utec::algebra::Tensor<int, 3> t1(3, 4, 5);
    t1.fill(11);
    utec::algebra::Tensor<int, 3> t2(1, 1, 5);
    t2.fill(51);
    const auto r = t1 * t2;
    std::cout << r;
}

TEST_CASE("Question #5.2") {
    execute_test("question_5_test_2.in", test_2);
}