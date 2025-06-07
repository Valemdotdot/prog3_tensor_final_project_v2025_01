//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    utec::algebra::Tensor<int, 2> t1(3, 4);
    t1.fill(1);
    utec::algebra::Tensor<int, 2> t2(3, 1);
    t2.fill(5);
    const auto r = t1 * t2;
    std::cout << r;
}

TEST_CASE("Question #5.1") {
    execute_test("question_5_test_1.in", test_1);
}