//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 1, 5);
    std::iota(t1.begin(), t1.end(), 100);
    std::iota(t2.begin(), t2.end(), 10);
    const auto rest = t1 - t2;
    std::cout << rest;
}

TEST_CASE("Question #5.3") {
    execute_test("question_5_test_3.in", test_3);
}