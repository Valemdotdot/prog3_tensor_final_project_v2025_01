//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 4, 5), t3(3, 4, 5);
    std::iota(t1.begin(), t1.end(), 5);
    std::iota(t2.begin(), t2.end(), 10);
    std::iota(t3.begin(), t3.end(), 100);
    auto r = t1 * t2 + t3;
    std::cout << r;
}

TEST_CASE("Question #4.1") {
    execute_test("question_4_test_1.in", test_1);
}