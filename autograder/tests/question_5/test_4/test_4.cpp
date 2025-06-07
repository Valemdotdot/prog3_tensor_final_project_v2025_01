//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<double, 2> t1(6, 7), t2(6, 1);
    std::iota(t1.begin(), t1.end(), 10);
    t2.fill(100);
    const auto sum = t1 + t2;
    std::cout << t1 << std::endl;
    std::cout << t2 << std::endl;
    std::cout << sum << std::endl;
}

TEST_CASE("Question #5.4") {
    execute_test("question_5_test_4.in", test_4);
}