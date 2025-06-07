//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<int, 1> t1(30);
    std::iota(t1.begin(), t1.end(), 1);
    std::cout << t1 << '\n';
    try {
        const auto r = transpose_2d(t1);
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #6.3") {
    execute_test("question_6_test_3.in", test_3);
}