//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    utec::algebra::Tensor<double, 3> t1(1, 5, 3);
    utec::algebra::Tensor<double, 3> t2(4, 3, 2);
    t1.fill(10);
    t2.fill(5);
    try {
        const auto r = utec::algebra::matrix_product(t1, t2);
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #7.1") {
    execute_test("question_7_test_1.in", test_1);
}