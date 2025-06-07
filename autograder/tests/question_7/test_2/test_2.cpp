//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    utec::algebra::Tensor<double, 2> t1(5, 3);
    utec::algebra::Tensor<double, 2> t2(5, 2);
    t1.fill(10);
    t2.fill(5);
    try {
        const auto r = utec::algebra::matrix_product(t1, t2);
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #7.2") {
    execute_test("question_7_test_2.in", test_2);
}