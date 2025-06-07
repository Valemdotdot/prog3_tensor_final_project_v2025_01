//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<int, 2> t(6, 2);
    try {
        t = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11
        };
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #1.4") {
    execute_test("question_1_test_4.in", test_4);
}