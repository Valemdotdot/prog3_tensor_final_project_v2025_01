//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    utec::algebra::Tensor<int, 2> t(6, 2);
    try {
        t = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12
        };
        t.reshape(3, 2, 2);
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #2.2") {
    execute_test("question_2_test_2.in", test_2);
}