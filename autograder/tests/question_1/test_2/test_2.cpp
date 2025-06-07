//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    try {
        const utec::algebra::Tensor<int, 2> t(2, 2, 2);
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #1.2") {
    execute_test("question_1_test_2.in", test_2);
}