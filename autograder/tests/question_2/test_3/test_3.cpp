//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<int, 2> t(6, 2);
    t = {
        111, 222, 333, 444, 555, 666,
        777, 888, 999, 111, 111, 112
    };
    std::cout << t << '\n';
    t.reshape(2, 3);
    std::cout << t;
}

TEST_CASE("Question #2.3") {
    execute_test("question_2_test_3.in", test_3);
}