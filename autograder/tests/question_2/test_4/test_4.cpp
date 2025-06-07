//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<int, 2> t(6, 2);
    t = {
        100, 200, 300, 40, 500, 60,
        700, 800, 90, 1000, 110, 1200
    };
    std::cout << t << '\n';
    t.reshape(12, 1);
    std::cout << t << '\n';
    t.reshape(1, 20);
    std::cout << t;
}

TEST_CASE("Question #2.4") {
    execute_test("question_2_test_4.in", test_4);
}