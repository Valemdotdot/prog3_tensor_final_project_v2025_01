//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<int, 3> t(2, 2, 3);
    t.fill(117);
    std::cout << t;
}

TEST_CASE("Question #1.3") {
    execute_test("question_1_test_3.in", test_3);
}