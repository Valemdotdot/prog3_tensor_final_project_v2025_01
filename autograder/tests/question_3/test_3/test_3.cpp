//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_3() {
    utec::algebra::Tensor<double, 2> t1(20, 50), t2(50, 20);
    t1.fill(1);
    t2.fill(4);
    try {
        auto result = t1 + t2;
    }
    catch(const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #3.3") {
    execute_test("question_3_test_3.in", test_3);
}