//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_2() {
    utec::algebra::Tensor<int, 3> t1(3, 4, 5), t2(3, 5, 4), t3(4, 3, 5);
    std::iota(t1.begin(), t1.end(), 5);
    std::iota(t2.begin(), t2.end(), 10);
    std::iota(t3.begin(), t3.end(), 100);
    try {
        const auto r = t1 * t2 + t3;
        std::cout << r;
    }
    catch (const exception& e) {
        std::cout << e.what();
    }
}

TEST_CASE("Question #4.2") {
    execute_test("question_4_test_2.in", test_2);
}