//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<int, 3> t1(3, 4, 5);
    std::iota(t1.begin(), t1.end(), 10);
    const auto r = 50 + (t1 + 90)/10;
    std::cout << r;
}

TEST_CASE("Question #4.4") {
    execute_test("question_4_test_4.in", test_4);
}