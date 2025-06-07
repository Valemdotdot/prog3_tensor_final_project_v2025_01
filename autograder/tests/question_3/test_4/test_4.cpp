//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_4() {
    utec::algebra::Tensor<double, 3> t1(200, 50, 3), t2(200, 50, 3), t3(200, 50, 3);
    t1.fill(3);
    t2.fill(4);
    t3.fill(5);
    const auto r = t1 + t2 - t3;
    std::cout << std::accumulate(r.cbegin(), r.cend(), double{0});
}

TEST_CASE("Question #3.4") {
    execute_test("question_3_test_4.in", test_4);
}