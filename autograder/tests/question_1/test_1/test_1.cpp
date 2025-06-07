//
// Created by rudri on 9/12/2020.
//
#include "catch.hpp"
#include "redirect_io.h"
#include "tensor.h"
using namespace std;

static void test_1() {
    const utec::algebra::Tensor<int, 2> t(2, 2);
    constexpr std::array<size_t, 2> a1 = {2, 2};
    const auto a2 = t.shape();
    for (size_t i = 0; i < a1.size(); ++i) {
        REQUIRE(a1[i] == a2[i]);
    }
}

TEST_CASE("Question #1.1") {
    execute_test("question_1_test_1.in", test_1);
}