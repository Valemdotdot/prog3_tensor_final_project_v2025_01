#include <iostream>
#include <cassert>
#include "utec/algebra/Tensor.h"

using namespace utec::algebra;

void test_case_1() {
    Tensor<int, 2> t(2, 3);
    t.fill(7);
    int x = t(1, 2);
    assert(x == 7);
    std::cout << "Caso 1 OK\n";
}

void test_case_2() {
    Tensor<int, 2> t2(2, 3);
    t2(1, 2) = 42;
    t2.reshape({3, 2});
    int y = t2({2, 1});
    assert(y == 42);
    std::cout << "Caso 2 OK\n";
}

void test_case_3() {
    bool exception_thrown = false;
    try {
        Tensor<int, 3> t3(2, 2, 2);
        t3.reshape({2, 4, 1}); // 2*2*2=8 vs 2*4*1=8 => válido
        t3.reshape({3, 3, 1}); // 9 != 8 => debe lanzar excepción
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "Caso 3 OK\n";
}

void test_case_4() {
    Tensor<double, 2> a(2, 2), b(2, 2);
    a(0, 1) = 5.5;
    b.fill(2.0);
    auto sum = a + b;
    auto diff = sum - b;
    assert(sum(0, 1) == 7.5);
    assert(diff(0, 1) == 5.5);
    std::cout << "Caso 4 OK\n";
}

void test_case_5() {
    Tensor<float, 1> v(3);
    v.fill(2.0f);
    auto scaled = v * 4.0f;
    assert(scaled(2) == 8.0f);

    Tensor<int, 3> cube(2, 2, 2);
    cube.fill(1);
    auto cube2 = cube * cube;
    assert(cube2(1, 1, 1) == 1);
    std::cout << "Caso 5 OK\n";
}

void test_case_6() {
    Tensor<int, 2> m(2, 1);
    m(0, 0) = 3;
    m(1, 0) = 4;

    Tensor<int, 2> n(2, 3);
    n.fill(5);

    auto p = m * n;
    auto expected_shape = std::array<size_t, 2>{2, 3};
    assert(p.shape()[0] == expected_shape[0] && p.shape()[1] == expected_shape[1]);
    assert(p(0, 2) == 15);
    assert(p(1, 1) == 20);
    std::cout << "Caso 6 OK\n";
}

void test_case_7() {
    Tensor<int, 2> m2(2, 3);
    m2(1, 0) = 99;
    auto mt = m2.transpose_2d();
    auto expected_shape = std::array<size_t, 2>{3, 2};
    assert(mt.shape()[0] == expected_shape[0] && mt.shape()[1] == expected_shape[1]);
    assert(mt(0, 1) == 99);
    std::cout << "Caso 7 OK\n";
}

int main() {
    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();
    test_case_5();
    test_case_6();
    test_case_7();
    std::cout << "Todos los tests pasaron correctamente ✅\n";
    return 0;
}