//
// Created by Romina Valeria on 7/06/25.
//

#ifndef TENSOR_H
#define TENSOR_H

#pragma once
#include <vector>
#include <array>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace utec::algebra {

template <typename T, size_t Rank>
class Tensor {
private:
    std::array<size_t, Rank> _shape;
    std::vector<T> _data;

    size_t compute_index(const std::array<size_t, Rank>& indices) const {
        size_t idx = 0;
        size_t stride = 1;
        for (size_t i = Rank; i-- > 0;) {
            idx += indices[i] * stride;
            stride *= _shape[i];
        }
        return idx;
    }

public:
    Tensor(const std::array<size_t, Rank>& shape) : _shape(shape) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        _data.resize(total);
    }

    template <typename... Dims>
    Tensor(Dims... dims) : _shape{static_cast<size_t>(dims)...} {
        static_assert(sizeof...(Dims) == Rank, "Número incorrecto de dimensiones");
        size_t total = 1;
        ((total *= dims), ...);
        _data.resize(total);
    }

    template <typename... Idxs>
    T& operator()(Idxs... idxs) {
        static_assert(sizeof...(Idxs) == Rank, "Número incorrecto de índices");
        std::array<size_t, Rank> indices{static_cast<size_t>(idxs)...};
        return _data[compute_index(indices)];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... idxs) const {
        static_assert(sizeof...(Idxs) == Rank, "Número incorrecto de índices");
        std::array<size_t, Rank> indices{static_cast<size_t>(idxs)...};
        return _data[compute_index(indices)];
    }

    // Acceso con std::array (interno)
    T& operator()(const std::array<size_t, Rank>& idxs) {
        return _data[compute_index(idxs)];
    }

    const T& operator()(const std::array<size_t, Rank>& idxs) const {
        return _data[compute_index(idxs)];
    }

    const std::array<size_t, Rank>& shape() const noexcept {
        return _shape;
    }

    void fill(const T& value) noexcept {
        std::fill(_data.begin(), _data.end(), value);
    }

    size_t size() const noexcept {
        return _data.size();
    }

    Tensor operator+(const Tensor& other) const {
        if (_shape != other._shape)
            throw std::invalid_argument("Shapes incompatibles para suma");
        Tensor result(_shape);
        for (size_t i = 0; i < _data.size(); ++i)
            result._data[i] = _data[i] + other._data[i];
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (_shape != other._shape)
            throw std::invalid_argument("Shapes incompatibles para resta");
        Tensor result(_shape);
        for (size_t i = 0; i < _data.size(); ++i)
            result._data[i] = _data[i] - other._data[i];
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result(_shape);
        for (size_t i = 0; i < _data.size(); ++i)
            result._data[i] = _data[i] * scalar;
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        std::array<size_t, Rank> result_shape;

        for (size_t i = 0; i < Rank; ++i) {
            if (_shape[i] == other._shape[i])
                result_shape[i] = _shape[i];
            else if (_shape[i] == 1)
                result_shape[i] = other._shape[i];
            else if (other._shape[i] == 1)
                result_shape[i] = _shape[i];
            else
                throw std::invalid_argument("Shapes incompatibles para broadcasting");
        }

        Tensor result(result_shape);
        std::array<size_t, Rank> idx{};
        size_t total = result.size();

        for (size_t flat = 0; flat < total; ++flat) {
            // reconstruir índice multidimensional
            size_t rem = flat;
            for (int i = Rank - 1; i >= 0; --i) {
                idx[i] = rem % result_shape[i];
                rem /= result_shape[i];
            }

            std::array<size_t, Rank> idx_a, idx_b;
            for (size_t i = 0; i < Rank; ++i) {
                idx_a[i] = (_shape[i] == 1) ? 0 : idx[i];
                idx_b[i] = (other._shape[i] == 1) ? 0 : idx[i];
            }

            result._data[flat] = this->operator()(idx_a) * other(idx_b);
        }

        return result;
    }

    void reshape(const std::array<size_t, Rank>& new_shape) {
        size_t new_total = 1;
        for (auto d : new_shape) new_total *= d;
        if (new_total != _data.size())
            throw std::invalid_argument("Nuevo shape incompatible con cantidad de elementos");
        _shape = new_shape;
    }

    Tensor transpose_2d() const {
        static_assert(Rank == 2, "transpose_2d solo válido para Rank 2");
        size_t rows = _shape[0];
        size_t cols = _shape[1];
        Tensor result(std::array<size_t, 2>{cols, rows});
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }
};

}


#endif //TENSOR_H
