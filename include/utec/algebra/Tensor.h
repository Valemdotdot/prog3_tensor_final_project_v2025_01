
#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H

#include <iostream>
#include <array>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <iomanip>

namespace utec::algebra {

template<typename T, int Rank>
class Tensor {
public:
    using ShapeType = std::array<size_t, Rank>;

    template<typename... Dims>
    explicit Tensor(Dims... dims) {
        static_assert(sizeof...(Dims) == Rank, "Number of dimensions must match Rank.");
        shape_ = ShapeType{static_cast<size_t>(dims)...};
        data_.resize(total_size(shape_));
    }


    // Constructor con array de shape
    explicit Tensor(const ShapeType& shape) : shape_(shape) {
        allocate_storage();
    }

    const ShapeType& shape() const noexcept {
        return shape_;
    }

    void reshape(const ShapeType& new_shape) {
        size_t new_total = total_size(new_shape);
        if (new_total != data_.size()) {
            throw std::invalid_argument("Total size of new shape does not match tensor size");
        }
        shape_ = new_shape;
    }

    template<typename... Dims>
    void reshape(Dims... dims) {
        static_assert(sizeof...(Dims) == Rank, "Number of arguments must match tensor rank.");
        ShapeType new_shape{static_cast<size_t>(dims)...};
        this->reshape(new_shape);
    }


    void fill(const T& value) noexcept {
        std::fill(data_.begin(), data_.end(), value);
    }

    Tensor operator+(const Tensor& other) const {
        validate_shape_match(other);
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        validate_shape_match(other);
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] - other.data_[i];
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        validate_shape_match(other);
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * other.data_[i];
        return result;
    }

    Tensor operator*(const T& scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * scalar;
        return result;
    }


    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size())
            throw std::invalid_argument("Data size does not match tensor size");
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        print_tensor(os, t.data_, t.shape_, 0, 0);
        return os;
    }
    Tensor operator+(T scalar) const {
        Tensor result = *this;
        for (auto& elem : result.data_)
            elem += scalar;
        return result;
    }

    Tensor operator-(T scalar) const {
        Tensor result = *this;
        for (auto& elem : result.data_)
            elem -= scalar;
        return result;
    }

    Tensor operator*(T scalar) const {
        Tensor result = *this;
        for (auto& elem : result.data_)
            elem *= scalar;
        return result;
    }

    Tensor operator/(T scalar) const {
        Tensor result = *this;
        for (auto& elem : result.data_)
            elem /= scalar;
        return result;
    }

    friend Tensor operator+(T scalar, const Tensor& tensor) {
        return tensor + scalar;
    }

    friend Tensor operator-(T scalar, const Tensor& tensor) {
        Tensor result = tensor;
        for (auto& elem : result.data_)
            elem = scalar - elem;
        return result;
    }

    friend Tensor operator*(T scalar, const Tensor& tensor) {
        return tensor * scalar;
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
        std::array<size_t, Rank> idx{static_cast<size_t>(indices)...};
        size_t offset = 0;
        size_t stride = 1;
        // Cálculo del offset (row-major order)
        for (int i = Rank - 1; i >= 0; --i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx[i] * stride;
            stride *= shape_[i];
        }
        return data_[offset];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
        std::array<size_t, Rank> idx{static_cast<size_t>(indices)...};
        size_t offset = 0;
        size_t stride = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
            offset += idx[i] * stride;
            stride *= shape_[i];
        }
        return data_[offset];
    }


    static void print_indent(std::ostream& os, int indent) {
        for (int i = 0; i < indent; ++i) os << "  ";
    }

    template <size_t N>
    static void print_tensor(std::ostream& os, const std::vector<T>& data, const std::array<size_t, N>& shape, int depth, size_t offset) {
        if constexpr (N == 1) {
            for (size_t i = 0; i < shape[0]; ++i) {
                os << data[offset + i] << ' ';
            }
        } else {
            size_t block = 1;
            for (size_t i = 1; i < N; ++i)
                block *= shape[i];

            os << "{\n";
            for (size_t i = 0; i < shape[0]; ++i) {
                print_tensor(os, data, slice_shape(shape), depth + 1, offset + i * block);
                os << '\n';
            }
            os << "}";
        }
    }

    auto cbegin() const noexcept { return data_.cbegin(); }
    auto cend() const noexcept { return data_.cend(); }

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }


    template <size_t N>
    static std::array<size_t, N - 1> slice_shape(const std::array<size_t, N>& shape) {
        std::array<size_t, N - 1> result{};
        for (size_t i = 1; i < N; ++i) {
            result[i - 1] = shape[i];
        }
        return result;
    }

private:
    ShapeType shape_;
    std::vector<T> data_;

    void allocate_storage() {
        data_.resize(total_size(shape_));
    }

    size_t total_size(const ShapeType& shape) const {
        return std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
    }

    void validate_shape_match(const Tensor& other) const {
        if (shape_ != other.shape_)
            throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
    }
};

} // namespace utec::algebra
// namespace utec::algebra

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_TENSOR_H