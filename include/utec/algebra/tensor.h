//
// Created by Romina Valeria on 7/06/25.
//

#ifndef TENSOR_H
#define TENSOR_H


#pragma once

#include <array>
#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>
#include <numeric>


namespace utec::algebra {

inline bool can_broadcast(const std::array<size_t, 2>& a, const std::array<size_t, 2>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i] && a[i] != 1 && b[i] != 1) return false;
  }
  return true;
}

template<size_t N>
inline bool can_broadcast(const std::array<size_t, N>& a, const std::array<size_t, N>& b) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i] && a[i] != 1 && b[i] != 1) return false;
  }
  return true;
}

inline size_t broadcast_index(size_t i, size_t dim_size) {
  return (dim_size == 1) ? 0 : i;
}

template<typename T, size_t N>
class Tensor {
 private:
  std::array<size_t, N> dimensions_{};
  std::vector<T> data_;

  size_t compute_flat_index(const std::array<size_t, N>& indices) const {
    size_t flat_index = 0;
    size_t stride = 1;
    for (size_t i = N; i-- > 0;) {
      flat_index += indices[i] * stride;
      stride *= dimensions_[i];
    }
    return flat_index;
  }

 public:
  template<typename... Dims>
  Tensor(Dims... dims) {
    if (sizeof...(Dims) != N) {
      throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N));
    }
    dimensions_ = std::array<size_t, N>{static_cast<size_t>(dims)...};
    size_t total_size = 1;
    for (auto d : dimensions_) total_size *= d;
    data_.resize(total_size);
  }

  explicit Tensor(const std::array<size_t, N>& dims) {
    dimensions_ = dims;
    size_t total_size = 1;
    for (auto d : dimensions_) total_size *= d;
    data_.resize(total_size);
  }

  Tensor() {
    dimensions_.fill(1);
    data_.resize(1);
  }

  const std::array<size_t, N>& shape() const { return dimensions_; }

  template<typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == N, "Access must use N indices");
    std::array<size_t, N> idx = {static_cast<size_t>(indices)...};
    return data_[compute_flat_index(idx)];
  }

  template<typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == N, "Access must use N indices");
    std::array<size_t, N> idx = {static_cast<size_t>(indices)...};
    return data_[compute_flat_index(idx)];
  }

  typename std::vector<T>::iterator begin() { return data_.begin(); }
  typename std::vector<T>::iterator end() { return data_.end(); }
  typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
  typename std::vector<T>::const_iterator end() const { return data_.end(); }
  typename std::vector<T>::const_iterator cbegin() const { return data_.cbegin(); }
  typename std::vector<T>::const_iterator cend() const { return data_.cend(); }

  void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

  Tensor<T, N>& operator=(std::initializer_list<T> list) {
    if (list.size() != data_.size()) throw std::runtime_error("Data size does not match tensor size");
    std::copy(list.begin(), list.end(), data_.begin());
    return *this;
  }

  template<typename... Dims>
  void reshape(Dims... new_dims) {
    if (sizeof...(Dims) != N) {
      throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N));
    }
    std::array<size_t, N> new_shape = std::array<size_t, N>{static_cast<size_t>(new_dims)...};
    size_t new_total = 1;
    for (auto d : new_shape) new_total *= d;

    if (new_total > data_.size()) {
      data_.resize(new_total, T{});
    }

    dimensions_ = new_shape;
    data_.resize(new_total);
  }

  Tensor<T, N> operator+(const Tensor<T, N>& other) const {
    if (!can_broadcast(dimensions_, other.dimensions_))
      throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");

    Tensor<T, N> result(dimensions_);

    std::array<size_t, N> indices{};
    for (size_t i = 0; i < data_.size(); ++i) {
      size_t temp = i;
      for (int d = N - 1; d >= 0; --d) {
        indices[d] = temp % dimensions_[d];
        temp /= dimensions_[d];
      }

      std::array<size_t, N> other_indices{};
      for (size_t d = 0; d < N; ++d) {
        other_indices[d] = broadcast_index(indices[d], other.dimensions_[d]);
      }

      result.data_[i] = data_[i] + other.data_[other.compute_flat_index(other_indices)];
    }
    return result;
  }

  Tensor<T, N> operator-(const Tensor<T, N>& other) const {
    if (!can_broadcast(dimensions_, other.dimensions_))
      throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");

    Tensor<T, N> result(dimensions_);

    std::array<size_t, N> indices{};
    for (size_t i = 0; i < data_.size(); ++i) {
      size_t temp = i;
      for (int d = N - 1; d >= 0; --d) {
        indices[d] = temp % dimensions_[d];
        temp /= dimensions_[d];
      }

      std::array<size_t, N> other_indices{};
      for (size_t d = 0; d < N; ++d) {
        other_indices[d] = broadcast_index(indices[d], other.dimensions_[d]);
      }

      result.data_[i] = data_[i] - other.data_[other.compute_flat_index(other_indices)];
    }
    return result;
  }

  Tensor<T, N> operator*(const Tensor<T, N>& other) const {
    if (!can_broadcast(dimensions_, other.dimensions_))
      throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");

    Tensor<T, N> result(dimensions_);

    std::array<size_t, N> indices{};
    for (size_t i = 0; i < data_.size(); ++i) {
      size_t temp = i;
      for (int d = N - 1; d >= 0; --d) {
        indices[d] = temp % dimensions_[d];
        temp /= dimensions_[d];
      }

      std::array<size_t, N> other_indices{};
      for (size_t d = 0; d < N; ++d) {
        other_indices[d] = broadcast_index(indices[d], other.dimensions_[d]);
      }

      result.data_[i] = data_[i] * other.data_[other.compute_flat_index(other_indices)];
    }
    return result;
  }

  Tensor<T, N> operator+(const T& scalar) const {
    Tensor<T, N> result = *this;
    for (auto& x : result.data_) x += scalar;
    return result;
  }

  Tensor<T, N> operator-(const T& scalar) const {
    Tensor<T, N> result = *this;
    for (auto& x : result.data_) x -= scalar;
    return result;
  }

  Tensor<T, N> operator*(const T& scalar) const {
    Tensor<T, N> result = *this;
    for (auto& x : result.data_) x *= scalar;
    return result;
  }

  Tensor<T, N> operator/(const T& scalar) const {
    Tensor<T, N> result = *this;
    for (auto& x : result.data_) x /= scalar;
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    if constexpr (N == 1) {
      os << "[";
      for (size_t i = 0; i < tensor.data_.size(); ++i) {
        os << tensor.data_[i];
        if (i + 1 < tensor.data_.size()) os << ", ";
      }
      os << "]";
    } else if constexpr (N == 2) {
      os << "{\n";
      for (size_t i = 0; i < tensor.dimensions_[0]; ++i) {
        for (size_t j = 0; j < tensor.dimensions_[1]; ++j) {
          os << tensor.data_[i * tensor.dimensions_[1] + j];
          if (j + 1 < tensor.dimensions_[1]) os << " ";
        }
        if (i + 1 < tensor.dimensions_[0]) os << "\n";
      }
      os << "\n}";
    } else if constexpr (N == 3) {
      os << "{\n";
      for (size_t i = 0; i < tensor.dimensions_[0]; ++i) {
        os << "{\n";
        for (size_t j = 0; j < tensor.dimensions_[1]; ++j) {
          for (size_t k = 0; k < tensor.dimensions_[2]; ++k) {
            size_t idx = i * tensor.dimensions_[1] * tensor.dimensions_[2] +
                        j * tensor.dimensions_[2] + k;
            os << tensor.data_[idx];
            if (k + 1 < tensor.dimensions_[2]) os << " ";
          }
          if (j + 1 < tensor.dimensions_[1]) os << "\n";
        }
        os << "\n}";
        if (i + 1 < tensor.dimensions_[0]) os << "\n";
      }
      os << "\n}";
    } else {
      os << "[";
      for (size_t i = 0; i < tensor.data_.size(); ++i) {
        os << tensor.data_[i];
        if (i + 1 < tensor.data_.size()) os << ", ";
      }
      os << "]";
    }
    return os;
  }
};

template<typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
  return tensor + scalar;
}

template<typename T, size_t N>
Tensor<T, N> operator-(const T& scalar, const Tensor<T, N>& tensor) {
  Tensor<T, N> result = tensor;
  for (auto& x : result.begin()) x = scalar - x;
  return result;
}

template<typename T, size_t N>
Tensor<T, N> operator*(const T& scalar, const Tensor<T, N>& tensor) {
  return tensor * scalar;
}

template<typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& input) {
  if constexpr (N < 2) {
    throw std::runtime_error("Cannot transpose 1D tensor: need at least 2 dimensions");
  }

  auto shape = input.shape();
  std::array<size_t, N> new_shape = shape;
  std::swap(new_shape[N - 1], new_shape[N - 2]);

  Tensor<T, N> result;
  if constexpr (N == 2) {
    result = Tensor<T, N>(new_shape[0], new_shape[1]);
  } else if constexpr (N == 3) {
    result = Tensor<T, N>(new_shape[0], new_shape[1], new_shape[2]);
  } else if constexpr (N == 4) {
    result = Tensor<T, N>(new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
  }

  std::array<size_t, N> idx{};
  std::array<size_t, N> transposed_idx{};

  size_t total_elements = 1;
  for (auto dim : shape) total_elements *= dim;

  for (size_t i = 0; i < total_elements; ++i) {
    size_t temp = i;
    for (int d = N - 1; d >= 0; --d) {
      idx[d] = temp % shape[d];
      temp /= shape[d];
    }
    transposed_idx = idx;
    std::swap(transposed_idx[N - 1], transposed_idx[N - 2]);

    if constexpr (N == 2) {
      result(transposed_idx[0], transposed_idx[1]) = input(idx[0], idx[1]);
    } else if constexpr (N == 3) {
      result(transposed_idx[0], transposed_idx[1], transposed_idx[2]) = input(idx[0], idx[1], idx[2]);
    } else if constexpr (N == 4) {
      result(transposed_idx[0], transposed_idx[1], transposed_idx[2], transposed_idx[3]) = input(idx[0], idx[1], idx[2], idx[3]);
    }
  }

  return result;
}

template<typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& A, const Tensor<T, N>& B) {
  const auto& a_shape = A.shape();
  const auto& b_shape = B.shape();

  if constexpr (N < 2) {
    throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
  }

  if (a_shape[N - 1] != b_shape[N - 2]) {
    throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
  }

  for (size_t i = 0; i < N - 2; ++i) {
    if (a_shape[i] != b_shape[i]) {
      throw std::runtime_error("Matrix dimensions are compatible for multiplication but batch dimensions do not match");
    }
  }

  std::array<size_t, N> result_shape = a_shape;
  result_shape[N - 1] = b_shape[N - 1];
  result_shape[N - 2] = a_shape[N - 2];

  Tensor<T, N> result;
  if constexpr (N == 2) {
    result = Tensor<T, N>(result_shape[0], result_shape[1]);
  } else if constexpr (N == 3) {
    result = Tensor<T, N>(result_shape[0], result_shape[1], result_shape[2]);
  } else if constexpr (N == 4) {
    result = Tensor<T, N>(result_shape[0], result_shape[1], result_shape[2], result_shape[3]);
  }

  std::array<size_t, N> idx{};
  std::array<size_t, N> a_idx{};
  std::array<size_t, N> b_idx{};

  size_t total_elements = 1;
  for (auto dim : result_shape) total_elements *= dim;

  for (size_t i = 0; i < total_elements; ++i) {
    size_t temp = i;
    for (int d = N - 1; d >= 0; --d) {
      idx[d] = temp % result_shape[d];
      temp /= result_shape[d];
    }

    T sum = 0;
    for (size_t k = 0; k < a_shape[N - 1]; ++k) {
      for (size_t d = 0; d < N; ++d) {
        a_idx[d] = idx[d];
        b_idx[d] = idx[d];
      }
      a_idx[N - 1] = k;
      b_idx[N - 2] = k;

      if constexpr (N == 2) {
        sum += A(a_idx[0], a_idx[1]) * B(b_idx[0], b_idx[1]);
      } else if constexpr (N == 3) {
        sum += A(a_idx[0], a_idx[1], a_idx[2]) * B(b_idx[0], b_idx[1], b_idx[2]);
      } else if constexpr (N == 4) {
        sum += A(a_idx[0], a_idx[1], a_idx[2], a_idx[3]) * B(b_idx[0], b_idx[1], b_idx[2], b_idx[3]);
      }
    }

    if constexpr (N == 2) {
      result(idx[0], idx[1]) = sum;
    } else if constexpr (N == 3) {
      result(idx[0], idx[1], idx[2]) = sum;
    } else if constexpr (N == 4) {
      result(idx[0], idx[1], idx[2], idx[3]) = sum;
    }
  }

  return result;
}

}

#endif //TENSOR_H
