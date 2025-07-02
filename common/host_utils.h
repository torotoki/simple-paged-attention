#pragma once

#include <random>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

template <typename T>
class Matrix {
 public:
    const float eps = 0.0001;
    vector<T> data;
    vector<T*> row_ptrs;
    int num_rows, num_cols;

    Matrix(int num_rows, int num_cols) :
      num_rows(num_rows),
      num_cols(num_cols),
      row_ptrs(num_rows),
      data(num_rows * num_cols) {
      for (int i = 0; i < num_rows; ++i) {
        row_ptrs[i] = &data[i * num_cols];
      }
    }

    T* get() {
      return data.data();
    }

    T& at(int row, int col) {
      return data[row * num_cols + col];
    }

    const T& at(int row, int col) const {
      return data[row * num_cols + col];
    }

    bool operator==(const Matrix<T>& B) const {
      if (num_rows != B.num_rows) return false;
      if (num_cols != B.num_cols) return false;

      for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
          if (abs(at(i, j) - B.at(i, j)) > eps) return false;
        }
      }
      return true;
    }

    void set_row(int row, vector<T>& row_vector) {
      assert(row < num_rows);
      assert(row_vector.size() == num_cols);
      for (size_t i = 0; i < row_vector.size(); ++i) {
        data[row * num_cols + i] = row_vector[i];
      }
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
  for (int i = 0; i < matrix.num_rows; ++i) {
    for (int j = 0; j < matrix.num_cols; ++j) {
      os << matrix.at(i, j);
      if (j < matrix.num_cols - 1) os << "\t";
    }
    os << "\n";
  }
  return os;
}

class InputGenerator {

 public:
  InputGenerator(int seed = 42)
    : seed(seed), random_generator(seed) {
  }

  template <typename T>
  vector<T> generateRandomVector(
      size_t size,
      T minimum = 0,
      T maximum = 32767
  ) {
    vector<T> result(size);
    if (std::is_same_v<T, int>) {
      std::uniform_int_distribution<> dist(minimum, maximum);

      for (size_t i = 0; i < size; ++i) {
        result[i] = dist(random_generator);
      }
    } else { 
      std::uniform_real_distribution<> dist(minimum, maximum);

      for (size_t i = 0; i < size; ++i) {
        result[i] = dist(random_generator);
      }
    }
    return result;
  }

  template <typename T>
  vector<T> generateConstantVector(size_t size) {
    vector<T> result(size, 1);
    return result;
  }
  
  template <typename T>
  Matrix<T> generateConstantMatrix(
      size_t num_rows, size_t num_cols, T value = 1.0f
  ) {
    Matrix<T> matrix(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; ++i) {
      for (size_t j = 0; j < num_cols; ++j)
        matrix.at(i, j) = value;
    }
    return matrix;
  }

  template <typename T>
  vector<T> generateSortedVector(
      size_t size,
      unsigned int min_step = 1,
      unsigned int max_step = 10
  ) {
    vector<T> result(size);
    std::uniform_int_distribution<> step_dist(min_step, max_step);

    T current = 0;
    T value_limit = std::numeric_limits<T>::max() - max_step - 10;
    for (size_t i = 0; i < size; ++i) {
      assert(current < value_limit);
      unsigned int step = step_dist(random_generator);
      current += step;
      result[i] = current;
    }

    return result;
  }

 private:
  int seed;
  std::mt19937 random_generator;
};

