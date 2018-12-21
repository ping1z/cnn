#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

class Matrix
{
  private:
    size_t num_row{};
    size_t num_col{};
    double *data{nullptr};

  public:
    //
    Matrix(size_t num_r, size_t num_c);
    Matrix(size_t num_r, size_t num_c, double default_value);
    Matrix(std::vector<std::vector<double>> data_set);
    Matrix(const Matrix &source);     // copy constructor
    Matrix(Matrix &&source) noexcept; // move constructor
    ~Matrix();
    // operators
    Matrix &operator=(const Matrix &rhs); // copy assignment
    Matrix &operator=(Matrix &&rhs);      // move assignment
    Matrix operator+(const Matrix &rhs);  // add assignment
    Matrix operator-(const Matrix &rhs);  // subtract assignment

    //
    size_t get_cols() const
    {
        return num_col;
    };
    size_t get_rows() const
    {
        return num_row;
    };
    double *get_data()
    {
        return data;
    };
    double get_value(size_t r, size_t c) const;
    void set_value(size_t r, size_t c, double value);
    void random(double low, double hi);
    void print() const;

    static Matrix dot(const Matrix &a, const Matrix &b);
    static Matrix divide(const Matrix &a, double v);
    static double sum(const Matrix &a);
    static Matrix sum(const Matrix &a, double axis);
    static Matrix product(const Matrix &a, const Matrix &b);
    static Matrix product(const Matrix &a, const double b);
    static Matrix transpose(const Matrix &m);
    static double sqrt_norm(const Matrix &m);
};

#endif // _MATRIX_H_
