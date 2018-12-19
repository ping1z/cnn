#include "matrix.h"

Matrix::Matrix(size_t num_r, size_t num_c)
{
    num_row = num_r;
    num_col = num_c;
    data = (double *)malloc(num_row * num_col * sizeof(double));
}

Matrix::Matrix(std::vector<std::vector<double>> data_set)
{
    num_col = data_set.size();
    num_row = data_set.front().size();
    data = (double *)malloc(num_row * num_col * sizeof(double));
    for (size_t i{0}; i < data_set.size(); i++)
    {
        const std::vector<double> &data_record = data_set[i];
        for (size_t j{0}; j < data_record.size(); j++)
        {
            set_value(j, i, data_record[j]);
        }
    }
}

Matrix::Matrix(const Matrix &source)
    : num_row{source.num_row}, num_col{source.num_col}
{
    data = (double *)malloc(num_row * num_col * sizeof(double));
    for (size_t i{0}; i < num_row; i++)
    {
        for (size_t j{0}; j < num_col; j++)
        {
            set_value(i, j, source.get_value(i, j));
        }
    }
}

Matrix::Matrix(Matrix &&source) noexcept : num_row{source.num_row}, num_col{source.num_col}, data{source.data}
{
    source.data = nullptr;
}

Matrix::~Matrix()
{
    delete[] data;
}

Matrix &Matrix::operator=(const Matrix &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    delete[] data;
    num_row = rhs.num_row;
    num_col = rhs.num_col;
    data = (double *)malloc(num_row * num_col * sizeof(double));
    for (size_t i{0}; i < num_row; i++)
    {
        for (size_t j{0}; j < num_col; j++)
        {
            set_value(i, j, rhs.get_value(i, j));
        }
    }
    return *this;
}

Matrix &Matrix::operator=(Matrix &&rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    delete[] data;
    num_row = rhs.num_row;
    num_col = rhs.num_col;
    data = rhs.data;
    rhs.data = nullptr;
    return *this;
}

Matrix Matrix::operator+(const Matrix &rhs)
{
    size_t num_row = rhs.get_rows();
    size_t num_col = rhs.get_cols();
    Matrix res{num_row, num_col};
    for (size_t i{0}; i < num_row; i++)
    {
        for (size_t j{0}; j < num_col; j++)
        {
            res.set_value(i, j, get_value(i, j) + rhs.get_value(i, j));
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix &rhs)
{
    size_t num_row = rhs.get_rows();
    size_t num_col = rhs.get_cols();
    Matrix res{num_row, num_col};
    for (size_t i{0}; i < num_row; i++)
    {
        for (size_t j{0}; j < num_col; j++)
        {
            res.set_value(i, j, get_value(i, j) - rhs.get_value(i, j));
        }
    }
    return res;
}

double Matrix::get_value(size_t r, size_t c) const
{
    if (r >= num_row || c >= num_col)
    {
        // throw exception
    }
    return data[r * num_col + c];
}

void Matrix::set_value(size_t r, size_t c, double value)
{
    if (data == nullptr || r >= num_row || c >= num_col)
    {
        // throw exception
    }
    data[r * num_col + c] = value;
}

void Matrix::random(double low, double hi)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rand(low, hi);
    for (size_t i{0}; i < num_row; i++)
    {
        for (size_t j{0}; j < num_col; j++)
        {
            // set_value(i, j, i + 1);
            set_value(i, j, rand(gen));
        }
    }
}

void Matrix::print() const
{
    for (size_t i{0}; i < num_row; i++)
    {
        std::cout << "[ ";
        for (size_t j{0}; j < num_col; j++)
        {
            std::cout << data[i * num_col + j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}

Matrix Matrix::dot(const Matrix &a, const Matrix &b)
{
    Matrix mt{a.get_rows(), b.get_cols()};
    for (size_t i{0}; i < a.get_rows(); i++)
    {
        for (size_t j{0}; j < a.get_cols(); j++)
        {
            double a_val = a.get_value(i, j);
            for (size_t k{0}; k < b.get_cols(); k++)
            {
                double b_val = b.get_value(j, k);
                double cur_val = j == 0 ? 0.0 : mt.get_value(i, k);
                mt.set_value(i, k, a_val * b_val + cur_val);
            }
        }
    }
    return mt;
}

Matrix Matrix::divide(const Matrix &a, double v)
{
    Matrix mt{a.get_rows(), a.get_cols()};
    for (size_t i{0}; i < a.get_rows(); i++)
    {
        for (size_t j{0}; j < a.get_cols(); j++)
        {
            double a_val = a.get_value(i, j);
            mt.set_value(i, j, a_val / v);
        }
    }
    return mt;
}

double Matrix::sum(const Matrix &a)
{
    double r{0.0};
    for (size_t i{0}; i < a.get_rows(); i++)
    {
        for (size_t j{0}; j < a.get_cols(); j++)
        {
            r += a.get_value(i, j);
        }
    }
    return r;
}

Matrix Matrix::sum(const Matrix &a, double axis)
{
    // check axis is 0 / 1
    if (axis == 1)
    {
        Matrix mt{a.get_rows(), 1};
        for (size_t i{0}; i < a.get_rows(); i++)
        {
            double sum{0.0};
            for (size_t j{0}; j < a.get_cols(); j++)
            {
                sum += a.get_value(i, j);
            }
            mt.set_value(i, 0, sum);
        }
        return mt;
    }
    else
    {
        Matrix mt{1, a.get_cols()};
        for (size_t i{0}; i < a.get_cols(); i++)
        {
            double sum{0.0};
            for (size_t j{0}; j < a.get_rows(); j++)
            {
                sum += a.get_value(i, j);
            }
            mt.set_value(0, i, sum);
        }
        return mt;
    }
}

Matrix Matrix::product(const Matrix &a, const double b)
{
    Matrix m{a.get_rows(), a.get_cols()};
    for (size_t i{0}; i < m.get_rows(); i++)
    {
        for (size_t j{0}; j < m.get_cols(); j++)
        {
            m.set_value(i, j, a.get_value(i, j) * b);
        }
    }
    return m;
}

Matrix Matrix::product(const Matrix &a, const Matrix &b)
{
    // check a and b have same dimension
    Matrix m{a.get_rows(), a.get_cols()};
    for (size_t i{0}; i < m.get_rows(); i++)
    {
        for (size_t j{0}; j < m.get_cols(); j++)
        {
            m.set_value(i, j, a.get_value(i, j) * b.get_value(i, j));
        }
    }
    return m;
}

Matrix Matrix::transpose(const Matrix &m)
{
    Matrix tm{m.get_cols(), m.get_rows()};
    for (size_t i{0}; i < m.get_rows(); i++)
    {
        for (size_t j{0}; j < m.get_cols(); j++)
        {
            tm.set_value(j, i, m.get_value(i, j));
        }
    }
    return tm;
}
