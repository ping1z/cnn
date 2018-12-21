#ifndef _NEURAL_NET_
#define _NEURAL_NET_

#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <limits>

#include "nn_layer_config.h"
#include "matrix.h"

class NeuralNet
{
  private:
    size_t num_layers{};
    std::vector<NNLayerConfig> layer_configs;
    double learning_rate;
    double regularization_rate;
    double momentum_rate;

    std::vector<Matrix> w_mts;
    std::vector<Matrix> b_mts;
    std::vector<Matrix> z_mts;
    std::vector<Matrix> a_mts;
    std::vector<Matrix> dw_mts;
    std::vector<Matrix> db_mts;
    std::vector<Matrix> dz_mts;
    std::vector<Matrix> da_mts;
    std::vector<Matrix> vdw_mts;
    std::vector<Matrix> vdb_mts;

    // utility functions
    static Matrix calculateLayerZ(const Matrix &a, const Matrix &w, const Matrix &b);
    static Matrix activation(Matrix &mt, NNActivationType &a_type);
    static Matrix dActivation(Matrix &mt, NNActivationType &a_type);
    static void regularization_weight(Matrix &dw, Matrix &w, const double r_rate, const size_t m);
    static double sigmod(const double val);
    static double dSigmod(const double val);
    static double tanh(const double val);
    static double dTanh(const double val);
    static double relu(const double val);
    static double dRelu(const double val);
    static double leakyRelu(const double val);
    static double dLeakyRelu(const double val);

  public:
    NeuralNet(const std::vector<NNLayerConfig> &configs, const double l_rate = 0.1, const double r_rate = 0.0, const double m_rate = 0.0);

    // api functions
    void forward(const Matrix &data_mt);
    void backprop(const Matrix &label_mt);
    double calculateCost(const Matrix &label_mt);
};

#endif // _NEURAL_NET_
