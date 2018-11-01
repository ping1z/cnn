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
    std::vector<Matrix> w_mts;
    std::vector<Matrix> b_mts;
    std::vector<Matrix> z_mts;
    std::vector<Matrix> a_mts;
    std::vector<Matrix> dw_mts;
    std::vector<Matrix> db_mts;
    std::vector<Matrix> dz_mts;
    std::vector<Matrix> da_mts;

    // utility functions
    static Matrix calculateLayerZ(const Matrix &a, const Matrix &w, const Matrix &b);
    static Matrix activation(Matrix &mt, NNActivationType &a_type);
    static Matrix dActivation(Matrix &mt, NNActivationType &a_type);
    static double sigmod(const double val);
    static double dSigmod(const double val);
    static double tanh(const double val);
    static double dTanh(const double val);
    static double relu(const double val);
    static double dRelu(const double val);
    static double leakyRelu(const double val);
    static double dLeakyRelu(const double val);

  public:
    NeuralNet(const std::vector<NNLayerConfig> &configs);

    // api functions
    void forward(const Matrix &data_mt);
    void backprop(const Matrix &label_mt);
    double calculateCost(const Matrix &label_mt);
};

#endif // _NEURAL_NET_
