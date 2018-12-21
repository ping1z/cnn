#include "neural_net.h"

NeuralNet::NeuralNet(const std::vector<NNLayerConfig> &configs, const double l_rate, const double r_rate, const double m_rate)
: learning_rate {l_rate}, regularization_rate {r_rate}, momentum_rate {m_rate}
{
    layer_configs = configs;
    num_layers = layer_configs.size();
    w_mts.reserve(num_layers);
    b_mts.reserve(num_layers);
    w_mts.emplace_back(0, 0);
    b_mts.emplace_back(0, 0);

    // momentum matrix
    if (momentum_rate > 0.0) {
        vdw_mts.reserve(num_layers);
        vdb_mts.reserve(num_layers);
        vdw_mts.emplace_back(0, 0);
        vdb_mts.emplace_back(0, 0);
    }

    for (size_t i{1}; i < num_layers; i++)
    {
        w_mts.emplace_back(layer_configs[i].num_unit, layer_configs[i - 1].num_unit);
        w_mts.back().random(-0.5, 0.5);
        b_mts.emplace_back(layer_configs[i].num_unit, 1);
        b_mts.back().random(-0.0, 0.0);

        std::cout << "Layer " << i << ": " << std::endl;
        std::cout << "Weight MT:" << std::endl;
        w_mts.back().print();
        std::cout << std::endl;
        std::cout << "Bias MT:" << std::endl;
        b_mts.back().print();
        std::cout << std::endl;

        if (momentum_rate > 0.0) {
            vdw_mts.emplace_back(layer_configs[i].num_unit, layer_configs[i - 1].num_unit, 0.0);
            vdb_mts.emplace_back(layer_configs[i].num_unit, 1, 0.0);

            std::cout << "V - dWeight MT:" << std::endl;
            vdw_mts.back().print();
            std::cout << std::endl;

            std::cout << "V - dBias MT:" << std::endl;
            vdb_mts.back().print();
            std::cout << std::endl;

        }
    }
}

void NeuralNet::forward(const Matrix &data_mt)
{
    a_mts.clear();
    a_mts.reserve(num_layers);
    a_mts.emplace_back(data_mt);
    z_mts.clear();
    z_mts.reserve(num_layers);

    for (size_t i{1}; i < num_layers; i++)
    {
        z_mts.push_back(calculateLayerZ(w_mts[i], a_mts[i - 1], b_mts[i]));
        a_mts.push_back(activation(z_mts.back(), layer_configs[i].activation_type));
    }
}

void NeuralNet::backprop(const Matrix &label_mt)
{
    size_t num_sample{label_mt.get_cols()};
    dz_mts.clear();
    dz_mts.reserve(num_layers);
    dw_mts.clear();
    dw_mts.reserve(num_layers);
    db_mts.clear();
    db_mts.reserve(num_layers);

    // output layer
    Matrix &a_mt = a_mts.back();
    Matrix dz_mt{a_mt.get_rows(), a_mt.get_cols()};
    // dz = a - y
    for (size_t i{0}; i < dz_mt.get_rows(); i++)
    {
        for (size_t j{0}; j < dz_mt.get_cols(); j++)
        {
            // assume output layer is using sigmod activation
            dz_mt.set_value(i, j, a_mt.get_value(i, j) - label_mt.get_value(i, j));
        }
    }
    dz_mts.push_back(dz_mt);

    // dw =  1/m * dz * a[num_layers - 1].T
    dw_mts.push_back(Matrix::divide(Matrix::dot(dz_mt, Matrix::transpose(a_mts[num_layers - 2])), num_sample));

    // db = 1 / m * dz
    db_mts.push_back(Matrix::divide(Matrix::sum(dz_mt, 1), num_sample));

    // hidden layer
    for (size_t i{num_layers - 1}; i-- > 1;)
    {
        Matrix &dz_p_mt = dz_mts.back();
        Matrix &w_p_mt = w_mts.at(i + 1);
        Matrix &a_mt = a_mts.at(i);
        Matrix da_mt = dActivation(a_mt, layer_configs[i].activation_type);
        // dz
        dz_mts.push_back(Matrix::product(Matrix::dot(Matrix::transpose(w_p_mt), dz_p_mt), da_mt));

        Matrix &dz_mt = dz_mts.back();
        // dw
        dw_mts.push_back(Matrix::divide(Matrix::dot(dz_mt, Matrix::transpose(a_mts[i - 1])), num_sample));
        // db
        db_mts.push_back(Matrix::divide(Matrix::sum(dz_mt, 1), num_sample));
    }

    // update weight and bias
    for (size_t i{1}; i < num_layers; i++)
    {
        Matrix &w = w_mts[i];
        Matrix &b = b_mts[i];
        Matrix &dw = dw_mts.back();
        Matrix &db = db_mts.back();

        if (regularization_rate > 0.0) {
            regularization_weight(dw, w, regularization_rate, num_sample);
        }

        if (momentum_rate > 0.0) {
            Matrix &vdw = vdw_mts[i];
            vdw = Matrix::product(vdw, momentum_rate) + Matrix::product(dw, 1.0 - momentum_rate);
            Matrix &vdb = vdb_mts[i];
            vdb = Matrix::product(vdb, momentum_rate) + Matrix::product(db, 1.0 - momentum_rate);
            w = w - Matrix::product(vdw, learning_rate);
            b = b - Matrix::product(vdb, learning_rate);
        }else{
            w = w - Matrix::product(dw_mts.back(), learning_rate);
            b = b - Matrix::product(db_mts.back(), learning_rate);
        }
        dw_mts.pop_back();
        db_mts.pop_back();
    }
}

double NeuralNet::calculateCost(const Matrix &label_mt)
{
    Matrix &a_mt = a_mts.back();
    double loss{0.0};
    // only for consider binary classification
    for (size_t i{0}; i < a_mt.get_cols(); i++)
    {
        double a_y = a_mt.get_value(0, i);
        double l_y = label_mt.get_value(0, i);
        // -100.0 handles log(0) = -Infinity
        if (l_y == 0)
        {
            loss -= a_y == 1.0 ? -100.0 : log(1 - a_y);
        }
        else
        {
            loss -= a_y == 0.0 ? -100.0 : log(a_y);
        }
    }

    // if regularization is enabled 
    // if (regularization_rate > 0.0) {
    //     double sqrt_norm = 0.0;
    //     for(size_t i {1}; i < w_mts.size(); i++) {
    //         sqrt_norm += Matrix::sqrt_norm(w_mts[i]);
    //     }
    //     loss += sqrt_norm * regularization_rate / 2.0;
    // }

    return loss / label_mt.get_cols();
}

/****************************
 * static utility functions
 *
 ****************************/

Matrix NeuralNet::calculateLayerZ(const Matrix &w, const Matrix &a, const Matrix &b)
{
    Matrix mt{w.get_rows(), a.get_cols()};
    for (size_t i{0}; i < w.get_rows(); i++)
    {
        for (size_t j{0}; j < w.get_cols(); j++)
        {
            double w_val = w.get_value(i, j);
            for (size_t k{0}; k < a.get_cols(); k++)
            {
                double a_val = a.get_value(j, k);
                double cur_val = j == 0 ? b.get_value(i, 0) : mt.get_value(i, k);
                mt.set_value(i, k, a_val * w_val + cur_val);
            }
        }
    }
    return mt;
}

Matrix NeuralNet::activation(Matrix &z_mt, NNActivationType &a_type)
{
    Matrix mt{z_mt.get_rows(), z_mt.get_cols()};
    for (size_t i{0}; i < mt.get_rows(); i++)
    {
        for (size_t j{0}; j < mt.get_cols(); j++)
        {
            double val = z_mt.get_value(i, j);
            switch (a_type)
            {
            case NNActivationType::Sigmod:
                val = sigmod(val);
                break;
            case NNActivationType::Tanh:
                val = tanh(val);
                break;
            case NNActivationType::Relu:
                val = relu(val);
                break;
            case NNActivationType::Leaky_Relu:
                val = leakyRelu(val);
                break;
            default:
                break;
            }
            mt.set_value(i, j, val);
        }
    }
    return mt;
}

Matrix NeuralNet::dActivation(Matrix &z_mt, NNActivationType &a_type)
{
    Matrix mt{z_mt.get_rows(), z_mt.get_cols()};
    for (size_t i{0}; i < mt.get_rows(); i++)
    {
        for (size_t j{0}; j < mt.get_cols(); j++)
        {
            double val = z_mt.get_value(i, j);
            switch (a_type)
            {
            case NNActivationType::Sigmod:
                val = dSigmod(val);
                break;
            case NNActivationType::Tanh:
                val = dTanh(val);
                break;
            case NNActivationType::Relu:
                val = dRelu(val);
                break;
            case NNActivationType::Leaky_Relu:
                val = dLeakyRelu(val);
                break;
            default:
                break;
            }
            mt.set_value(i, j, val);
        }
    }
    return mt;
}

void NeuralNet::regularization_weight(Matrix &dw, Matrix &w, const double r_rate, const size_t m) {
    for (size_t i{0}; i < dw.get_rows(); i++)
    {
        for (size_t j{0}; j < dw.get_cols(); j++)
        {
            double dw_v = dw.get_value(i, j);
            double w_v = w.get_value(i, j);
            dw.set_value(i, j, dw_v + r_rate / m * w_v);
        }
    }
}

double NeuralNet::sigmod(const double val)
{
    return 1.0 / (1.0 + exp(-val));
}

double NeuralNet::dSigmod(const double val)
{
    return val * (1.0 - val);
}

double NeuralNet::tanh(const double val)
{
    double a = exp(val);
    double b = exp(-val);
    return (a - b) / (a + b);
}

double NeuralNet::dTanh(const double val)
{
    return 1 - val * val;
}

double NeuralNet::relu(const double val)
{
    return val > 0.0 ? val : 0.0;
}

double NeuralNet::dRelu(const double val)
{
    return val > 0.0 ? 1 : 0.0;
}

double NeuralNet::leakyRelu(const double val)
{
    return val > 0.0 ? val : 0.01 * val;
}

double NeuralNet::dLeakyRelu(const double val)
{
    return val > 0.0 ? 1 : 0.01;
}
