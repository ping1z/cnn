#include <iostream>
#include "neural_net.h"

int main()
{
    std::cout << "Hello Neural Network!" << std::endl;
    std::vector<std::vector<double>> train_data{};
    train_data.push_back({0, 0});
    train_data.push_back({0, 1});
    train_data.push_back({1, 1});
    train_data.push_back({1, 0});

    std::vector<std::vector<double>> train_label{{0}, {1}, {0}, {1}};

    Matrix train_data_mt{train_data};

    Matrix train_label_mt{train_label};

    NeuralNet net{{{2, NNActivationType::None}, {4, NNActivationType::Leaky_Relu}, {1, NNActivationType::Sigmod}}, 0.1, 0.0001, 0.9};

    size_t epoch{0};
    double loss = 1.0;
    while (epoch++ < 100000 || loss >= 0.01)
    {
        net.forward(train_data_mt);
        loss = net.calculateCost(train_label_mt);
        std::cout << "Epoch: " << epoch << ", Loss: " << loss << std::endl;
        net.backprop(train_label_mt);
    }
    return 0;
}
