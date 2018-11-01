#ifndef NN_LAYER_CONFIG
#define NN_LAYER_CONFIG

#include <iostream>

enum NNActivationType
{
    None,
    Sigmod,
    Tanh,
    Relu,
    Leaky_Relu
};

class NNLayerConfig
{
  public:
    size_t num_unit;
    NNActivationType activation_type;
    NNLayerConfig(const size_t num, const NNActivationType a_type);
    ~NNLayerConfig();
};

#endif // NN_LAYER_CONFIG
