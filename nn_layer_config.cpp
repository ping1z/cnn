#include "nn_layer_config.h"

NNLayerConfig::NNLayerConfig(const size_t num, const NNActivationType a_type)
    : num_unit{num}, activation_type{a_type}
{
}

NNLayerConfig::~NNLayerConfig()
{
}
