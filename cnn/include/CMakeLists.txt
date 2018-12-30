target_sources(cnn
    PRIVATE
        matrix.h
        nn_layer_config.h
    PUBLIC
        neural_net.h
)

target_include_directories(cnn PUBLIC ${CMAKE_CURRENT_LIST_DIR})
