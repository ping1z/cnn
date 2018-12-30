project(cnn_test)

message(PROJECT_BINARY_DIR ${PROJECT_BINARY_DIR})

include(CTest)

include(GoogleTest.cmake)

add_executable(cnn_test "")

add_subdirectory(src)

target_link_libraries(cnn_test gtest gmock_main)

add_test(NAME cnn_test COMMAND cnn_test)
