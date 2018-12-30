# Download and unpack googletest from git

set(GT_DONWLOAD_DIR ${PROJECT_BINARY_DIR}/googletest-download)

file(REMOVE "${GT_DONWLOAD_DIR}/CMakeCache.txt")

# Download and unpack googletest at configure time
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt.in 
    ${GT_DONWLOAD_DIR}/CMakeLists.txt
)

execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${GT_DONWLOAD_DIR}"
    RESULT_VARIABLE result
    )

if(result)
    message(FATAL_ERROR "CMake step for downloading Google_Test failed: ${result}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${GT_DONWLOAD_DIR}"
    RESULT_VARIABLE result
    )

if(result)
    message(FATAL_ERROR "CMake step for building Google_Test failed: ${result}")
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${PROJECT_BINARY_DIR}/googletest-src"
                 "${PROJECT_BINARY_DIR}/googletest-build")

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include")
endif()
