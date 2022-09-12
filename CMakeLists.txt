cmake_minimum_required(VERSION 3.23)
project(PytorchCPPTest)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH "/mnt/c/Users/folle/Downloads/libtorch-cxx11-abi-shared-with-deps-1.12.1+cpu/libtorch")
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_executable(PytorchCPPTest main.cpp)
target_link_libraries(PytorchCPPTest "${TORCH_LIBRARIES}")
