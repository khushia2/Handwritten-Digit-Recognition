cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(mnist_example LANGUAGES CXX CUDA)


find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Release")


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(MNIST PATHS common)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)


include(CTest)

add_executable(mnist_example
  main.cpp
  kernel.cu
  kernel.h
  common/mnist_reader_common.hpp
  common/mnist_reader.hpp
  common/mnist_utils.hpp
)


# CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)

target_compile_features(mnist_example PRIVATE cxx_range_for)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( mnist_example PROPERTIES
                       # CUDA_SEPARABLE_COMPILATION ON
                       CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      )

target_link_libraries(mnist_example ${CUDA_LIBRARIES})


include_directories(mnist_example
  ${PROJECT_SOURCE_DIR}/src
  ${CUDA_INCLUDE_DIRS}
  ${MNIST_INCLUDE_DIR}
)

target_compile_definitions(mnist_example PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
if(APPLE)
  # We need to add the default path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET mnist_example PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
