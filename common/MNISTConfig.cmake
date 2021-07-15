# Config file for the MNIST package
# It defines the following variables
#  MNIST_INCLUDE_DIR   - include directory of MNIST
#  MNIST_DATA_DIR      - directory of the actual MNIST data
#  MNIST_FOUND         - MNIST is available

set(MNIST_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/common)
set(MNIST_DATA_DIR /ece408/mnist)
set(MNIST_FOUND TRUE)
