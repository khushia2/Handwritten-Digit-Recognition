#ifndef KERNEL_H_
#define KERNEL_H_

#include "common/mnist_reader.hpp"

void wrapper(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset);

#endif
