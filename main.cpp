//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include "common/mnist_reader.hpp"
#include <assert.h>
#include "kernel.h"


int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    assert(dataset.test_images.size() > 0); // Sanity check

    // Each image should have 28 * 28 = 784 values/pixels
    std::cout << "Size of one training image = " << dataset.training_images[0].size() << std::endl;

    std::cout << "The first training image bitmap (without grayscale info):" << std::endl;
    for (size_t y = 0; y < 28; y++) {
        for (size_t x = 0; x < 28; x++) {
            auto c = dataset.training_images[0][y*28 + x] ? '.' : ' ';
            std::cout << (float)dataset.training_images[0][y*28 + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "size of element: " << sizeof(dataset.training_images[0][0]) << std::endl;

    assert(dataset.training_labels.size() > 0); // Sanity check
    std::cout << "Label of the first image is: " << static_cast<int>(dataset.training_labels[0]) << std::endl;

    wrapper(dataset);

    return 0;

}
