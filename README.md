# Handwritten-Digit-Recognition
A fully-connected Neural Network to recognize handwritten digits built using C++ and CUDA

# Introduction
In this project, I implemented a fully-connected Neural Network to recognize handwritten digits using C++ and CUDA. The main aim was to speedup training by parallelizing forward and backward propagation on a GPU. The Neural Network consisted of an input layer of 784 neurons, a hidden layer of 128 neurons and an output layer of 10 neurons. I implemted one kernel for forward propagation and three kernels for backward propagation. I used sigmoid as the activation function and stochastic gradient descent for updating weights. In the end, two training epochs took 5.87s and a acheived a tes accuracy of 95.1%. This marked a speedup of 2800 times because the initial sequencial code took 274 minutes compared to 5.87s that I was able to achieve.  

# Dataset
The MNIST dataset contains 60,000 training and 10,000 testing images of handwritten digits from 0-9 in black and white pixels of size 28x28.

# CUDA Optimizations
* Tiled matrix multiplication
* Better memory coalescing by accessing weight matrices in column-major order
* Utilizing shared memory
