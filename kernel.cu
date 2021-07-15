#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "common/mnist_reader.hpp"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "kernel.h"
#define BLOCK_SIZE 16
#define N2 128

__device__ float sigmoid(float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((float) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

__global__ void back_propagation_1(float* d_layer3, float* d_layer2, float* d_theta3,
                                  float* d_theta2, float* d_w2, int label, int n2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_theta3_shared[10];

    if (index < 10) {
        float expected = (index == label) ? 1.0 : 0.0;
        float d_layer3_temp = d_layer3[index];
        d_theta3_shared[index] = d_layer3_temp * (1 - d_layer3_temp) * (expected - d_layer3_temp);
        d_theta3[index] = d_theta3_shared[index];
    }

    __syncthreads();

    if (index < n2) {
        float sum = 0.0;
        for (int j = 0; j < 10; j++) {
            // --- Column major order
            sum += d_w2[index * 10 + j] * d_theta3_shared[j];
            // --- Row major order
            //sum += d_w2[j * n2 + index] * d_theta3[j];
        }
        float d_layer2_temp = d_layer2[index];
        d_theta2[index] = d_layer2_temp * (1 - d_layer2_temp) * sum;
    }
}

__global__ void back_propagation_2_w2(float* d_w2, float* d_delta2, float* d_theta3, float* d_layer2,
                                     float learning_rate, float momentum, int n2) {
    // --- Column major order
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float temp;
    // int idx = threadIdx.y * blockDim.x + threadIdx.x;
    // __shared__ float d_theta3_shared[10];
    // __shared__ float d_layer2_shared[N2];
    // if (idx<10)
    //   d_theta3_shared[idx] = d_theta3[idx];
    // if (idx<n2)
    //   d_layer2_shared[idx] = d_layer2[idx];
    //
    // __syncthreads();

    if (j < 10 && i < n2) {
        temp = (learning_rate * d_theta3[j] * d_layer2[i]) + (momentum * d_delta2[i * 10 + j]);
        d_delta2[i * 10 + j] = temp;
        d_w2[i * 10 + j] += temp;
    }
    // --- Row major order
    /*
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 10 && i < n2) {
        d_delta2[j * n2 + i] = (learning_rate * d_theta3[j] * d_layer2[i]) + (momentum * d_delta2[j * n2 + i]);
        d_w2[j * n2 + i] += d_delta2[j * n2 + i];
    }
    */
}

__global__ void back_propagation_2_w1(float* d_w1, float* d_delta1, float* d_theta2, float* d_layer1,
                                     float learning_rate, float momentum, int n2) {
    // --- Column major order
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float temp;
    if (j < n2 && i < 784) {
        temp = (learning_rate * d_theta2[j] * d_layer1[i]) + (momentum * d_delta1[i * n2 + j]);
        d_delta1[i * n2 + j] = temp;
        d_w1[i * n2 + j] += temp;
    }
    // --- Row major order
    /*
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n2 && i < 784) {
        d_delta1[j * 784 + i] = (learning_rate * d_theta2[j] * d_layer1[i]) + (momentum * d_delta1[j * 784 + i]);
        d_w1[j * 784 + i] += d_delta1[j * 784 + i];
    }
    */
}

__global__ void perceptron(float* d_layer1, float* d_layer2, float* d_layer3, float* d_w1,
                           float* d_w2, int n2) {
    __shared__ float x_shared[BLOCK_SIZE];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float y_val = 0.0;

    /**
    if (index < n2) {
        float y_val = 0;

        for (int i = 0; i < 784; i++) {
            y_val += d_layer1[i] * d_w1[index * 784 + i];
        }
        d_layer2[index] = 1.0 / (1.0 + exp(-1.0 * y_val));
    }
    **/

    #pragma unroll
    for (unsigned int m = 0; m < ceil(784*1.0/BLOCK_SIZE); ++m)
    {
        if ((m * BLOCK_SIZE + threadIdx.x) < 784){
          x_shared[threadIdx.x] = d_layer1[threadIdx.x + m * BLOCK_SIZE];
        } else{
          x_shared[threadIdx.x] = 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
            // --- Column-major ordering - faster lower Accuracy
            y_val += d_w1[index + (e + BLOCK_SIZE * m) * 128] * x_shared[e];
            // --- Row-major ordering - slower higher Accuracy
            //y_val += d_w1[index * 784 + (e + BLOCK_SIZE * m)] * x_shared[e];
        }
        __syncthreads();
    }

    if (index < n2) d_layer2[index] = 1.0 / (1.0 + exp(-1.0 * y_val));

    __syncthreads();

    // pass through third layer
    if (index < 10) {
        float val2 = 0.0;
        for (int i = 0; i < n2; i++) {
            // --- Column-major ordering
            val2 += d_layer2[i] * d_w2[i * 10 + index];
            // --- Row-major ordering
            //val2 += d_layer2[i] * d_w2[index * n2 + i];
        }
        val2 = 1.0 / (1.0 + exp(-1.0 * val2));
        d_layer3[index] = val2;
    }
}


void wrapper(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset) {
    int n2;
    int label;
    float learning_rate;
    float momentum;
    float* layer1;
    float* layer2;
    float* layer3;
    float* w1;
    float* w2;
    float* delta2;
    float* delta1;
    float* d_layer1;
    float* d_layer2;
    float* d_layer3;
    float* d_w1;
    float* d_w2;
    float* d_theta3;
    float* d_theta2;
    float* d_delta2;
    float* d_delta1;


    float* training_images;
    float* test_images;

    // set the number of nodes in hidden layer
    n2 = 128;
    learning_rate = 5e-2;
    momentum = 0.9;

    // Neural Network Architecture 3 layers: 784 -> n2 -> 10

    // allocate GPU memory
    cudaMalloc((void **)&d_layer1, 784 * sizeof(float));
    cudaMalloc((void **)&d_layer2, n2 * sizeof(float));
    cudaMalloc((void **)&d_layer3, 10 * sizeof(float));
    cudaMalloc((void **)&d_w1, 784 * n2 * sizeof(float));
    cudaMalloc((void **)&d_w2, n2 * 10 * sizeof(float));
    cudaMalloc((void **)&d_delta1, 784 * n2 * sizeof(float));
    cudaMalloc((void **)&d_delta2, n2 * 10 * sizeof(float));
    cudaMalloc((void **)&d_theta2, n2 * sizeof(float));
    cudaMalloc((void **)&d_theta3, 10 * sizeof(float));


    // allocate host memory

    layer1 = (float *)malloc(784 * sizeof(float));
    layer2 = (float *)malloc(n2 * sizeof(float));
    layer3 = (float *)malloc(10 * sizeof(float));
    w1 = (float *)malloc(784 * n2 * sizeof(float));
    w2 = (float *)malloc(n2 * 10 * sizeof(float));
    delta1 = (float *)malloc(784 * n2 * sizeof(float));
    delta2 = (float *)malloc(n2 * 10 * sizeof(float));
    training_images = (float *)malloc(60000 * 784 * sizeof(float));
    test_images = (float *)malloc(10000 * 784 * sizeof(float));

    // cudaHostAlloc((void **)&layer1, 784 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&layer2, n2 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&layer3, 10 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&w1, 784 * n2 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&w2, n2 * 10 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&delta1, 784 * n2 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&delta2, n2 * 10 * sizeof(float),cudaHostAllocPortable);
    // cudaHostAlloc((void **)&training_images, 60000 * 784 * sizeof(float),cudaHostAllocDefault);
    // cudaHostAlloc((void **)&test_images, 10000 * 784 * sizeof(float),cudaHostAllocPortable);

    // initialize weights to 0
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < 784; j++) {
            int sign = rand() % 2;
            /*
            // Row major ordering w1: (n2 x 784)
            w1[i * 784 + j] = (float)(rand() % 6) / 10.0;
            if (sign == 1) {
                w1[i * 784 + j] = -1 * w1[i * 784 + j];
            }
            delta1[i * 784 + j] = 0.0;
            */
            // Column major ordering w1: (784 x n2)
            w1[j * n2 + i] = (float)(rand() % 6) / 10.0;
            if (sign == 1) {
                w1[j * n2 + i] = -1 * w1[j * n2 + i];
            }
            delta1[j * n2 + i] = 0.0;
        }
    }
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < n2; j++) {
            int sign = rand() % 2;
            /*
            // Row major ordering w2: (10 x n2)
            w2[i * n2 + j] = (float)(rand() % 6) / 10.0;
            if (sign == 1) {
                w2[i * n2 + j] = -1 * w2[i * n2 + j];
            }
            delta2[i * n2 + j] = 0.0;
            */
            // Column major ordering w2: (n2 x 10)
            w2[j * 10 + i] = (float)(rand() % 6) / 10.0;
            if (sign == 1) {
                w2[j * 10 + i] = -1 * w2[j * 10 + i];
            }
            delta2[j * 10 + i] = 0.0;

        }
    }

    // convert training and test data to float
    for (int i = 0; i < 60000; i++) {
        for (int j = 0; j < 784; j++) {
            //training_images[i * 784 + j] = (float)dataset.training_images[i][j];
            training_images[i * 784 + j] = dataset.training_images[i][j] == 0 ? 0.0 : 1.0;
        }
    }
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 784; j++) {
            //test_images[i * 784 + j] = (float)dataset.test_images[i][j];
            test_images[i * 784 + j] = dataset.test_images[i][j] == 0 ? 0.0 : 1.0;
        }
    }

    // Copy necessary host memory to GPU memory
    cudaMemcpy(d_w1, w1, 784 * n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, n2 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta1, delta1, 784 * n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta2, delta2, n2 * 10 * sizeof(float), cudaMemcpyHostToDevice);


    // Set the block and grid dim
    // For forward process: one thread block of size n2 for each image. Currently just using one thread block.
    dim3 DimGridF(ceil(128*1.0/BLOCK_SIZE), 1, 1);
    dim3 DimBlockF(BLOCK_SIZE, 1, 1);

    dim3 DimGridB_1(1, 1, 1);
    dim3 DimBlockB_1(n2, 1, 1);
/*
    // Row major order weight matrices
    dim3 DimGridB_2_w2(ceil(n2/10.0), 1, 1);
    dim3 DimBlockB_2_w2(10, 10, 1);

    dim3 DimGridB_2_w1(ceil(784*1.0/BLOCK_SIZE), ceil(n2*1.0/BLOCK_SIZE), 1);
    dim3 DimBlockB_2_w1(BLOCK_SIZE, BLOCK_SIZE, 1);
*/
    // Column major order weight matrices
    dim3 DimGridB_2_w2(1, ceil(n2/10.0), 1);
    dim3 DimBlockB_2_w2(10, 10, 1);

    dim3 DimGridB_2_w1(ceil(n2*1.0/BLOCK_SIZE), ceil(784*1.0/BLOCK_SIZE), 1);
    dim3 DimBlockB_2_w1(BLOCK_SIZE, BLOCK_SIZE, 1);

    float ms;
    cudaEvent_t start,stop;
    // cudaStream_t stream1;
    // cudaStream_t stream2;
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);

    // float * d_train;
    // float * temp = d_layer1;
    // cudaMalloc((void **)&d_train, 60000 * 784 * sizeof(float));


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // cudaMemcpy(d_train, training_images, 60000 * 784 * sizeof(float), cudaMemcpyHostToDevice);


    // Begin training
    for(int epochs = 0; epochs < 2; epochs++) {
        for (int i = 0; i < dataset.training_images.size(); i++) {

            label = static_cast<int>(dataset.training_labels[i]);

            // Performing forward process
            cudaMemcpy(d_layer1, &(training_images[i * 784]), 784 * sizeof(float),  cudaMemcpyHostToDevice);
            // d_layer1 = d_train + i * 784;
            perceptron<<<DimGridF, DimBlockF>>>(d_layer1, d_layer2, d_layer3, d_w1, d_w2, n2);
            cudaDeviceSynchronize();

            /*
            // Checking layer1, layer2, layer3 values
            if (i == 0) {
                cudaMemcpy(layer1, d_layer1, 784 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(layer2, d_layer2, n2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(layer3, d_layer3, 10 * sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "printing layer1: " << std::endl;
                for (int j = 0; j < 784; j++) {
                    std::cout << layer1[j] << " ";
                }
                std::cout << std::endl;
                std::cout << "printing layer2: " << std::endl;
                for (int j = 0; j < n2; j++) {
                    std::cout << layer2[j] << " ";
                }
                std::cout << std::endl;
                std::cout << "printing layer3: " << std::endl;
                for (int j = 0; j < 10; j++) {
                    std::cout << layer3[j] << " ";
                }
                std::cout << std::endl;
            }
            */

            // Performing backpropagation (2 parts)
            // part 1
            back_propagation_1<<<DimGridB_1, DimBlockB_1>>>(d_layer3, d_layer2, d_theta3, d_theta2, d_w2, label, n2);
            cudaDeviceSynchronize();

            // part 2
            back_propagation_2_w2<<<DimGridB_2_w2, DimBlockB_2_w2>>>(d_w2, d_delta2, d_theta3, d_layer2, learning_rate, momentum, n2);
            back_propagation_2_w1<<<DimGridB_2_w1, DimBlockB_2_w1>>>(d_w1, d_delta1, d_theta2, d_layer1, learning_rate, momentum, n2);
            cudaDeviceSynchronize();

            /*
            // Checking w1 and w2 values
            if (i == 0) {
                cudaMemcpy(w1, d_w1, 784 * n2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(w2, d_w2, n2 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "printing w1: " << std::endl;
                for (int j = 0; j < n2; j++) {
                  for (int k = 0; k < 784; k++) {
                      std::cout << w1[j * 784 + k] << " ";
                  }
                  std::cout << std::endl;
                }
                std::cout << "printing w2: " << std::endl;
                for (int j = 0; j < 10; j++) {
                  for (int k = 0; k < n2; k++) {
                      std::cout << w2[j * n2 + k] << " ";
                  }
                  std::cout << std::endl;
                }
            }
            */
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GPU execution time: " << ms << std::endl;
    // d_layer1 =temp;

    // Copy weights to host
    /*
    cudaMemcpy(w1, d_w1, 784 * n2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w2, d_w2, n2 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "printing w1: " << std::endl;
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < 784; j++) {
          std::cout << w1[i * 784 + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "printing w2: " << std::endl;
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < n2; j++) {
          std::cout << w2[i * n2 + j] << " ";
      }
      std::cout << std::endl;
    }
    */

    // Testing
    int prediction = 0;
    int numCorrect = 0;
    for (int i = 0; i < dataset.test_images.size(); i++) {

        label = static_cast<int>(dataset.test_labels[i]);

        cudaMemcpy(d_layer1, &(test_images[i * 784]), 784 * sizeof(float), cudaMemcpyHostToDevice);
        perceptron<<<DimGridF, DimBlockF>>>(d_layer1, d_layer2, d_layer3, d_w1, d_w2, n2);
        cudaDeviceSynchronize();

        cudaMemcpy(layer3, d_layer3, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < 10; j++) {
            if (layer3[j] > layer3[prediction]) {
                prediction = j;
            }
        }

        if (prediction == label) {
            numCorrect++;
        }
    }
    std::cout << "numCorrect: " << numCorrect << std::endl;
    std::cout << "test size: " << dataset.test_images.size() << std::endl;
    std::cout << "Accuracy: " << numCorrect / (1.0 * dataset.test_images.size()) << std::endl;

    // free memory
    free(layer1);
    free(layer2);
    free(layer3);
    free(w1);
    free(w2);
    free(delta1);
    free(delta2);
    free(training_images);
    free(test_images);

    // cudaFreeHost(layer1);
    // cudaFreeHost(layer2);
    // cudaFreeHost(layer3);
    // cudaFreeHost(w1);
    // cudaFreeHost(w2);
    // cudaFreeHost(delta1);
    // cudaFreeHost(delta2);
    // cudaFreeHost(training_images);
    // cudaFreeHost(test_images);

    cudaFree(d_layer1);
    cudaFree(d_layer2);
    cudaFree(d_layer3);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_theta3);
    cudaFree(d_theta2);
    cudaFree(d_delta2);
    cudaFree(d_delta1);

}
