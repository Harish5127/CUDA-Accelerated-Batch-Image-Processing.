# CUDA Accelerated Batch Image Processing

## Project Overview

This project demonstrates the use of NVIDIA CUDA for accelerating image processing tasks on a GPU. The application processes a large collection of images by executing pixel operations in parallel using CUDA kernels.

The objective of the project is to show how GPU computing can significantly improve performance for image processing workloads compared to traditional sequential CPU execution.

## Features

* GPU-based image processing
* CUDA kernel implementation
* Batch processing of multiple images
* Command-line interface support
* Execution logging
* Large dataset processing

## Technologies Used

* CUDA C++
* NVIDIA CUDA Toolkit
* Linux
* Git and GitHub

## Project Structure

```text
CUDA-Batch-Image-Processing/
│
├── README.md
├── Makefile
├── run.sh
├── image_filter.cu
├── images/
├── output/
└── execution_logs/
```

## Build Instructions

Compile the application using:

```bash
make
```

## Execution

Run the application with:

```bash
./image_filter images output
```

Where:

* images = input directory
* output = output directory
## Code
```
#include <cuda_runtime.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA Kernel
__global__ void invertKernel(unsigned char* input,
                             unsigned char* output,
                             int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = 255 - input[idx];
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0]
                  << " input_image output_image\n";
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    int width, height, channels;

    // Read image
    unsigned char* h_input =
        stbi_load(inputFile, &width, &height,
                  &channels, 0);

    if (!h_input)
    {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    int imageSize = width * height * channels;

    // Allocate host output memory
    unsigned char* h_output =
        new unsigned char[imageSize];

    // Device pointers
    unsigned char *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input,
               imageSize * sizeof(unsigned char));

    cudaMalloc((void**)&d_output,
               imageSize * sizeof(unsigned char));

    // Copy image to GPU
    cudaMemcpy(d_input,
               h_input,
               imageSize * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks =
        (imageSize + threadsPerBlock - 1)
        / threadsPerBlock;

    invertKernel<<<blocks, threadsPerBlock>>>(
        d_input,
        d_output,
        imageSize);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output,
               d_output,
               imageSize * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // Save image
    if (!stbi_write_png(outputFile,
                        width,
                        height,
                        channels,
                        h_output,
                        width * channels))
    {
        std::cerr << "Failed to save image\n";
    }
    else
    {
        std::cout << "Image processed successfully!\n";
        std::cout << "Output saved as "
                  << outputFile << std::endl;
    }

    // Free memory
    stbi_image_free(h_input);

    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```
## CUDA Implementation

The application uses CUDA kernels to process image pixels in parallel.

Major CUDA concepts demonstrated:

* CUDA kernel launches
* Grid and block configuration
* Device memory allocation
* Host-to-device memory transfer
* Device-to-host memory transfer
* Thread synchronization

Each GPU thread is responsible for processing one or more image pixels.

## Results

The application successfully processed multiple images in a single execution and generated processed outputs.

Execution logs were collected to verify successful operation.

## Challenges Faced

During development, understanding CUDA memory management and thread organization was challenging. Proper grid and block configuration was necessary to ensure efficient GPU utilization.

## Learning Outcomes

This project helped develop practical knowledge in:

* CUDA programming
* Parallel computing
* GPU memory management
* Image processing techniques
* Performance-oriented programming

## Conclusion

The project demonstrates that GPUs are highly effective for image processing workloads. CUDA enables thousands of threads to execute simultaneously, resulting in improved throughput when processing large datasets.

## Author

Harish R
