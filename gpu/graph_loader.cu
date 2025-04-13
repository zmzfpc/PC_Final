// graph_loader.cu
//
// This file loads a CSR graph from two NumPy files (csr_indptr.npy and csr_indices.npy)
// into GPU memory. It uses the cnpy library to read .npy files.
// Compile with: nvcc graph_loader.cu cnpy.cpp -o graph_loader

#include <iostream>
#include <vector>
#include <string>
#include "cnpy.h"           // Include the cnpy header to load .npy files.
#include <cuda_runtime.h>
#include <cstdlib>

// Helper macro to check CUDA errors.
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Function to load the CSR graph from .npy files and copy data to the GPU.
// Parameters:
//   indptrFile  - Filename for the 'csr_indptr.npy' file
//   indicesFile - Filename for the 'csr_indices.npy' file
//   d_indptr    - Device pointer that will hold the indptr array
//   d_indices   - Device pointer that will hold the indices array
//   numIndptr   - Number of elements in the indptr array (output)
//   numIndices  - Number of elements in the indices array (output)
void loadGraphToGPU(const std::string& indptrFile, const std::string& indicesFile,
                     int** d_indptr, int** d_indices, size_t &numIndptr, size_t &numIndices) {
    // Load the indptr array from the .npy file using cnpy.
    std::cout << "Loading CSR indptr from file: " << indptrFile << std::endl;
    cnpy::NpyArray indptr_npy = cnpy::npy_load(indptrFile);
    // Pointer to host data.
    int* h_indptr = reinterpret_cast<int*>(indptr_npy.data);
    // Compute total number of elements from the shape array.
    numIndptr = 1;
    for (size_t dim : indptr_npy.shape) {
        numIndptr *= dim;
    }
    std::cout << "CSR indptr contains " << numIndptr << " elements." << std::endl;

    // Load the indices array.
    std::cout << "Loading CSR indices from file: " << indicesFile << std::endl;
    cnpy::NpyArray indices_npy = cnpy::npy_load(indicesFile);
    int* h_indices = reinterpret_cast<int*>(indices_npy.data);
    numIndices = 1;
    for (size_t dim : indices_npy.shape) {
        numIndices *= dim;
    }
    std::cout << "CSR indices contains " << numIndices << " elements." << std::endl;

    // Allocate GPU memory for both arrays.
    CUDA_CHECK(cudaMalloc((void**)d_indptr, numIndptr * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_indices, numIndices * sizeof(int)));

    // Transfer data from host (CPU) to device (GPU).
    CUDA_CHECK(cudaMemcpy(*d_indptr, h_indptr, numIndptr * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_indices, h_indices, numIndices * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "Graph loaded into GPU memory successfully." << std::endl;
}

int main(int argc, char** argv) {
    // File paths (default filenames, can be overridden by command-line arguments)
    std::string indptrFile = "csr_indptr.npy";
    std::string indicesFile = "csr_indices.npy";
    if (argc > 1) {
        indptrFile = argv[1];
    }
    if (argc > 2) {
        indicesFile = argv[2];
    }

    // Device pointers for the CSR components.
    int* d_indptr = nullptr;
    int* d_indices = nullptr;
    size_t numIndptr = 0;
    size_t numIndices = 0;

    // Load the graph into GPU memory.
    loadGraphToGPU(indptrFile, indicesFile, &d_indptr, &d_indices, numIndptr, numIndices);

    // At this point, the CSR arrays are available on the device.
    // Insert your kernel calls or further processing below.

    // For now, we simply free the allocated GPU memory.
    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));

    return 0;
}
