// gpu/graph_loader.cu
//
// This file loads a CSR graph from two binary files into GPU memory.
// Each binary file is expected to begin with an int (the number of elements)
// followed by that many int values.
// Compile with: nvcc -x c graph_loader.cu -o graph_loader

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do {                              \
    if ((err) != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",       \
                cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                               \
    } } while(0)

// Function to load an integer array from a binary file.
// The file format: [int count][int array...]
int* load_binary_int_array(const char* filename, size_t* count) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    if(fread(count, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error reading count from file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    int* data = (int*) malloc((*count) * sizeof(int));
    if (fread(data, sizeof(int), *count, fp) != *count) {
        fprintf(stderr, "Error reading data from file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    return data;
}

// Loads the CSR graph from the two files and copies the arrays to GPU memory.
void loadGraphToGPU(const char* indptrFile, const char* indicesFile,
                    int** d_indptr, int** d_indices, size_t *numIndptr, size_t *numIndices) {
    printf("Loading CSR indptr from file: %s\n", indptrFile);
    int* h_indptr = load_binary_int_array(indptrFile, numIndptr);
    printf("CSR indptr contains %zu elements.\n", *numIndptr);

    printf("Loading CSR indices from file: %s\n", indicesFile);
    int* h_indices = load_binary_int_array(indicesFile, numIndices);
    printf("CSR indices contains %zu elements.\n", *numIndices);

    CUDA_CHECK(cudaMalloc((void**) d_indptr, (*numIndptr) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) d_indices, (*numIndices) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(*d_indptr, h_indptr, (*numIndptr) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_indices, h_indices, (*numIndices) * sizeof(int), cudaMemcpyHostToDevice));

    printf("Graph loaded into GPU memory successfully.\n");

    free(h_indptr);
    free(h_indices);
}

int main(int argc, char** argv) {
    const char* indptrFile = "csr_indptr.bin";
    const char* indicesFile = "csr_indices.bin";
    if(argc > 1) {
        indptrFile = argv[1];
    }
    if(argc > 2) {
        indicesFile = argv[2];
    }

    int* d_indptr = NULL;
    int* d_indices = NULL;
    size_t numIndptr = 0;
    size_t numIndices = 0;

    loadGraphToGPU(indptrFile, indicesFile, &d_indptr, &d_indices, &numIndptr, &numIndices);

    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));

    return 0;
}
