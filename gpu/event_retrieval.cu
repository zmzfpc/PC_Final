// gpu/event_retrieval.cu
//
// This file demonstrates an event-driven GPU retrieval approach in C.
// Each query event retrieves the first neighbor from a dummy CSR graph.
// Compile with: nvcc -x c event_retrieval.cu -o event_retrieval

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                              \
    if ((call) != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",       \
                cudaGetErrorString(call), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                               \
    } } while(0)

// Kernel: retrieves the first neighbor of query_node.
__global__ void event_retrieve_kernel(int query_node,
                                        const int* d_indptr,
                                        const int* d_indices,
                                        int* d_result) {
    int start = d_indptr[query_node];
    int end   = d_indptr[query_node + 1];
    if(start < end) {
        *d_result = d_indices[start];
    } else {
        *d_result = -1;
    }
}

int main() {
    // Dummy CSR graph with 4 nodes.
    const int num_nodes = 4;
    int h_indptr[5]   = {0, 2, 4, 6, 8};
    int h_indices[8]  = {1, 2, 0, 2, 1, 3, 0, 2};
    const int num_edges = 8;

    int *d_indptr = NULL, *d_indices = NULL;
    size_t size_indptr = (num_nodes + 1) * sizeof(int);
    size_t size_indices = num_edges * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&d_indptr, size_indptr));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, size_indices));
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr, size_indptr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, size_indices, cudaMemcpyHostToDevice));

    const int num_queries = 4;
    int h_queries[4] = {0, 1, 2, 3};
    int* h_results = (int*) malloc(num_queries * sizeof(int));

    // Allocate arrays for device results and CUDA streams.
    int** d_results = (int**) malloc(num_queries * sizeof(int*));
    cudaStream_t* streams = (cudaStream_t*) malloc(num_queries * sizeof(cudaStream_t));

    for (int i = 0; i < num_queries; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc((void**)&d_results[i], sizeof(int)));
        event_retrieve_kernel<<<1, 1, 0, streams[i]>>>(h_queries[i], d_indptr, d_indices, d_results[i]);
    }

    for (int i = 0; i < num_queries; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaMemcpy(&h_results[i], d_results[i], sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("Event-driven retrieval results (first neighbor per query):\n");
    for (int i = 0; i < num_queries; i++) {
        printf("Query node %d -> first neighbor: %d\n", h_queries[i], h_results[i]);
    }

    for (int i = 0; i < num_queries; i++) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(d_results);
    free(streams);
    free(h_results);

    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));

    return 0;
}
