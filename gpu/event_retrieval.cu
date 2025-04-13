// event_retrieval.cu
//
// This file demonstrates an event-driven retrieval approach on GPUs using CUDA streams.
// Each query event (e.g., a retrieval request for a query node) is handled in its own stream,
// launching an asynchronous kernel that retrieves the first neighbor (as an example).
//
// Compile with:
//   nvcc event_retrieval.cu -o event_retrieval
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Helper macro for checking CUDA errors.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: event_retrieve_kernel
//
// Each instance of this kernel processes a single query event.
// It takes as input a query node (e.g., the index of the node to be queried)
// along with the graph in CSR format (device pointers d_indptr and d_indices).
// For demonstration purposes, it simply retrieves the first neighbor (if any)
// and writes it to the output pointer d_result.
// ---------------------------------------------------------------------------
__global__ void event_retrieve_kernel(int query_node,
                                        const int* d_indptr,
                                        const int* d_indices,
                                        int* d_result) {
    // This kernel is intended to be launched with a single thread.
    // Retrieve the start and end pointers for the neighbors of query_node.
    int start = d_indptr[query_node];
    int end   = d_indptr[query_node + 1];
    
    // If there is at least one neighbor, return the first neighbor.
    // Otherwise, return -1.
    if (start < end) {
        *d_result = d_indices[start];
    } else {
        *d_result = -1;
    }
}

int main() {
    // -----------------------------------------------------------------------
    // For demonstration, we create a dummy CSR graph with 4 nodes.
    // Assume the graph is as follows:
    //   - Node 0 neighbors: [1, 2]
    //   - Node 1 neighbors: [0, 2]
    //   - Node 2 neighbors: [1, 3]
    //   - Node 3 neighbors: [0, 2]
    //
    // The CSR representation then consists of:
    //   indptr:  [0, 2, 4, 6, 8]  (size num_nodes + 1)
    //   indices: [1,2, 0,2, 1,3, 0,2]  (concatenated neighbor lists)
    // -----------------------------------------------------------------------
    const int num_nodes = 4;
    int h_indptr[]   = {0, 2, 4, 6, 8};
    int h_indices[]  = {1, 2, 0, 2, 1, 3, 0, 2};
    const int num_edges = 8; // Total number of entries in h_indices

    // Allocate and copy the CSR arrays to device memory.
    int *d_indptr = nullptr, *d_indices = nullptr;
    size_t indptr_size = (num_nodes + 1) * sizeof(int);
    size_t indices_size = num_edges * sizeof(int);
    
    CUDA_CHECK(cudaMalloc((void**)&d_indptr, indptr_size));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, indices_size));
    
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr, indptr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, indices_size, cudaMemcpyHostToDevice));
    
    // -----------------------------------------------------------------------
    // Define a set of query events.
    // For this example, we will simply issue one retrieval per node.
    // -----------------------------------------------------------------------
    const int num_queries = 4;
    int h_queries[num_queries] = {0, 1, 2, 3}; // Query each node.
    
    // Arrays to hold the results on the host.
    int h_results[num_queries] = {0};
    
    // Create arrays to hold device pointers for results and streams for each query.
    int* d_results[num_queries];       // Each query result is stored in its own device memory.
    cudaStream_t streams[num_queries];   // One CUDA stream per event retrieval.
    
    // Launch an asynchronous kernel for each query event in its own stream.
    for (int i = 0; i < num_queries; ++i) {
        // Create a stream for this retrieval event.
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        
        // Allocate device memory for the result of this query.
        CUDA_CHECK(cudaMalloc((void**)&d_results[i], sizeof(int)));
        
        // Launch the event retrieval kernel in the created stream.
        // Here we launch one thread (a single query event).
        event_retrieve_kernel<<<1, 1, 0, streams[i]>>>(h_queries[i], d_indptr, d_indices, d_results[i]);
    }
    
    // Synchronize all streams and copy the results back to host.
    for (int i = 0; i < num_queries; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaMemcpy(&h_results[i], d_results[i], sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    // Output the retrieval results.
    std::cout << "Event-driven retrieval results (first neighbor per query):" << std::endl;
    for (int i = 0; i < num_queries; ++i) {
        std::cout << "Query node " << h_queries[i] << " -> first neighbor: " << h_results[i] << std::endl;
    }
    
    // Clean up: Free device memory and destroy CUDA streams.
    for (int i = 0; i < num_queries; ++i) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));
    
    return 0;
}
