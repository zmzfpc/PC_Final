// timestep_retrieval.cu
//
// This file demonstrates a fixed time-step retrieval approach on GPUs.
// Retrieval queries are grouped into batches and launched concurrently at fixed intervals.
// In this example, for each time-step the kernel processes a set of query nodes using a CSR graph.
//
// Compile with:
//   nvcc timestep_retrieval.cu -o timestep_retrieval
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

// Macro for CUDA error checking.
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: timestep_retrieve_kernel
//
// This kernel processes a batch of query nodes concurrently.
// Each thread handles one query event by retrieving the first neighbor from the CSR graph.
// Parameters:
//   d_query_nodes: device array of query node indices (batch)
//   num_queries: number of queries in the batch
//   d_indptr: CSR index pointer array (device)
//   d_indices: CSR neighbor indices array (device)
//   d_results: device array where each thread writes its result
// ---------------------------------------------------------------------------
__global__ void timestep_retrieve_kernel(const int* d_query_nodes, int num_queries,
                                           const int* d_indptr, const int* d_indices,
                                           int* d_results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        int query_node = d_query_nodes[tid];
        int start = d_indptr[query_node];
        int end   = d_indptr[query_node + 1];
        // For demonstration, select the first neighbor if available; otherwise, return -1.
        d_results[tid] = (start < end) ? d_indices[start] : -1;
    }
}

int main() {
    // -------------------------------------------------------------------------
    // Create a dummy CSR graph representing the following graph:
    //   Node 0: [1, 2]
    //   Node 1: [0, 2]
    //   Node 2: [1, 3]
    //   Node 3: [0, 2]
    // The CSR arrays are defined as follows:
    //   h_indptr:   [0, 2, 4, 6, 8]  (size: num_nodes + 1)
    //   h_indices:  [1,2, 0,2, 1,3, 0,2]
    // -------------------------------------------------------------------------
    const int num_nodes = 4;
    int h_indptr[]  = {0, 2, 4, 6, 8};
    int h_indices[] = {1, 2, 0, 2, 1, 3, 0, 2};
    const int num_edges = 8;

    // Allocate device memory for the CSR arrays.
    int* d_indptr = nullptr;
    int* d_indices = nullptr;
    size_t size_indptr = (num_nodes + 1) * sizeof(int);
    size_t size_indices = num_edges * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&d_indptr, size_indptr));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, size_indices));
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr, size_indptr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, size_indices, cudaMemcpyHostToDevice));
    
    // -------------------------------------------------------------------------
    // Fixed time-step parameters.
    // We simulate a fixed number of time steps (e.g., 3).
    // In each time step, a batch of queries is processed together.
    // -------------------------------------------------------------------------
    const int num_time_steps = 3;
    const int fixed_interval_ms = 1000; // Fixed interval in milliseconds (1 second)
    
    // For this example, we use 2 queries per time step.
    // Define different batches for each time step.
    std::vector<std::vector<int>> time_step_queries = {
        {0, 1},  // Time step 0: query nodes 0 and 1
        {2, 3},  // Time step 1: query nodes 2 and 3
        {0, 2}   // Time step 2: query nodes 0 and 2
    };

    // Process batches in fixed time steps.
    for (int t = 0; t < num_time_steps; ++t) {
        std::cout << "Time-step " << t << " retrieval:" << std::endl;
        
        // Extract queries for this time step.
        std::vector<int> h_query_nodes = time_step_queries[t];
        int num_queries = h_query_nodes.size();
        
        // Allocate device memory for the batch of query nodes and corresponding results.
        int* d_query_nodes = nullptr;
        int* d_results = nullptr;
        size_t size_queries = num_queries * sizeof(int);
        CUDA_CHECK(cudaMalloc((void**)&d_query_nodes, size_queries));
        CUDA_CHECK(cudaMalloc((void**)&d_results, size_queries)); // One result per query
        
        // Copy the batch of query nodes to device memory.
        CUDA_CHECK(cudaMemcpy(d_query_nodes, h_query_nodes.data(), size_queries, cudaMemcpyHostToDevice));
        
        // Launch the kernel to process the query batch.
        int threadsPerBlock = 256;
        int blocks = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
        timestep_retrieve_kernel<<<blocks, threadsPerBlock>>>(d_query_nodes, num_queries, d_indptr, d_indices, d_results);
        
        // Synchronize device execution for the current batch.
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Retrieve the results from the device.
        std::vector<int> h_results(num_queries);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, size_queries, cudaMemcpyDeviceToHost));
        
        // Output the retrieval results for this time step.
        for (int i = 0; i < num_queries; ++i) {
            std::cout << "  Query node " << h_query_nodes[i] << " -> first neighbor: " << h_results[i] << std::endl;
        }
        
        // Free the device memory used for this batch.
        CUDA_CHECK(cudaFree(d_query_nodes));
        CUDA_CHECK(cudaFree(d_results));
        
        // Wait for the fixed interval before processing the next time step.
        std::this_thread::sleep_for(std::chrono::milliseconds(fixed_interval_ms));
    }
    
    // Free the CSR graph arrays from device memory.
    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));
    
    return 0;
}
