// gpu/timestep_retrieval.cu
//
// This file demonstrates a fixed time-step (batched) retrieval approach in C.
// Each time step processes a batch of query nodes concurrently.
// Compile with: nvcc -x c timestep_retrieval.cu -o timestep_retrieval

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>  // for sleep (use unistd.h on Unix systems)

#define CUDA_CHECK(call) do {                              \
    if ((call) != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",       \
                cudaGetErrorString(call), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                               \
    } } while(0)

// Kernel: each thread retrieves the first neighbor for its query node.
__global__ void timestep_retrieve_kernel(const int* d_query_nodes, int num_queries,
                                           const int* d_indptr, const int* d_indices,
                                           int* d_results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_queries) {
        int query_node = d_query_nodes[tid];
        int start = d_indptr[query_node];
        int end   = d_indptr[query_node + 1];
        d_results[tid] = (start < end) ? d_indices[start] : -1;
    }
}

int main() {
    // Dummy CSR graph with 4 nodes.
    const int num_nodes = 4;
    int h_indptr[5]  = {0, 2, 4, 6, 8};
    int h_indices[8] = {1, 2, 0, 2, 1, 3, 0, 2};
    const int num_edges = 8;

    int *d_indptr = NULL, *d_indices = NULL;
    size_t size_indptr = (num_nodes + 1) * sizeof(int);
    size_t size_indices = num_edges * sizeof(int);
    CUDA_CHECK(cudaMalloc((void**)&d_indptr, size_indptr));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, size_indices));
    CUDA_CHECK(cudaMemcpy(d_indptr, h_indptr, size_indptr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, size_indices, cudaMemcpyHostToDevice));

    const int num_time_steps = 3;
    const int fixed_interval_sec = 1; // 1 second interval

    // Define queries for each time step.
    int queries0[2] = {0, 1};
    int queries1[2] = {2, 3};
    int queries2[2] = {0, 2};
    int* time_step_queries[3];
    int time_step_query_counts[3];
    time_step_queries[0] = queries0; time_step_query_counts[0] = 2;
    time_step_queries[1] = queries1; time_step_query_counts[1] = 2;
    time_step_queries[2] = queries2; time_step_query_counts[2] = 2;

    for (int t = 0; t < num_time_steps; t++) {
        printf("Time-step %d retrieval:\n", t);

        int num_queries = time_step_query_counts[t];
        int* h_query_nodes = time_step_queries[t];

        int *d_query_nodes = NULL, *d_results = NULL;
        size_t size_queries = num_queries * sizeof(int);
        CUDA_CHECK(cudaMalloc((void**)&d_query_nodes, size_queries));
        CUDA_CHECK(cudaMalloc((void**)&d_results, size_queries));

        CUDA_CHECK(cudaMemcpy(d_query_nodes, h_query_nodes, size_queries, cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocks = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
        timestep_retrieve_kernel<<<blocks, threadsPerBlock>>>(d_query_nodes, num_queries, d_indptr, d_indices, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* h_results = (int*) malloc(size_queries);
        CUDA_CHECK(cudaMemcpy(h_results, d_results, size_queries, cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_queries; i++) {
            printf("  Query node %d -> first neighbor: %d\n", h_query_nodes[i], h_results[i]);
        }

        free(h_results);
        CUDA_CHECK(cudaFree(d_query_nodes));
        CUDA_CHECK(cudaFree(d_results));

        sleep(fixed_interval_sec);
    }

    CUDA_CHECK(cudaFree(d_indptr));
    CUDA_CHECK(cudaFree(d_indices));

    return 0;
}
