// metrics_logger.cpp
//
// Example code to demonstrate logging GPU metrics (latency) and CPU overhead
// for a retrieval kernel. It runs a simple dummy kernel multiple times and
// collects performance stats, then writes them to a CSV file.
//
// Compile and run:
//   nvcc metrics_logger.cpp -o metrics_logger
//   ./metrics_logger
//

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Macro for CUDA error checking.
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------
// A trivial kernel that does almost nothing. 
// You would replace this with your retrieval kernel.
//
// For demonstration, each thread just increments a counter in global memory.
// ---------------------------------------------------------------------
__global__ void dummy_retrieval_kernel(int* d_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread does a small atomic add to simulate minimal work.
    atomicAdd(d_counter, 1);
}

// ---------------------------------------------------------------------
// Host function to run the kernel and measure GPU latency using cudaEvents.
// Also measures CPU overhead using std::chrono.
//
// Parameters:
//   runs - how many times to run the kernel
//   threads - number of threads in the kernel per launch
//   blocks - number of blocks in the kernel per launch
//
// Returns:
//   A vector of pairs, each containing {cpu_time_ms, gpu_time_ms} for each run.
// ---------------------------------------------------------------------
std::vector<std::pair<float, float>> run_and_measure(int runs, int threads, int blocks) {
    // Prepare a device counter for demonstration.
    int* d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // Vector to store (CPU_time, GPU_time) for each run.
    std::vector<std::pair<float, float>> measurements;
    measurements.reserve(runs);

    for (int i = 0; i < runs; ++i) {
        // CPU timing start
        auto cpu_start = std::chrono::high_resolution_clock::now();

        // Create CUDA events for GPU timing.
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        // Record GPU start event
        CUDA_CHECK(cudaEventRecord(startEvent, 0));

        // Launch kernel
        dummy_retrieval_kernel<<<blocks, threads>>>(d_counter);

        // Record GPU stop event
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        // Synchronize so we can get the final GPU timing for this run
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        // CPU timing stop
        auto cpu_end = std::chrono::high_resolution_clock::now();

        // Calculate GPU time (milliseconds)
        float gpu_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, startEvent, stopEvent));

        // Calculate CPU time (milliseconds)
        float cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

        // Store the pair of measurements
        measurements.push_back({cpu_time_ms, gpu_time_ms});

        // Destroy the events
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_counter));
    return measurements;
}

// ---------------------------------------------------------------------
// Utility function to write the metrics (CPU time, GPU time) to a CSV file.
//
// The CSV columns: RunIndex, CPU_Time_ms, GPU_Time_ms
// ---------------------------------------------------------------------
void write_csv(const std::string& filename,
               const std::vector<std::pair<float, float>>& measurements) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    fout << "RunIndex,CPU_Time_ms,GPU_Time_ms\n";
    for (size_t i = 0; i < measurements.size(); ++i) {
        fout << i << "," << measurements[i].first << "," << measurements[i].second << "\n";
    }
    fout.close();
    std::cout << "Metrics written to " << filename << std::endl;
}

int main() {
    // Example usage: run the dummy retrieval kernel 10 times with 256 threads in 1 block.
    const int runs = 10;
    const int threads = 256;
    const int blocks = 1;

    // Run the kernel multiple times, measuring CPU and GPU times.
    std::vector<std::pair<float, float>> metrics = run_and_measure(runs, threads, blocks);

    // Print out results to console
    std::cout << "Run  CPU_time_ms  GPU_time_ms\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        std::cout << i << "    " << metrics[i].first << "    " << metrics[i].second << "\n";
    }

    // Optionally, write results to a CSV file
    write_csv("retrieval_metrics.csv", metrics);

    return 0;
}
