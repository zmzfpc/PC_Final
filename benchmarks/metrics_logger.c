// benchmarks/metrics_logger.c
//
// This file demonstrates performance metric logging for a dummy retrieval kernel.
// It measures both CPU and GPU times, then writes the measurements to a CSV file.
// Compile with: nvcc -x c metrics_logger.c -o metrics_logger

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) do {                              \
    if ((call) != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",       \
                cudaGetErrorString(call), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                               \
    } } while(0)

// Dummy kernel: each thread performs an atomic addition.
__global__ void dummy_retrieval_kernel(int* d_counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(d_counter, 1);
}

// Structure to hold metrics.
typedef struct {
    float cpu_time_ms;
    float gpu_time_ms;
} Metric;

// Runs the kernel 'runs' times and collects metrics.
Metric* run_and_measure(int runs, int threads, int blocks) {
    int *d_counter = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    Metric* metrics = (Metric*) malloc(runs * sizeof(Metric));

    for (int i = 0; i < runs; i++) {
        struct timespec cpu_start, cpu_end;
        clock_gettime(CLOCK_MONOTONIC, &cpu_start);

        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        CUDA_CHECK(cudaEventRecord(startEvent, 0));

        dummy_retrieval_kernel<<<blocks, threads>>>(d_counter);

        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        clock_gettime(CLOCK_MONOTONIC, &cpu_end);

        float gpu_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, startEvent, stopEvent));

        float cpu_time_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0f +
                            (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1.0e6f;

        metrics[i].cpu_time_ms = cpu_time_ms;
        metrics[i].gpu_time_ms = gpu_time_ms;

        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
    }

    CUDA_CHECK(cudaFree(d_counter));
    return metrics;
}

void write_csv(const char* filename, Metric* metrics, int runs) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        return;
    }
    fprintf(fp, "RunIndex,CPU_Time_ms,GPU_Time_ms\n");
    for (int i = 0; i < runs; i++) {
        fprintf(fp, "%d,%.3f,%.3f\n", i, metrics[i].cpu_time_ms, metrics[i].gpu_time_ms);
    }
    fclose(fp);
    printf("Metrics written to %s\n", filename);
}

int main() {
    const int runs = 10;
    const int threads = 256;
    const int blocks = 1;

    Metric* metrics = run_and_measure(runs, threads, blocks);

    printf("Run  CPU_time_ms  GPU_time_ms\n");
    for (int i = 0; i < runs; i++) {
        printf("%d    %.3f    %.3f\n", i, metrics[i].cpu_time_ms, metrics[i].gpu_time_ms);
    }

    write_csv("retrieval_metrics.csv", metrics, runs);
    free(metrics);

    return 0;
}
