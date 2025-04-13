#!/bin/bash
# run_all.sh
# -------------
# This script automates the build and execution of the GraphRAG GPU retrieval pipeline.
# It runs the following steps:
#   1. Convert OGB BioKG to CSR format using the provided Python script.
#   2. Compile the CUDA/C++ components:
#         - graph_loader.cu (with cnpy)
#         - event_retrieval.cu
#         - timestep_retrieval.cu
#         - metrics_logger.cpp (for performance logging)
#   3. Run the compiled executables to verify the graph loading,
#      event-driven, and fixed time-step retrieval.
#   4. Run the metrics logger to generate a CSV file with performance data.
#   5. Generate and display latency plots using the Python analysis script.
#
# Usage:
#   ./run_all.sh

# Ensure the script exits if any command fails.
set -e

# Create a build directory if it does not exist.
BUILD_DIR="./build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# -----------------------------------------
# Step 1: Convert OGB BioKG to CSR format
# -----------------------------------------
echo "Converting BioKG to CSR format..."
# Adjust --node_type and --output_dir as needed.
python3 python/convert_biokg_to_csr.py --node_type gene --output_dir ./data/biokg_csr
echo "Conversion completed."

# ---------------------------------------------------
# Step 2: Compile CUDA/C++ components using nvcc
# ---------------------------------------------------
# (Make sure cnpy.cpp and cnpy.h are in the correct location relative to graph_loader.cu.)

echo "Compiling graph_loader.cu..."
nvcc gpu/graph_loader.cu cnpy.cpp -o "$BUILD_DIR/graph_loader"
echo "graph_loader compiled successfully."

echo "Compiling event_retrieval.cu..."
nvcc gpu/event_retrieval.cu -o "$BUILD_DIR/event_retrieval"
echo "event_retrieval compiled successfully."

echo "Compiling timestep_retrieval.cu..."
nvcc gpu/timestep_retrieval.cu -o "$BUILD_DIR/timestep_retrieval"
echo "timestep_retrieval compiled successfully."

echo "Compiling metrics_logger.cpp..."
nvcc benchmarks/metrics_logger.cpp -o "$BUILD_DIR/metrics_logger"
echo "metrics_logger compiled successfully."

# ---------------------------------------------------
# Step 3: Run the compiled executables to verify
# ---------------------------------------------------
echo "Running graph_loader..."
"$BUILD_DIR/graph_loader"

echo "Running event_retrieval..."
"$BUILD_DIR/event_retrieval"

echo "Running timestep_retrieval..."
"$BUILD_DIR/timestep_retrieval"

# ---------------------------------------------------
# Step 4: Run the metrics logger to collect performance data
# ---------------------------------------------------
echo "Running metrics_logger to generate performance metrics..."
"$BUILD_DIR/metrics_logger"

# ---------------------------------------------------
# Step 5: Plot latency metrics from metrics_logger.csv
# ---------------------------------------------------
echo "Generating latency plot..."
python3 analysis/latency_plot.py

echo "All steps completed successfully."
