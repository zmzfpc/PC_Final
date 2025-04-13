#!/bin/bash
# run.sh
# -------------
# This script automates the build and execution of the GraphRAG GPU retrieval pipeline,
# using the C versions of the project components. It performs the following steps:
#
#   1. Run the Python conversion step (if needed) to convert BioKG to CSR format.
#      (This example uses convert_biokg_to_csr.py which should now produce binary files.)
#   2. Create a build directory.
#   3. Compile the CUDA/C source files:
#         - gpu/graph_loader.cu
#         - gpu/event_retrieval.cu
#         - gpu/timestep_retrieval.cu
#         - benchmarks/metrics_logger.c
#   4. Run the compiled executables.
#   5. Generate a latency plot using a Python script.
#
# Usage:
#   ./run.sh

# Exit script if any command fails.
set -e

# Create the build directory if it doesn't exist.
BUILD_DIR="./build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# ---------------------------------------------
# Step 1: Convert BioKG to CSR in binary format
# ---------------------------------------------
echo "Converting BioKG to CSR format..."
# This command runs a Python script that should output binary CSR files 
# (csr_indptr.bin, csr_indices.bin) into the data/biokg_csr directory.
python3 python/convert_biokg_to_csr.py --node_type gene --output_dir ./data/biokg_csr
echo "Conversion completed."

# ---------------------------------------------
# Step 2: Compile the CUDA/C components
# ---------------------------------------------
echo "Compiling gpu/graph_loader.cu (C version)..."
nvcc -x c gpu/graph_loader.cu -o "$BUILD_DIR/graph_loader"

echo "Compiling gpu/event_retrieval.cu (C version)..."
nvcc -x c gpu/event_retrieval.cu -o "$BUILD_DIR/event_retrieval"

echo "Compiling gpu/timestep_retrieval.cu (C version)..."
nvcc -x c gpu/timestep_retrieval.cu -o "$BUILD_DIR/timestep_retrieval"

echo "Compiling benchmarks/metrics_logger.c (C version)..."
nvcc -x c benchmarks/metrics_logger.c -o "$BUILD_DIR/metrics_logger"

# ---------------------------------------------
# Step 3: Run the compiled executables
# ---------------------------------------------
echo "Running graph_loader..."
"$BUILD_DIR/graph_loader"

echo "Running event_retrieval..."
"$BUILD_DIR/event_retrieval"

echo "Running timestep_retrieval..."
"$BUILD_DIR/timestep_retrieval"

echo "Running metrics_logger..."
"$BUILD_DIR/metrics_logger"

# ---------------------------------------------
# Step 4: Generate and view latency plots
# ---------------------------------------------
echo "Generating latency plot..."
python3 analysis/latency_plot.py

echo "All steps completed successfully."
