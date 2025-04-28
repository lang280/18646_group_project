#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect GPU architecture (compute capability)
compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
if [ -z "$compute_capability" ]; then
    # Default to 70 (Volta) if detection fails
    compute_capability=70
fi

echo -e "${GREEN}Detected GPU compute capability: sm_${compute_capability}${NC}"

# Compile the benchmark
echo -e "${GREEN}Compiling benchmark 'output_delta_kernel_batch'...${NC}"
nvcc -O3 -arch=sm_${compute_capability} --use_fast_math -o output_delta_kernel_batch output_delta_kernel_batch.cu

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Compilation successful${NC}"
    
    # Run the benchmark
    echo -e "${GREEN}Running benchmark 'output_delta_kernel_batch'...${NC}"
    ./output_delta_kernel_batch "$@"
    
    # Check if benchmark completed successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully${NC}"
    else
        echo -e "${RED}Benchmark failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}Compilation failed${NC}"
    exit 1
fi

exit 0 