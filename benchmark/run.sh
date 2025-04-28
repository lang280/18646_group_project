#!/bin/bash

# CUDA Neural Network Kernel Benchmark Runner
# Usage: ./run.sh [benchmark_name] [options]

set -e  # Exit on error

# Default values
BENCHMARK=""
BATCH_SIZE=128
KERNEL_TYPE="all"
PROFILE=0
PROFILE_METRICS="achieved_occupancy,inst_executed,gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,flop_count_dp"
COMPILE_ONLY=0
RUN_ONLY=0
VERBOSE=0

# ANSI colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print usage information
print_usage() {
    echo -e "${BLUE}Usage:${NC} ./run.sh [benchmark_name] [options]"
    echo
    echo "benchmark_name: Name of the benchmark file (without .cu extension)"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo "  --batch-size N      Set the batch size (default: 128)"
    echo "  --kernel N          Run only specific kernel (0=original, 1=optimized, 2=transposed)"
    echo "  --profile           Enable NVIDIA profiling"
    echo "  --metrics LIST      Comma-separated list of profiling metrics (with --profile)"
    echo "  --compile           Compile only, don't run"
    echo "  --run               Run only, don't compile"
    echo "  --verbose           Print more detailed output"
    echo "  --help              Display this help message"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./run.sh hidden_layer_kernel_batch"
    echo "  ./run.sh hidden_layer_kernel_batch --batch-size 256 --kernel 2"
    echo "  ./run.sh hidden_layer_kernel_batch --profile"
}

# Process command line arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No benchmark specified${NC}"
    print_usage
    exit 1
fi

BENCHMARK=$1
shift

# Check if benchmark file exists
if [ ! -f "${BENCHMARK}.cu" ]; then
    echo -e "${RED}Error: Benchmark file '${BENCHMARK}.cu' not found${NC}"
    exit 1
fi

# Process remaining arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --kernel)
            KERNEL_TYPE="$2"
            shift 2
            ;;
        --profile)
            PROFILE=1
            shift
            ;;
        --metrics)
            PROFILE_METRICS="$2"
            shift 2
            ;;
        --compile)
            COMPILE_ONLY=1
            shift
            ;;
        --run)
            RUN_ONLY=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Determine CUDA compute capability
detect_compute_capability() {
    echo "70"
}

COMPUTE_CAP=$(detect_compute_capability)
ARCH="sm_${COMPUTE_CAP}"

# Print configuration if verbose
if [ $VERBOSE -eq 1 ]; then
    echo -e "${BLUE}Configuration:${NC}"
    echo "  Benchmark:      ${BENCHMARK}"
    echo "  Batch size:     ${BATCH_SIZE}"
    echo "  Kernel type:    ${KERNEL_TYPE}"
    echo "  GPU arch:       ${ARCH}"
    echo "  Profile:        $([ $PROFILE -eq 1 ] && echo 'Yes' || echo 'No')"
    [ $PROFILE -eq 1 ] && echo "  Profile metrics: ${PROFILE_METRICS}"
    echo
fi

# Compile benchmark if not run-only
if [ $RUN_ONLY -eq 0 ]; then
    echo -e "${GREEN}Compiling benchmark '${BENCHMARK}'...${NC}"
    
    # Create build directory if it doesn't exist
    mkdir -p build
    
    # Compile with optimization flags
    NVCC_CMD="nvcc -O3 -arch=${ARCH} --use_fast_math -lineinfo ${BENCHMARK}.cu -o build/${BENCHMARK}"
    
    if [ $VERBOSE -eq 1 ]; then
        echo "Compile command: ${NVCC_CMD}"
    fi
    
    eval $NVCC_CMD
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Compilation successful${NC}"
    else
        echo -e "${RED}Compilation failed${NC}"
        exit 1
    fi
fi

# Exit if compile-only
if [ $COMPILE_ONLY -eq 1 ]; then
    exit 0
fi

# Run the benchmark
if [ $COMPILE_ONLY -eq 0 ]; then
    echo -e "${GREEN}Running benchmark '${BENCHMARK}'...${NC}"
    
    BENCH_CMD="./build/${BENCHMARK}"
    
    # Add batch size parameter if specified
    if [ "$BATCH_SIZE" != "128" ]; then
        BENCH_CMD="${BENCH_CMD} --batch-size ${BATCH_SIZE}"
    fi
    
    # Add kernel parameter if specified
    if [ "$KERNEL_TYPE" != "all" ]; then
        BENCH_CMD="${BENCH_CMD} --kernel ${KERNEL_TYPE}"
    fi
    
    # Run with or without profiling
    if [ $PROFILE -eq 1 ]; then
        PROFILE_CMD="nvprof --metrics ${PROFILE_METRICS} ${BENCH_CMD}"
        
        echo -e "${YELLOW}Running with NVIDIA profiler${NC}"
        if [ $VERBOSE -eq 1 ]; then
            echo "Profile command: ${PROFILE_CMD}"
        fi
        
        eval $PROFILE_CMD
    else
        if [ $VERBOSE -eq 1 ]; then
            echo "Run command: ${BENCH_CMD}"
        fi
        
        eval $BENCH_CMD
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully${NC}"
    else
        echo -e "${RED}Benchmark failed${NC}"
        exit 1
    fi
fi

exit 0
