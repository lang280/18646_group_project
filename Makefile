CC = gcc
NVCC = nvcc
CFLAGS = -O3
CUDAFLAGS = -O3 -arch=sm_75

# Define sources
CUDA_SOURCES = forward.cu backward.cu nn.cu
C_SOURCES = new.cu
HEADERS = nn.h forward.h backward.h logging.h

# Define objects
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)
OBJECTS = $(CUDA_OBJECTS) $(C_OBJECTS)

# Define the target
TARGET = cuda_nn

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^ -lcudart -lcuda -lm

# Compile CUDA sources
%.o: %.cu $(HEADERS)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Compile C sources
%.o: %.c $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)

run:
	make clean
	make
	./cuda_nn

# Phony targets
.PHONY: all clean