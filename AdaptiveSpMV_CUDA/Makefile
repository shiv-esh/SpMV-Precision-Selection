# Compiler settings
CC = gcc
NVCC = nvcc

# Directories
INCLUDE_DIR = include
SRC_DIR = src
BUILD_DIR = build

# Flags
CFLAGS = -I$(INCLUDE_DIR) -c
NVCCFLAGS = -I$(INCLUDE_DIR) -arch=sm_50 -c

# Source files
C_SOURCES = $(wildcard $(SRC_DIR)/*.c)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Object files
C_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Executable name
TARGET = $(BUILD_DIR)/spmv_mtx_comparison

# Default rule
all: $(TARGET)

# Rule to link all object files and create the final executable
$(TARGET): $(C_OBJS) $(CU_OBJS)
	$(NVCC) -o $@ $^

# Rule to compile C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

# Rule to compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean rule to remove all compiled files
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean
