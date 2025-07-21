# Compilers
CC = gcc
MPI_CC = mpicc

# Compiler flags
CFLAGS = -Wall -Wextra -march=native -I./include
#CFLAGS = -O2 -Wall -Wextra -I./include # Optimization flag
#CFLAGS = -g -Wall -Wextra -I./include # Debugging flag
MPI_FLAGS = -Wall -Wextra -I./include

# Source and object files
SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.c) # Search for all .c files in src directory
OBJS = $(SRCS:.c=.o) # Replace .c with .o for object files


# Executables
EXEC_SERIAL = stencil_serial 
EXEC_PARALLEL = stencil_parallel

# Default target
all: $(EXEC_SERIAL) #$(EXEC_PARALLEL) # Build both serial and parallel versions

# Serial version (compiled with gcc)
$(EXEC_SERIAL): $(SRC_DIR)/stencil_template_serial.o

	$(CC) $(CFLAGS) -o $@ $^

# Rule for serial object file
$(SRC_DIR)/stencil_template_serial.o: $(SRC_DIR)/stencil_template_serial.c

	$(CC) $(CFLAGS) -c $< -o $@

# Parallel version (compiled with mpicc)
#$(EXEC_PARALLEL): $(SRC_DIR)/stencil_template_parallel.o

#	$(MPI_CC) $(MPI_FLAGS) -o $@ $^ -fopenmp

# Rule for parallel object file (MPI)
#$(SRC_DIR)/stencil_template_parallel.o: $(SRC_DIR)/stencil_template_parallel.c

#	$(MPI_CC) $(MPI_FLAGS) -c $< -o $@

# Clean target
clean:

	rm -f $(OBJS) $(EXEC_SERIAL) $(EXEC_PARALLEL)

.PHONY: all clean