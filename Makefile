# Compilers
CC = gcc
MPI_CC = mpicc

# Compiler flags
CFLAGS  = -Wall -Wextra -march=native -I./include
OMPFLAG = -fopenmp

# Source directory
SRC_DIR = src

# Source files
SRC_SERIAL       = $(SRC_DIR)/stencil_template_serial.c
SRC_SERIAL_NO_OMP = $(SRC_DIR)/stencil_template_serial_nomp.c
SRC_PARALLEL     = $(SRC_DIR)/stencil_template_parallel.c

# Object files
OBJ_SERIAL       = $(SRC_SERIAL:.c=.o)
OBJ_SERIAL_NO_OMP = $(SRC_SERIAL_NO_OMP:.c=.o)
OBJ_PARALLEL     = $(SRC_PARALLEL:.c=.o)

# Executables
EXEC_SERIAL      = stencil_serial
EXEC_SERIAL_NO_OMP = stencil_serial_nomp
EXEC_PARALLEL    = stencil_parallel

# Default target
all: $(EXEC_SERIAL) $(EXEC_SERIAL_NO_OMP) $(EXEC_PARALLEL) # Build both serial and parallel versions

# =================== SERIAL WITH OPENMP ===================
$(EXEC_SERIAL): $(OBJ_SERIAL)
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $^

$(OBJ_SERIAL): $(SRC_SERIAL)
	$(CC) $(CFLAGS) $(OMPFLAG) -c $< -o $@

# =================== SERIAL WITHOUT OPENMP ===================
$(EXEC_SERIAL_NO_OMP): $(OBJ_SERIAL_NO_OMP)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_SERIAL_NO_OMP): $(SRC_SERIAL_NO_OMP)
	$(CC) $(CFLAGS) -c $< -o $@

# =================== PARALLEL WITH MPI+OPENMP ===================
$(EXEC_PARALLEL): $(OBJ_PARALLEL)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) -o $@ $^

$(OBJ_PARALLEL): $(SRC_PARALLEL)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) -c $< -o $@

# Clean target
clean:

	rm -f $(SRC_DIR)/*.o $(EXEC_SERIAL) $(EXEC_SERIAL_NO_OMP) $(EXEC_PARALLEL)

.PHONY: all clean