# ============================================================================
# Makefile for 2D Heat Diffusion Stencil Simulation - Multiple Implementations
# ============================================================================
#
# This Makefile builds multiple versions of the heat diffusion simulation
# to demonstrate the progression from serial to highly optimized parallel code:
#
# BUILD TARGETS (in order of complexity):
# 1. Serial (CPU-only)           - Basic single-threaded implementation
# 2. OpenMP (Shared Memory)      - Multi-threaded on single node
# 3. MPI (Distributed Memory)    - Multi-process across nodes
# 4. MPI+OpenMP (Hybrid)         - Distributed + shared memory parallelism  
# 5. MPI+OpenMP Optimized        - Advanced hybrid with communication overlap
#
# USAGE:
#   make                    - Build all versions
#   make stencil_serial     - Build serial version only
#   make stencil_omp        - Build OpenMP version only
#   make stencil_mpi        - Build MPI-only version only
#   make stencil_parallel   - Build MPI+OpenMP hybrid version
#   make stencil_parallel_2 - Build optimized MPI+OpenMP version
#   make clean             - Remove all build artifacts
#
# COMPILER REQUIREMENTS:
#   - GCC with OpenMP support (gcc with -fopenmp)
#   - MPI implementation (OpenMPI, MPICH, etc.)
#   - Modern CPU supporting -march=native optimizations
#
# PERFORMANCE FLAGS:
#   - -Ofast: Aggressive optimizations including fast-math
#   - -flto: Link-time optimization for cross-module optimizations
#   - -march=native: CPU-specific optimizations for target architecture
#   - -fopenmp: OpenMP support for shared memory parallelism
#
# ============================================================================

# Build configuration
BUILDDIR = build
CC = gcc
MPI_CC = mpicc

# Compiler flags
CFLAGS = -Wall -Wextra -I./include

# Optimization flags for high performance computing
OPT_CFLAGS = -Ofast -flto -march=native
OMPFLAG = -fopenmp

# Source directory
SRC_DIR = src

# Create build directory if it doesn't exist
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ============================================================================
# SOURCE FILES AND BUILD TARGETS (ordered by complexity)
# ============================================================================

# Source files (progression: Serial → OpenMP → MPI → MPI+OpenMP → Optimized)
SRC_SERIAL          = $(SRC_DIR)/stencil_serial_nomp.c   # 1. Pure serial implementation (no OpenMP)
SRC_SERIAL_OMP      = $(SRC_DIR)/stencil_serial.c        # 2. Serial with OpenMP (shared memory)
SRC_MPI_ONLY        = $(SRC_DIR)/stencil_parallel.c      # 3. MPI-only (distributed memory)
SRC_MPI_OMP_HYBRID  = $(SRC_DIR)/stencil_parallel.c      # 4. MPI+OpenMP hybrid (same source, different flags)
SRC_MPI_OMP_OPTIMIZED = $(SRC_DIR)/stencil_parallel_2.c  # 5. Optimized hybrid (communication-computation overlap)

# Object files (corresponding to source files above)
OBJ_SERIAL          = $(BUILDDIR)/stencil_serial_nomp.o   # Pure serial object
OBJ_SERIAL_OMP      = $(BUILDDIR)/stencil_serial.o       # OpenMP object
OBJ_MPI_ONLY        = $(BUILDDIR)/stencil_parallel_nomp.o # MPI-only object
OBJ_MPI_OMP_HYBRID  = $(BUILDDIR)/stencil_parallel.o     # Hybrid object
OBJ_MPI_OMP_OPTIMIZED = $(BUILDDIR)/stencil_parallel_2.o # Optimized object

# Executables (ordered by increasing parallelization complexity)
EXEC_SERIAL          = stencil_serial_nomp      # 1. Serial baseline (single-threaded)
EXEC_SERIAL_OMP      = stencil_serial           # 2. OpenMP version (multi-threaded)
EXEC_MPI_ONLY        = stencil_parallel_nomp    # 3. MPI-only (distributed, single-threaded per process)
EXEC_MPI_OMP_HYBRID  = stencil_parallel         # 4. Full hybrid (distributed + multi-threaded)
EXEC_MPI_OMP_OPTIMIZED = stencil_parallel_2     # 5. Optimized hybrid (advanced optimizations)

# Default target: Build all versions in logical progression order
all: $(EXEC_SERIAL) $(EXEC_SERIAL_OMP) $(EXEC_MPI_ONLY) $(EXEC_MPI_OMP_HYBRID) $(EXEC_MPI_OMP_OPTIMIZED)

# ============================================================================
# BUILD RULES (ordered by increasing parallelization complexity)
# ============================================================================

# =================== 1. SERIAL VERSION (Pure single-threaded) ===================
# Baseline implementation without any parallelization for performance comparison
# Uses stencil_serial_nomp.c source file specifically designed without OpenMP
$(EXEC_SERIAL): $(OBJ_SERIAL)
	$(CC) $(CFLAGS) $(OPT_CFLAGS) -o $@ $^

$(OBJ_SERIAL): $(SRC_SERIAL) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OPT_CFLAGS) -c $< -o $@

# =================== 2. OPENMP VERSION (Shared memory parallelism) ===================
# Multi-threaded version using OpenMP for shared memory parallelism on single node
# Uses stencil_serial.c with OpenMP pragmas enabled via -fopenmp flag
$(EXEC_SERIAL_OMP): $(OBJ_SERIAL_OMP)
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -o $@ $^

$(OBJ_SERIAL_OMP): $(SRC_SERIAL_OMP) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -c $< -o $@

# =================== 3. MPI-ONLY VERSION (Distributed memory only) ===================
# MPI-only version without OpenMP - pure distributed memory parallelism
# Uses stencil_parallel.c source but compiles WITHOUT -fopenmp flag
# OpenMP pragmas in source are ignored by compiler when -fopenmp is omitted
$(EXEC_MPI_ONLY): $(OBJ_MPI_ONLY)
	$(MPI_CC) $(CFLAGS) $(OPT_CFLAGS) -o $@ $^

$(OBJ_MPI_ONLY): $(SRC_MPI_ONLY) | $(BUILDDIR)
	$(MPI_CC) $(CFLAGS) $(OPT_CFLAGS) -c $< -o $@

# =================== 4. MPI+OPENMP HYBRID (Full hybrid parallelization) ===================
# Complete hybrid implementation: MPI for inter-node + OpenMP for intra-node
# Uses same stencil_parallel.c source as MPI-only but WITH -fopenmp flag enabled
# This demonstrates how same code can be MPI-only or hybrid based on compiler flags
$(EXEC_MPI_OMP_HYBRID): $(OBJ_MPI_OMP_HYBRID)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -o $@ $^

$(OBJ_MPI_OMP_HYBRID): $(SRC_MPI_OMP_HYBRID) | $(BUILDDIR)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -c $< -o $@

# =================== 5. OPTIMIZED MPI+OPENMP (Advanced hybrid with optimizations) ===================
# Highly optimized hybrid version with communication-computation overlap
# Uses stencil_parallel_2.c with split stencil, prefetching, and NUMA optimizations
# Implements non-blocking MPI communication overlapped with interior point computation
$(EXEC_MPI_OMP_OPTIMIZED): $(OBJ_MPI_OMP_OPTIMIZED)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -o $@ $^

$(OBJ_MPI_OMP_OPTIMIZED): $(SRC_MPI_OMP_OPTIMIZED) | $(BUILDDIR)
	$(MPI_CC) $(CFLAGS) $(OMPFLAG) $(OPT_CFLAGS) -c $< -o $@

# ============================================================================
# UTILITY TARGETS
# ============================================================================

# Clean all build artifacts - removes object files, executables, and build directory
clean:
	rm -rf $(BUILDDIR) $(EXEC_SERIAL) $(EXEC_SERIAL_OMP) $(EXEC_MPI_ONLY) $(EXEC_MPI_OMP_HYBRID) $(EXEC_MPI_OMP_OPTIMIZED)

# Help target to display comprehensive build and usage information
help:
	@echo "============================================================================"
	@echo "2D Heat Diffusion Stencil Simulation - Build Options"
	@echo "============================================================================"
	@echo ""
	@echo "AVAILABLE TARGETS (in order of increasing parallelization complexity):"
	@echo "  stencil_serial_nomp  - 1. Serial version (pure single-threaded baseline)"
	@echo "  stencil_serial       - 2. OpenMP version (shared memory parallelism)"
	@echo "  stencil_parallel_nomp- 3. MPI-only version (distributed memory only)"
	@echo "  stencil_parallel     - 4. MPI+OpenMP hybrid (full parallelization)"
	@echo "  stencil_parallel_2   - 5. Optimized hybrid (comm-comp overlap + optimizations)"
	@echo ""
	@echo "UTILITY TARGETS:"
	@echo "  all                  - Build all versions in complexity order"
	@echo "  clean                - Remove all build artifacts (objects + executables)"
	@echo "  help                 - Display this comprehensive help message"
	@echo ""
	@echo "COMPILATION DETAILS:"
	@echo "  - Serial versions use GCC compiler"
	@echo "  - MPI versions use mpicc wrapper (links MPI libraries automatically)"
	@echo "  - OpenMP enabled with -fopenmp flag where applicable"
	@echo "  - Aggressive optimizations: -Ofast -flto -march=native"
	@echo "  - Same source files compiled with different parallelization flags"
	@echo ""
	@echo "EXAMPLE USAGE:"
	@echo "  ./stencil_serial_nomp -x 1000 -y 1000 -n 100"
	@echo "  OMP_NUM_THREADS=4 ./stencil_serial -x 2000 -y 2000 -n 200"
	@echo "  mpirun -n 4 ./stencil_parallel_nomp -x 4000 -y 4000 -n 500"
	@echo "  OMP_NUM_THREADS=2 mpirun -n 4 ./stencil_parallel -x 8000 -y 8000 -n 1000"
	@echo "  OMP_NUM_THREADS=2 mpirun -n 4 ./stencil_parallel_2 -x 8000 -y 8000 -n 1000"
	@echo ""
	@echo "PERFORMANCE COMPARISON STRATEGY:"
	@echo "  1. Start with serial baseline for reference performance"
	@echo "  2. Test OpenMP scaling on single node (vary OMP_NUM_THREADS)"
	@echo "  3. Test MPI-only scaling across nodes (vary number of MPI ranks)"
	@echo "  4. Test hybrid scaling (balance MPI ranks vs OpenMP threads)"
	@echo "  5. Compare optimized version for communication overlap benefits"
	@echo "============================================================================"

# Individual target help for quick reference
targets:
	@echo "Available make targets:"
	@echo "  all, stencil_serial_nomp, stencil_serial, stencil_parallel_nomp,"
	@echo "  stencil_parallel, stencil_parallel_2, clean, help, targets"

# Declare phony targets (not actual files)
.PHONY: all clean help targets