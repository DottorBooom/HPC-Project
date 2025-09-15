# 2D Heat Diffusion Stencil Simulation - HPC Implementation

This repository contains comprehensive solutions for the **High Performance Computing (HPC) exam** in the Master's degree program "Data Science and Artificial Intelligence" at the **University of Trieste**. The project implements multiple parallel versions of a 2D heat diffusion simulation using stencil computation, demonstrating the progression from serial to highly optimized parallel implementations.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Implementation Versions](#implementation-versions)
4. [Build System](#build-system)
5. [Execution Guide](#execution-guide)
6. [Performance Analysis](#performance-analysis)
7. [Cluster Execution (SLURM)](#cluster-execution-slurm)
8. [Expected Results](#expected-results)
9. [Technical Details](#technical-details)
10. [Requirements](#requirements)

---

## 🎯 Project Overview

The project implements a **2D heat diffusion simulation** using the finite difference method with a 5-point stencil. The simulation models heat propagation in a 2D grid where energy sources inject heat at specified locations, and the heat diffuses according to the diffusion equation.

### Key Features:
- **Multiple parallelization strategies**: Serial, OpenMP, MPI, and hybrid MPI+OpenMP
- **Advanced optimizations**: Communication-computation overlap, cache prefetching, NUMA awareness
- **Comprehensive benchmarking**: Strong and weak scaling analysis
- **Production-ready code**: Professional documentation and error handling

### Physical Model:
```
∂u/∂t = α∇²u + f(x,y,t)
```
Where:
- `u(x,y,t)` is the temperature at position (x,y) and time t
- `α` is the thermal diffusivity
- `f(x,y,t)` represents heat sources

---

## 📁 Repository Structure

```
Assignment/
├── src/                          # Source code implementations
│   ├── stencil_serial_nomp.c     # Pure serial implementation
│   ├── stencil_serial.c          # OpenMP parallel implementation
│   ├── stencil_parallel.c        # MPI + OpenMP hybrid implementation
│   └── stencil_parallel_2.c      # Optimized hybrid with communication overlap
├── include/                      # Header files
│   ├── stencil_serial.h         # Serial version header
│   ├── stencil_parallel.h       # MPI version header
│   └── stencil_parallel_2.h     # Optimized version header
├── build/                        # Build artifacts (created during compilation)
├── results/                      # Performance analysis results
│   ├── strong_scaling/          # Strong scaling benchmark results
│   └── weak_scaling/            # Weak scaling benchmark results
├── Makefile                     # Comprehensive build system
├── Strong_scaling               # SLURM job script for strong scaling analysis
├── Weak_scaling                 # SLURM job script for weak scaling analysis
└── README.md                    # This file
```

---

## 🔧 Implementation Versions

The project demonstrates **5 different implementations** with increasing complexity and optimization:

### 1. **Serial Baseline** (`stencil_serial_nomp`)
- **Purpose**: Performance baseline and correctness reference
- **Parallelization**: None (single-threaded)
- **Use case**: Reference implementation and small problem debugging

### 2. **OpenMP Shared Memory** (`stencil_serial`)
- **Purpose**: Shared memory parallelization on single node
- **Parallelization**: OpenMP threading
- **Features**: Work-sharing constructs, thread-safe random number generation
- **Scaling**: Up to node capacity (typically 64-128 threads)

### 3. **MPI Distributed Memory** (`stencil_parallel_nomp`)
- **Purpose**: Distributed memory parallelization across nodes
- **Parallelization**: MPI processes only (compiled without OpenMP)
- **Features**: Domain decomposition, halo exchange, non-blocking communication
- **Scaling**: Multi-node clusters

### 4. **MPI+OpenMP Hybrid Standard** (`stencil_parallel`)
- **Purpose**: Full hybrid parallelization (distributed + shared memory)
- **Parallelization**: MPI between nodes + OpenMP within nodes
- **Features**: Hierarchical parallelism, optimal core utilization
- **Scaling**: Large-scale HPC systems

### 5. **MPI+OpenMP Optimized** (`stencil_parallel_2`)
- **Purpose**: Highly optimized hybrid with advanced techniques
- **Parallelization**: Advanced hybrid with communication-computation overlap
- **Features**: 
  - **Split stencil computation**: Interior vs. boundary points
  - **Non-blocking MPI communication** overlapped with interior computation
  - **Cache prefetching**: Hardware-specific optimizations (`_mm_prefetch`)
  - **NUMA awareness**: First-touch memory allocation policy
  - **Memory alignment**: 64-byte aligned allocation for SIMD

---

## 🔨 Build System

### Compilation

The project uses a comprehensive **Makefile** with optimized compilation flags:

```bash
# Build all versions
make all

# Build specific versions
make stencil_serial_nomp      # Pure serial
make stencil_serial           # OpenMP version
make stencil_parallel_nomp    # MPI-only version
make stencil_parallel         # Standard hybrid
make stencil_parallel_2       # Optimized hybrid

# Clean build artifacts
make clean

# Display help
make help
```

### Compiler Optimizations
- **`-Ofast`**: Aggressive optimizations including fast-math
- **`-flto`**: Link-time optimization for cross-module optimizations
- **`-march=native`**: CPU-specific optimizations for target architecture
- **`-fopenmp`**: OpenMP support (when applicable)

---

## 🚀 Execution Guide

### Command Line Parameters

All implementations share the same command-line interface:

```bash
./executable -x <width> -y <height> -n <iterations> -e <sources> -E <energy> [options]
```

**Parameters:**
- **`-x <width>`**: Grid width (number of points)
- **`-y <height>`**: Grid height (number of points)  
- **`-n <iterations>`**: Number of time steps
- **`-e <sources>`**: Number of random heat sources
- **`-E <energy>`**: Energy per source
- **`-p <0|1>`**: Periodic boundary conditions (0=off, 1=on)

### Execution Examples

#### 1. Serial Execution
```bash
./stencil_serial_nomp -x 1000 -y 1000 -n 100 -e 100 -E 1.0 -p 1
```

#### 2. OpenMP Execution
```bash
# Set thread count
export OMP_NUM_THREADS=8
./stencil_serial -x 2000 -y 2000 -n 200 -e 200 -E 1.0 -p 1
```

#### 3. MPI-Only Execution
```bash
mpirun -n 4 ./stencil_parallel_nomp -x 4000 -y 4000 -n 500 -e 500 -E 1.0 -p 1
```

#### 4. Hybrid MPI+OpenMP Execution
```bash
# Standard hybrid
export OMP_NUM_THREADS=4
mpirun -n 4 ./stencil_parallel -x 8000 -y 8000 -n 1000 -e 1000 -E 1.0 -p 1

# Optimized hybrid
export OMP_NUM_THREADS=4
mpirun -n 4 ./stencil_parallel_2 -x 8000 -y 8000 -n 1000 -e 1000 -E 1.0 -p 1
```

---

## 📊 Performance Analysis

### Output Metrics

All implementations provide comprehensive performance metrics:

#### Timing Information:
- **Total execution time**
- **Initialization time**
- **Computation time** (with percentage breakdown)
- **Communication time** (MPI versions)
- **Wait time** (synchronization overhead)
- **Other overhead** (memory allocation, I/O, etc.)

#### Parallel Performance Metrics:
- **Load imbalance**: Measure of work distribution quality
- **Load balance efficiency**: Ratio of average to maximum computation time
- **Communication efficiency**: Ratio of computation to total time

### Sample Output

```
Total time: 12.345678
Initialization time: 0.123456
Computation time: 10.234567 (82.89%)
Communication time: 1.456789 (11.80%)
Wait time for communication: 0.234567
Other time (overhead): 0.296299 (2.40%)

Max total time: 12.456789
Max computation time: 10.345678
Max communication time: 1.567890
Max wait time for communication: 0.345678

Load imbalance: 0.025432
Load balance efficiency: 0.988765
Communication efficiency: 0.821234
```

---

## 🖥️ Cluster Execution (SLURM)

The repository includes two professional **SLURM job scripts** for comprehensive performance analysis on HPC clusters.

### Strong Scaling Analysis

**Purpose**: Keep problem size constant while increasing core count to measure speedup.

```bash
# Submit strong scaling job
sbatch Strong_scaling
```

**Configuration:**
- **Fixed problem size**: 6144 × 6144 grid, 1000 iterations
- **Scaling range**: 1 to 128 cores (powers of 2)
- **Implementations tested**: OpenMP, MPI-only, Standard Hybrid, Optimized Hybrid
- **Metrics collected**: Speedup, efficiency, Amdahl's law analysis

### Weak Scaling Analysis

**Purpose**: Keep work per core constant while increasing both problem size and core count.

```bash
# Submit weak scaling job
sbatch Weak_scaling
```

**Configuration:**
- **Scaling formula**: Grid_size = Base_size × √(cores)
- **Base problem**: 2048 × 2048 per core
- **Scaling range**: 1 to 128 cores
- **Ideal result**: Constant execution time (100% efficiency)

### SLURM Job Features

Both scripts include:
- **Automatic compilation** and executable validation
- **Comprehensive error handling** and output parsing
- **Multiple runs** for statistical reliability
- **CSV output** for analysis tools
- **Professional logging** with performance breakdowns

---

## 📈 Expected Results

### Strong Scaling Results

**Ideal Behavior:**
- **Linear speedup**: Execution time reduces proportionally to core count
- **High efficiency**: > 80% efficiency up to moderate core counts
- **Amdahl's law**: Serial fraction limits theoretical maximum speedup

**CSV Output Files:**
- `results/strong_omp_slurm.csv`: OpenMP scaling data
- `results/strong_mpi_slurm.csv`: MPI-only scaling data
- `results/strong_hybrid_slurm.csv`: Standard hybrid scaling data
- `results/strong_hybrid_v2_slurm.csv`: Optimized hybrid scaling data

### Weak Scaling Results

**Ideal Behavior:**
- **Constant execution time**: Time remains stable as problem size and cores increase
- **100% efficiency**: Perfect weak scaling maintains efficiency = 1.0
- **Communication overhead**: May increase with scale

**CSV Output Files:**
- `results/weak_scaling/weak_omp_slurm.csv`: OpenMP weak scaling data
- `results/weak_scaling/weak_mpi_slurm.csv`: MPI-only weak scaling data
- `results/weak_scaling/weak_hybrid_slurm.csv`: Standard hybrid weak scaling data
- `results/weak_scaling/weak_hybrid_v2_slurm.csv`: Optimized hybrid weak scaling data

### Performance Comparison Expectations

1. **Serial baseline**: Provides reference performance
2. **OpenMP**: Good scaling up to node limits (memory bandwidth bound)
3. **MPI-only**: Excellent multi-node scaling but higher communication overhead
4. **Standard hybrid**: Best of both approaches with optimal resource utilization
5. **Optimized hybrid**: Superior performance due to communication-computation overlap

---

## 🔬 Technical Details

### Algorithmic Approach

**5-Point Stencil Pattern:**
```
    [ 0  1  0 ]
    [ 1 -4  1 ]  ×  u[i,j]
    [ 0  1  0 ]
```

**Domain Decomposition:**
- **1D decomposition**: Horizontal strips for simplicity
- **Halo exchange**: Ghost cells for boundary communication
- **Load balancing**: Equal-sized sub-domains per process

### Advanced Optimizations (Version 2)

#### Communication-Computation Overlap:
1. **Energy injection** on interior and boundary points
2. **Buffer preparation** for halo exchange
3. **Start non-blocking communication** (MPI_Isend/MPI_Irecv)
4. **Compute interior points** (no communication needed)
5. **Wait for communication completion** (MPI_Waitall)
6. **Copy halo data** from receive buffers
7. **Compute boundary points** using fresh halo data

#### Cache Optimization:
- **Spatial prefetching**: `_mm_prefetch(ptr, _MM_HINT_T0)`
- **Temporal prefetching**: `_mm_prefetch(ptr, _MM_HINT_T1)`
- **Non-temporal prefetching**: `_mm_prefetch(ptr, _MM_HINT_NTA)`

#### NUMA Optimization:
- **64-byte aligned allocation**: `aligned_alloc(64, size)`
- **First-touch policy**: Threads initialize memory they will use
- **NUMA-local allocation**: OpenMP parallel initialization

---

## 📋 Requirements

### Software Dependencies
- **GCC compiler**: Version 4.9+ with OpenMP support
- **MPI implementation**: OpenMPI 3.0+ or MPICH 3.0+
- **SLURM**: For cluster job submission (optional)
- **bc calculator**: For floating-point arithmetic in scripts

### Hardware Requirements
- **CPU**: Modern x86_64 processor (Intel/AMD)
- **Memory**: Sufficient RAM for problem size (8GB+ recommended)
- **Network**: High-speed interconnect for multi-node runs (InfiniBand preferred)

### Compilation Requirements
```bash
# Check GCC version and OpenMP support
gcc --version
gcc -fopenmp --version

# Check MPI installation
mpicc --version
mpirun --version

# Check SLURM (for cluster execution)
sinfo
squeue
```

---

## 🎯 Usage Recommendations

### For Learning:
1. **Start with serial version** to understand the algorithm
2. **Progress through OpenMP** to learn shared memory parallelization
3. **Explore MPI version** for distributed memory concepts
4. **Study hybrid approach** for real-world HPC applications

### For Performance Analysis:
1. **Use strong scaling** to measure speedup and identify bottlenecks
2. **Use weak scaling** to test scalability limits and memory effects
3. **Compare versions** to understand optimization benefits
4. **Analyze CSV results** with plotting tools (Python, R, Excel)

### For Production Use:
1. **Use optimized hybrid version** (`stencil_parallel_2`) for best performance
2. **Tune MPI tasks vs OpenMP threads** ratio for your hardware
3. **Monitor memory usage** for large problem sizes
4. **Consider NUMA topology** for optimal performance

---

**Author**: Davide Martinelli  
**Course**: High Performance Computing  
**Institution**: University of Trieste - Data Science and Artificial Intelligence  
**Academic Year**: 2024/2025
