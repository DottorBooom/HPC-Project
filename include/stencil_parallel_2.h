/**
 * @file stencil_parallel_2.h
 * @brief Optimized MPI+OpenMP hybrid parallel stencil heat diffusion simulation
 * 
 * This file contains the function prototypes, data structures, constants, and inline 
 * implementations for an advanced 2D heat diffusion simulation using a 5-point stencil 
 * pattern with highly optimized MPI+OpenMP hybrid parallelization. This version implements
 * communication-computation overlap and advanced memory optimization techniques.
 * 
 * **Advanced Optimization Features:**
 * - **Communication-Computation Overlap**: Non-blocking MPI communication allows 
 *   computation of interior points while halo exchange occurs
 * - **Cache Prefetching**: Manual prefetch instructions to optimize memory hierarchy usage
 * - **NUMA-Aware Memory Policy**: Optimized memory allocation for NUMA architectures
 * - **Split Stencil Updates**: Separate functions for interior and border grid points
 * - **Memory Access Optimization**: Aligned allocations and memory access patterns
 * - **Advanced Threading**: OpenMP parallelization with optimized scheduling policies
 * 
 * Key features:
 * - MPI domain decomposition with automatic load balancing
 * - OpenMP thread parallelization within each MPI process
 * - Non-blocking communication with overlapped computation
 * - Hardware-specific optimizations (prefetch, NUMA, alignment)
 * - Support for periodic and non-periodic boundary conditions  
 * - Comprehensive performance analysis and profiling support
 * 
 * @author Davide M.
 * @date 2025
 */

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* See COPYRIGHT in top-level directory. */

#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <sched.h>

#include <omp.h>
#include <mpi.h>

// Hardware optimization includes
#include <xmmintrin.h>  // For cache prefetch instructions (_mm_prefetch)

/* ==========================================================================
   =   Constants and Type Definitions for Optimized MPI+OpenMP             =
   ========================================================================== */

// Direction constants for neighbor identification in 2D grid
#define NORTH 0  // North neighbor (negative Y direction)
#define SOUTH 1  // South neighbor (positive Y direction)
#define EAST  2  // East neighbor (positive X direction)  
#define WEST  3  // West neighbor (negative X direction)

// Communication direction constants
#define SEND 0   // Index for send buffers
#define RECV 1   // Index for receive buffers

// Plane indices for double buffering technique
#define OLD 0    // Current state plane
#define NEW 1    // Updated state plane

// Array dimension indices
#define _x_ 0    // X-dimension index
#define _y_ 1    // Y-dimension index

// MPI communication tags for directional message identification
#define TAG_N 0  // Tag for North-South communication
#define TAG_S 1  // Tag for South-North communication  
#define TAG_E 2  // Tag for East-West communication
#define TAG_W 3  // Tag for West-East communication

typedef unsigned int uint;

// Type definitions for better code readability and maintainability
typedef uint    vec2_t[2];                    // 2D vector for coordinates and sizes
typedef double *restrict buffers_t[4];       // Communication buffers for 4 neighbors

/**
 * @brief Data structure representing a 2D simulation plane
 * 
 * Contains both the data array and dimension information for a simulation grid.
 * Used with double buffering to efficiently update grid values without copying.
 * Memory is allocated with 64-byte alignment for optimal SIMD performance.
 */
typedef struct {
    double *restrict data;  // Pointer to grid data (including ghost cells)
    vec2_t size;           // Interior grid dimensions [width, height]
} plane_t;

/* ==========================================================================
   =   Function Prototypes for Optimized MPI+OpenMP Hybrid Implementation  =
   ========================================================================== */

/**
 * @brief Inject energy into specified sources on the local grid domain
 * 
 * Adds energy to predefined heat sources within the local MPI domain.
 * For periodic boundaries, handles cross-domain energy injection.
 * 
 * @param periodic Flag indicating if periodic boundaries are enabled
 * @param Nsources Number of local heat sources
 * @param Sources Array of local source coordinates
 * @param energy Amount of energy to inject per source
 * @param plane Local simulation plane to inject energy into
 * @param N MPI process grid dimensions [Nx, Ny]
 * @return 0 on success
 */
extern int inject_energy (const int,
                          const int,
			              const vec2_t *,
			              const double,
                          plane_t *,
                          const vec2_t);

/**
 * @brief Update interior grid points using optimized stencil computation
 * 
 * Performs stencil computation on interior points (excluding borders) while
 * halo exchange communication occurs in parallel. This function is designed
 * for communication-computation overlap optimization.
 * 
 * **Optimization Features:**
 * - Hardware prefetching for optimal cache utilization
 * - OpenMP parallelization with static scheduling
 * - Memory access pattern optimization
 * - Temporal and spatial locality exploitation
 * 
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 * 
 * @note Only updates interior points (border points updated separately)
 * @note Designed for overlapping with MPI communication
 * @note Uses hardware-specific cache prefetch instructions
 */
extern int update_internal( const plane_t *,
                            plane_t * );

/**
 * @brief Update border grid points after halo exchange completion
 * 
 * Updates border grid points using data received from neighboring MPI processes.
 * Called after halo exchange communication is complete and ghost cells contain
 * valid neighboring data.
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param N MPI process grid dimensions [Nx, Ny]
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 * 
 * @note Called after halo exchange is complete
 * @note Updates only border points that depend on ghost cells
 * @note Handles periodic boundary conditions for single-process dimensions
 */
extern int update_border( const int,
                          const vec2_t,
                          const plane_t *,
                          plane_t * );

/**
 * @brief Calculate total energy in the local MPI domain
 * 
 * Computes the energy sum for the local MPI domain. Must be combined
 * with MPI_Reduce to get global energy total across all processes.
 * 
 * @param plane Local simulation plane to analyze
 * @param energy Pointer to store the local energy sum
 * @return 0 on success
 */
extern int get_total_energy(plane_t *,
                            double *);

/**
 * @brief Initialize MPI simulation parameters and domain decomposition
 * 
 * Sets up MPI-specific initialization including domain decomposition,
 * neighbor identification, NUMA-aware memory allocation, and heat source distribution.
 * 
 * @param Comm MPI communicator for the simulation
 * @param Me MPI rank of calling process
 * @param Ntasks Total number of MPI processes
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @param S Global grid dimensions
 * @param N Process grid dimensions (output)
 * @param periodic Periodic boundary flag (output)
 * @param output_energy_stat Energy output flag (output)
 * @param neighbours Array of neighbor ranks (output)
 * @param Niterations Number of iterations (output)
 * @param Nsources Total number of sources (output)
 * @param Nsources_local Local number of sources (output)
 * @param Sources_local Local sources array (output)
 * @param energy_per_source Energy per source (output)
 * @param planes Simulation planes (output)
 * @param buffers Communication buffers (output)
 * @return 0 on success, non-zero on error
 */
int initialize (MPI_Comm *,
                int,
                int,
                int,
                char **,
                vec2_t *,
                vec2_t *,
                int *,
                int *,
                int *,
                int *,
                int *,
                int *,
                vec2_t **,
                double *,
                plane_t *,
                buffers_t * );

/**
 * @brief Release allocated memory for planes and communication buffers
 * 
 * Frees memory allocated for simulation planes and MPI communication buffers.
 * 
 * @param planes Array of simulation planes
 * @param buffers Array of communication buffers
 * @return 0 on success
 */
int memory_release (plane_t *planes,
		            buffers_t *buffers);

/**
 * @brief Output energy statistics across all MPI processes
 * 
 * Collects energy data from all processes and outputs global statistics.
 * Only rank 0 performs the actual output.
 * 
 * @param step Current iteration step (-1 for final output)
 * @param plane Local simulation plane
 * @param budget Total injected energy
 * @param Me MPI rank
 * @param Comm MPI communicator
 * @return 0 on success
 */
int output_energy_stat ( int,
                         plane_t *,
                         double,
                         int,
                         MPI_Comm *);

/**
 * @brief Perform integer factorization for domain decomposition
 * 
 * Factors a number into its prime components for optimal MPI grid layout.
 * Used to determine the best process grid dimensions.
 * 
 * @param A Number to factorize
 * @param Nfactors Number of factors (output)
 * @param factors Array of factors (output)
 * @return 0 on success, 1 on error
 */
uint simple_factorization(uint, int *, uint **);

/**
 * @brief Initialize and distribute heat sources across MPI processes
 * 
 * Creates heat sources and assigns them to appropriate MPI processes
 * based on their spatial location within the global domain.
 * 
 * @param Me MPI rank
 * @param Ntasks Total MPI processes
 * @param Comm MPI communicator
 * @param mysize Local domain size
 * @param Nsources Total number of sources
 * @param Nsources_local Local number of sources (output)
 * @param Sources Local sources array (output)
 * @return 0 on success
 */
int initialize_sources(int,
                       int,
                       MPI_Comm *,
                       uint [2],
                       int,
                       int *,
                       vec2_t  **);

/**
 * @brief Allocate NUMA-aware memory for simulation planes and communication buffers
 * 
 * Allocates aligned memory optimized for NUMA architectures with proper
 * memory policies for optimal performance on HPC clusters.
 * 
 * @param neighbours Array of neighbor ranks
 * @param buffers Communication buffers (output)
 * @param planes Simulation planes (output)
 * @return 0 on success, non-zero on error
 */
int memory_allocate (const int *,
		            buffers_t *,
		            plane_t *);

inline int inject_energy (const int periodic,
                          const int Nsources,
			              const vec2_t *Sources,
			              const double energy,
                          plane_t *plane,
                          const vec2_t N)
{
    register const int xsize = plane->size[_x_];
    register const int ysize = plane->size[_y_];
    register uint fxsize = plane->size[_x_]+2;

    double * restrict data = plane->data;

    #define IDX(i, j) ((j) * (fxsize) + (i))
    for (int s = 0; s < Nsources; s++)
    {
        int x = Sources[s][_x_];
        int y = Sources[s][_y_];
        
        data[IDX(x,y)] += energy;
        
        if (periodic)
        {
            if ((N[_x_] == 1))
            {
                data[IDX(0, y)] += data[IDX(xsize + 1, y)]; // West from East
                data[IDX(xsize + 1, y)] += data[IDX(1, y)]; // East from West
            }

            if ((N[_y_] == 1))
            {
                data[IDX(x, 0)] += data[IDX(x, ysize + 1)]; // North from South
                data[IDX(x, ysize + 1)] += data[IDX(x, 1)]; // South from North
            }
        }                
    }
    #undef IDX
    return 0;
}

/**
 * @brief Update interior grid points with hardware prefetch optimization
 * 
 * Performs 5-point stencil computation on interior grid points (excluding borders)
 * while halo exchange communication occurs in parallel. This function is specifically
 * designed for communication-computation overlap optimization.
 * 
 * **Key Optimization Features:**
 * - **Hardware Cache Prefetching**: Uses _mm_prefetch instructions to preload
 *   data into CPU cache before it's needed, reducing memory latency
 * - **Temporal Prefetching**: Prefetches next row data during current row computation
 * - **Spatial Prefetching**: Prefetches future elements in the same row
 * - **Non-temporal Prefetch**: Uses _MM_HINT_NTA for write data that won't be reused
 * - **OpenMP Static Scheduling**: Ensures optimal thread-to-core affinity
 * 
 * **Communication Overlap Strategy:**
 * - Interior points (rows 2 to ysize-1, columns 2 to xsize-1) don't depend on
 *   ghost cells, so they can be computed while MPI halo exchange is ongoing
 * - This achieves perfect communication-computation overlap for large domains
 * 
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 * 
 * @note Only updates interior points (border points updated by update_border)
 * @note Designed for overlapping with non-blocking MPI communication
 * @note Uses Intel SSE prefetch instructions for cache optimization
 * @note Requires OpenMP parallel region to be active when called
 */
inline int update_internal( const plane_t  *oldplane,
                            plane_t *newplane) 
{
    const uint xsize = oldplane->size[_x_];
    const uint ysize = oldplane->size[_y_];
    const uint fxsize = xsize+2;

    #define IDX( i, j ) ( (j)*fxsize + (i) )

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.6;
    const double constant = (1-alpha) / 4.0;
    
    uint i, j;

    // OpenMP parallelization with static scheduling for consistent performance
    #pragma omp parallel for schedule(static)
    for (j = 2; j <= ysize-1; j++) {
        
        // **Row-level prefetching**: Prefetch next row at the beginning of each row
        // This ensures data is in cache when we process the next row
        if (j + 1 <= ysize-1) {
            for (uint prefetch_i = 2; prefetch_i <= xsize-1; prefetch_i += 8) {
                // Prefetch all 5 stencil points for the next row
                _mm_prefetch((char*)&old[IDX(prefetch_i, j+1)], _MM_HINT_T0);     // center
                _mm_prefetch((char*)&old[IDX(prefetch_i-1, j+1)], _MM_HINT_T0);   // west
                _mm_prefetch((char*)&old[IDX(prefetch_i+1, j+1)], _MM_HINT_T0);   // east
                _mm_prefetch((char*)&old[IDX(prefetch_i, j+2)], _MM_HINT_T0);     // south
                if (j > 2) {
                    _mm_prefetch((char*)&old[IDX(prefetch_i, j)], _MM_HINT_T0);   // north for next iteration
                }
            }
        }
        
        for ( i = 2; i <= xsize-1; i++) {
            
            // **Spatial prefetching**: Anticipatory prefetch for future elements in same row
            // Prefetch data 8 elements ahead to account for memory latency
            if (i + 8 <= xsize-1) {
                _mm_prefetch((char*)&old[IDX(i+8, j)], _MM_HINT_T0);     // center 8 ahead
                _mm_prefetch((char*)&old[IDX(i+7, j)], _MM_HINT_T0);     // west 8 ahead
                _mm_prefetch((char*)&old[IDX(i+9, j)], _MM_HINT_T0);     // east 8 ahead
                _mm_prefetch((char*)&old[IDX(i+8, j-1)], _MM_HINT_T0);   // north 8 ahead
                _mm_prefetch((char*)&old[IDX(i+8, j+1)], _MM_HINT_T0);   // south 8 ahead
            }
            
            // **Non-temporal prefetch** for write destination (write-only data)
            // Since new[] values won't be read again immediately, use non-temporal hint
            if (i + 4 <= xsize-1) {
                _mm_prefetch((char*)&new[IDX(i+4, j)], _MM_HINT_NTA);
            }
            
            // **5-point stencil computation** (optimized with const factors)
            const double center = old[ IDX(i,j) ];
            const double neighbors = old[IDX(i-1, j)] + old[IDX(i+1, j)] + 
                                   old[IDX(i, j-1)] + old[IDX(i, j+1)];
            new[ IDX(i,j) ] = center * alpha + neighbors * constant;
        }
    }

    #undef IDX
    return 0;
}

/**
 * @brief Update border grid points after halo exchange completion
 * 
 * Updates border grid points using data received from neighboring MPI processes.
 * Called after halo exchange communication is complete and ghost cells contain
 * valid neighboring data. This function handles only the boundary points that
 * depend on ghost cells received through MPI communication.
 * 
 * **Border Update Strategy:**
 * - Top and bottom borders (rows 1 and ysize) excluding corners
 * - Left and right borders (columns 1 and xsize) including corners
 * - Handles periodic boundary conditions for single-process dimensions
 * 
 * **OpenMP Parallelization:**
 * - Separate parallel loops for horizontal and vertical borders
 * - Static scheduling for optimal thread distribution
 * - No race conditions as border regions are disjoint
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param N MPI process grid dimensions [Nx, Ny]
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 * 
 * @note Called after halo exchange is complete and ghost cells are valid
 * @note Updates only border points that depend on ghost cells
 * @note Handles periodic boundary conditions for single-process dimensions
 * @note Thread-safe OpenMP parallelization with static scheduling
 */
inline int update_border( const int periodic, 
                          const vec2_t N, 
                          const plane_t *oldplane, 
                          plane_t *newplane ) 
{ 
    const uint xsize = oldplane->size[_x_];
    const uint ysize = oldplane->size[_y_];

    const uint fxsize = xsize+2;

    #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    // Update only the border points that depend on ghost cells received via MPI

    const double alpha = 0.6;
    const double constant =  (1-alpha) / 4.0;
    
    double center, neighbors;
    
    uint i, j;
    
    // Update the top and bottom borders (rows 1 and ysize)
    // Exclude corners to avoid race conditions with left/right border updates
    #pragma omp parallel for schedule(static)
    for ( i = 2; i <= xsize-1; i++ ) { // exclude corners
        // Top border (row 1) - depends on ghost row 0
        center = old[ IDX(i,1) ];
        neighbors = old[IDX(i-1, 1)] + old[IDX(i+1, 1)] + old[IDX(i, 0)] + old[IDX(i, 2)];
        new[ IDX(i,1) ] = center * alpha + neighbors * constant;

        // Bottom border (row ysize) - depends on ghost row ysize+1
        center = old[ IDX(i,ysize) ];
        neighbors = old[IDX(i-1, ysize)] + old[IDX(i+1, ysize)] + old[IDX(i, ysize-1)] + old[IDX(i, ysize+1)];
        new[ IDX(i,ysize) ] = center * alpha + neighbors * constant;
    }

    // Update the left and right borders (columns 1 and xsize)
    // Include all rows (including corners) to ensure complete coverage
    #pragma omp parallel for schedule(static)
    for ( j = 1; j <= ysize; j++ ) {
        // Left border (column 1) - depends on ghost column 0
        center = old[ IDX(1,j) ];
        neighbors = old[IDX(0, j)] + old[IDX(2, j)] + old[IDX(1, j-1)] + old[IDX(1, j+1)];
        new[ IDX(1,j) ] = center * alpha + neighbors * constant;

        // Right border (column xsize) - depends on ghost column xsize+1
        center = old[ IDX(xsize,j) ];
        neighbors = old[IDX(xsize-1, j)] + old[IDX(xsize+1, j)] + old[IDX(xsize, j-1)] + old[IDX(xsize, j+1)];
        new[ IDX(xsize,j) ] = center * alpha + neighbors * constant;
    }

    // Handle periodic boundary conditions for single-process dimensions
    if ( periodic ) {
        // If there is only a column of tasks, the periodicity on the X axis is local
        if ( N[_x_] == 1 ) {
            // Copy the values of the first column to the right ghost column (xsize+1)
            for ( j = 1; j <= ysize; j++ ) {
                new[ IDX( 0, j) ]       = new[ IDX(xsize, j) ];
                new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
            }
        }

        // If there is only a row of tasks, the periodicity on the Y axis is local
        if ( N[_y_] == 1 ) {
            // Copy the values of the first row to the bottom ghost row (ysize+1)
            for ( i = 1; i <= xsize; i++ ) {
                new[ IDX( i, 0 ) ]       = new[ IDX(i, ysize) ];
                new[ IDX( i, ysize+1) ] = new[ IDX(i, 1) ];
            }
        }
    }

    #undef IDX
    return 0;
}/**
 * @brief Calculate total energy in the local MPI domain with OpenMP reduction
 * 
 * Computes the energy sum for the local MPI domain by summing all interior
 * grid points. This local result must be combined with MPI_Reduce across
 * all processes to obtain the global energy total. The function uses 
 * OpenMP parallelization for efficient computation within each MPI process.
 * 
 * **Hybrid Parallelization:**
 * - MPI: Each process calculates local energy sum for its domain
 * - OpenMP: Threads within each process parallelize the summation
 * - Global total requires MPI_Reduce to combine all local results
 * 
 * **Performance Optimizations:**
 * - Collapsed nested loops for better thread work distribution
 * - OpenMP reduction for thread-safe accumulation
 * - Static scheduling for consistent performance
 * - Optional long double precision for numerical stability
 * 
 * @param plane Local simulation plane to analyze
 * @param energy Pointer to store the local energy sum
 * @return 0 on success
 * 
 * @note This function computes only local energy - use MPI_Reduce for global total
 * @note OpenMP reduction ensures thread-safe accumulation within each MPI process
 * @note Can use LONG_ACCURACY for higher precision calculations
 */
inline int get_total_energy( plane_t *plane,
                             double  *energy )
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{

    register const int xsize = plane->size[_x_];
    register const int ysize = plane->size[_y_];
    register const int fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i))

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    #pragma omp parallel for collapse(2) reduction(+:totenergy) schedule(static)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}