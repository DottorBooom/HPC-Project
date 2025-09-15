/**
 * @file stencil_parallel.h
 * @brief Header file for MPI+OpenMP hybrid parallel stencil heat diffusion simulation
 * 
 * This file contains the function prototypes, data structures, constants, and inline 
 * implementations for a 2D heat diffusion simulation using a 5-point stencil pattern
 * with MPI+OpenMP hybrid parallelization. The simulation combines distributed memory 
 * parallelism (MPI) with shared memory parallelism (OpenMP) for optimal performance 
 * on modern HPC clusters.
 * 
 * Key features:
 * - MPI domain decomposition with automatic load balancing
 * - OpenMP thread parallelization within each MPI process
 * - Halo exchange communication for boundary data between MPI processes
 * - Support for periodic and non-periodic boundary conditions  
 * - Optimized communication patterns with non-blocking MPI
 * - Thread-safe operations with proper OpenMP synchronization
 * - Comprehensive performance analysis including thread load balancing
 * 
 * @author Davide M.
 * @date 2025
 */

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

/* ==========================================================================
   =   Constants and Type Definitions for MPI Parallelization              =
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
 */
typedef struct {
    double *restrict data;  // Pointer to grid data (including ghost cells)
    vec2_t size;           // Interior grid dimensions [width, height]
} plane_t;

/* ==========================================================================
   =   Function Prototypes for MPI-Parallel Implementation                 =
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
 * @brief Update local grid domain using 5-point stencil operation
 * 
 * Performs the main heat diffusion calculation for the local MPI domain.
 * Assumes halo exchange has already been completed for boundary data.
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param N MPI process grid dimensions [Nx, Ny]
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 */
extern int update_plane (const int,
                         const vec2_t,
                         const plane_t *,
                         plane_t *);

/**
 * @brief Calculate total energy in the local grid domain
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
 * neighbor identification, memory allocation, and heat source distribution.
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
 * @brief Allocate memory for simulation planes and communication buffers
 * 
 * Allocates aligned memory for simulation data and MPI communication buffers.
 * Sets up the halo regions needed for inter-process communication.
 * 
 * @param neighbours Array of neighbor ranks
 * @param buffers Communication buffers (output)
 * @param planes Simulation planes (output)
 * @return 0 on success, non-zero on error
 */
int memory_allocate (const int *,
		            buffers_t *,
		            plane_t *);

/**
 * @brief Inject energy into heat sources within the local MPI domain
 * 
 * Adds specified amount of energy to predefined heat sources distributed
 * within the local MPI process domain. For periodic boundary conditions,
 * handles energy injection at ghost cells when the global domain has
 * periodic boundaries but only one MPI process in that dimension.
 * 
 * @param periodic Flag indicating if periodic boundaries are enabled globally
 * @param Nsources Number of heat sources local to this MPI process
 * @param Sources Array containing local coordinates of heat sources (x,y pairs)
 * @param energy Amount of energy to inject per source
 * @param plane Pointer to local simulation plane data
 * @param N MPI process grid dimensions [Nx, Ny]
 * @return 0 on success
 * 
 * @note This function operates only on local domain data
 * @note Periodic boundary handling only applies when N[dim] == 1
 */
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
    
    // Inject energy at each local heat source
    for (int s = 0; s < Nsources; s++)
    {
        int x = Sources[s][_x_];  // Local x-coordinate within this MPI domain
        int y = Sources[s][_y_];  // Local y-coordinate within this MPI domain
        
        // Inject energy at the source location
        data[IDX(x,y)] += energy;
        
        // Handle periodic boundary conditions for single-process dimensions
        // This is only needed when the entire global domain fits in one process
        // in a particular dimension (N[dim] == 1)
        if (periodic)
        {
            // X-direction periodic boundaries (only if single process in X)
            if ((N[_x_] == 1))
            {
                data[IDX(0, y)] += data[IDX(xsize + 1, y)]; // West from East
                data[IDX(xsize + 1, y)] += data[IDX(1, y)]; // East from West
            }

            // Y-direction periodic boundaries (only if single process in Y)
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
 * @brief Update local grid domain using 5-point stencil heat diffusion
 * 
 * Performs the main heat diffusion calculation for the local MPI domain using
 * a 5-point stencil pattern. Each interior grid point is updated based on its
 * four neighbors. This function assumes that halo exchange has already been
 * completed, so ghost cells contain valid data from neighboring MPI processes.
 * 
 * **Hybrid Parallelization:**
 * - MPI: Each process works on its local domain partition
 * - OpenMP: Threads within each MPI process parallelize the stencil computation
 * - The #pragma omp for directive distributes loop iterations among threads
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param N MPI process grid dimensions [Nx, Ny]
 * @param oldplane Current state plane (input)
 * @param newplane Updated state plane (output)
 * @return 0 on success
 * 
 * @note Assumes halo exchange is complete before calling
 * @note Periodic boundaries only handled for single-process dimensions
 * @note OpenMP threads work within OpenMP parallel region (not created here)
 * @note Thread-safe: each thread works on disjoint grid regions
 */
inline int update_plane ( const int periodic, 
                          const vec2_t N,         // MPI process grid dimensions
                          const plane_t *oldplane,
                          plane_t *newplane)
{
    register uint fxsize = oldplane->size[_x_]+2;  // Full grid width with halos
    
    register uint xsize = oldplane->size[_x_];     // Local interior width
    register uint ysize = oldplane->size[_y_];     // Local interior height

    #define IDX( i, j ) ( (j)*fxsize + (i))
    

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    // OpenMP parallelization within each MPI process
    // - The calling context should already be in an OpenMP parallel region
    // - This directive distributes loop iterations among available threads
    // - 'nowait' prevents implicit barrier, allowing thread reuse
    #pragma omp for collapse(2) schedule(static) nowait
    for (uint j = 1; j <= ysize; j++) {
        for (uint i = 1; i <= xsize; i++) 
        {
            // 5-point stencil heat diffusion formula
            // Alpha controls the heat retention vs. diffusion rate
            double alpha = 0.6;
            double result = old[IDX(i,j)] * alpha;
            double sum_i = (old[IDX(i-1, j)] + old[IDX(i+1, j)]) / 4.0 * (1 - alpha);
            double sum_j = (old[IDX(i, j-1)] + old[IDX(i, j+1)]) / 4.0 * (1 - alpha);
            result += (sum_i + sum_j);
            new[IDX(i,j)] = result;
        }
    }

    // Handle periodic boundary conditions for single-process dimensions
    // This is only needed when the entire global domain fits in one MPI process
    // in a particular dimension
    if (periodic) {
        // X-direction: only if there's a single process in X dimension
        if (N[_x_] == 1) {
            for (uint i = 1; i <= xsize; i++) {
                new[i] = new[IDX(i, ysize)];                // Top ghost from bottom
                new[IDX(i, ysize+1)] = new[i];              // Bottom ghost from top
            }
        }
        // Y-direction: only if there's a single process in Y dimension  
        if (N[_y_] == 1) {
            for (uint j = 1; j <= ysize; j++) {
                new[IDX(0, j)] = new[IDX(xsize, j)];        // Left ghost from right
                new[IDX(xsize+1, j)] = new[IDX(1, j)];      // Right ghost from left
            }
        }
    }
    #undef IDX
    return 0;
}

/**
 * @brief Calculate total energy in the local MPI domain
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
{
    register const int xsize = plane->size[_x_];
    register const int ysize = plane->size[_y_];
    register const int fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i))

   // Use long double for higher precision if enabled
   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    #pragma omp parallel for collapse(2) reduction(+:totenergy) schedule(static)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];
    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}