/**
 * @file stencil_serial.h
 * @brief Header file for serial stencil heat diffusion simulation
 * 
 * This file contains the function prototypes, constants, and inline implementations
 * for a 2D heat diffusion simulation using a 5-point stencil pattern.
 * The simulation supports both periodic and non-periodic boundary conditions.
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
#include <float.h>
#include <math.h>

#include <omp.h>

/* ==========================================================================
   =   Constants and Definitions                                            =
   ========================================================================== */

// Direction constants for boundary handling
#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

// Communication types (for future MPI extensions)
#define SEND 0
#define RECV 1

// Plane indices for double buffering
#define OLD 0  // Current state plane
#define NEW 1  // Updated state plane

// Array dimension indices
#define _x_ 0  // X-dimension index
#define _y_ 1  // Y-dimension index

/* ==========================================================================
   =   Function Prototypes                                                  =
   ========================================================================== */

/**
 * @brief Initialize simulation parameters from command line arguments
 * 
 * Parses command line arguments and sets up all simulation parameters including
 * grid dimensions, boundary conditions, number of iterations, energy sources,
 * and timing options. Also allocates memory for simulation planes and heat sources.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param S Array containing grid dimensions [x_size, y_size]
 * @param periodic Pointer to periodic boundary condition flag
 * @param Niterations Pointer to number of simulation iterations
 * @param Nsources Pointer to number of heat sources
 * @param Sources Pointer to array of heat source coordinates
 * @param energy_per_source Pointer to energy amount per source
 * @param planes Pointer to simulation planes array
 * @param output_energy_at_steps Pointer to energy output flag
 * @param injection_frequency Pointer to energy injection frequency
 * @return 0 on success, 1 on memory allocation error
 */
int initialize (int,
                char **,
                unsigned int *,
                int *,
                int *,
                int *,
                int **,
                double *,
                double **,
                int *,
                int *);

/**
 * @brief Release allocated memory for simulation data
 * 
 * Frees memory allocated for simulation planes and heat source arrays.
 * Should be called at the end of the simulation to prevent memory leaks.
 * 
 * @param data Pointer to simulation plane data
 * @param sources Pointer to heat sources array
 * @return 0 on success
 */
int memory_release (double *, int *);


/**
 * @brief Inject energy into specified sources on the simulation grid
 * 
 * Adds energy to predefined heat sources on the grid. For periodic boundaries,
 * also handles energy injection at boundary locations to maintain continuity.
 * 
 * @param periodic Flag indicating if periodic boundaries are enabled
 * @param Nsources Number of heat sources
 * @param Sources Array of source coordinates (x,y pairs)
 * @param energy Amount of energy to inject per source
 * @param mysize Grid dimensions [x_size, y_size]
 * @param plane Simulation plane to inject energy into
 * @return 0 on success
 */
extern int inject_energy (  int,
                            int,
                            int *,
                            double,
                            unsigned int [2],
                            double *); 

/**
 * @brief Update the simulation grid using 5-point stencil operation
 * 
 * Performs the main heat diffusion calculation using a 5-point stencil pattern.
 * Updates all interior grid points based on their neighbors. Also handles
 * periodic boundary conditions if enabled and measures computation time.
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param size Grid dimensions [x_size, y_size]
 * @param old Current state plane (input)
 * @param new Updated state plane (output)
 * @param time_spent Pointer to store computation time
 * @param thread_times Array to store per-thread timing information
 * @return 0 on success
 */
extern int update_plane (int,
                        unsigned int [2],
                        double *,
                        double *,
                        double *,
                        double *); 

/**
 * @brief Calculate total energy in the simulation grid
 * 
 * Sums up all energy values in the interior grid points to compute
 * the total system energy. Used for energy conservation verification.
 * 
 * @param size Grid dimensions [x_size, y_size]
 * @param plane Simulation plane to analyze
 * @param energy Pointer to store the calculated total energy
 * @return 0 on success
 */
extern int get_total_energy(unsigned int [2],
                            double *,
                            double *);


/**
 * @brief Write simulation data to binary file
 * 
 * Exports the current state of the simulation grid to a binary file.
 * Can be used for visualization or debugging purposes.
 * 
 * @param data Pointer to simulation data
 * @param size Grid dimensions [x_size, y_size]
 * @param filename Output filename
 * @param min Pointer to store minimum value (optional)
 * @param max Pointer to store maximum value (optional)
 * @return 0 on success, 1 on invalid filename, 2 on file open error
 */
int dump (const double *, 
        unsigned int [2], 
        const char *, 
        double *, 
        double * );

/**
 * @brief Allocate memory for simulation planes
 * 
 * Allocates contiguous memory for two simulation planes using double buffering.
 * The planes are used to store current and updated states during simulation.
 * 
 * @param size Grid dimensions [x_size, y_size]
 * @param planes Double pointer to store allocated plane addresses
 * @return 0 on success, 1 on allocation failure
 */
int memory_allocate (unsigned int [2],
		            double **);

/**
 * @brief Initialize heat source positions randomly
 * 
 * Creates an array of randomly distributed heat sources on the simulation grid.
 * Each source is defined by a pair of (x,y) coordinates within the grid bounds.
 * 
 * @param size Grid dimensions [x_size, y_size]
 * @param Nsources Number of heat sources to create
 * @param Sources Pointer to store allocated sources array
 * @return 0 on success, 1 on allocation failure
 */
int initialize_sources(unsigned int [2],
			        int ,
			        int **);

/**
 * @brief Export timing data and generate performance plots
 * 
 * Writes timing measurements to CSV file and generates GNU plot script
 * for visualizing performance characteristics of the simulation.
 * 
 * @param time_spent_injecting Array of energy injection times per iteration
 * @param time_spent_updating Array of grid update times per iteration
 * @param time_spent_simulation Array of total simulation times per iteration
 * @param Niterations Number of simulation iterations
 */
void export_and_plot(double *, 
                    double *, 
                    double *,
                    int);

/* ==========================================================================
   =   Inline Function Implementations                                     =
   ========================================================================== */

/**
 * @brief Inject energy into heat sources on the simulation grid
 * 
 * Adds specified amount of energy to predefined heat sources distributed
 * across the simulation grid. For periodic boundary conditions, also handles
 * energy injection at boundary ghost cells to maintain continuity.
 * 
 * @param periodic Flag indicating if periodic boundaries are enabled
 * @param Nsources Number of heat sources to inject energy into
 * @param Sources Array containing coordinates of heat sources (x,y pairs)
 * @param energy Amount of energy to inject per source
 * @param mysize Grid dimensions [x_size, y_size]
 * @param plane Pointer to simulation plane data
 * @return 0 on success
 * 
 * @note This function is not thread-safe when used with overlapping sources
 */
inline int inject_energy (  int periodic,
                            int Nsources,
			                      int *Sources,
			                      double energy,
			                      unsigned int mysize[2],
                            double *plane)
{
    #define IDX( i, j ) ( (j)*(mysize[_x_]+2) + (i) )
    
    // Iterate through all heat sources and inject energy
    // For low number of sources, simple loop is efficient
    // For high number of sources, parallelization might be considered
    for (int s = 0; s < Nsources; s++) {
        
        // Extract source coordinates from the sources array
        unsigned x = Sources[2*s];     // x-coordinate of source
        unsigned y = Sources[2*s+1];   // y-coordinate of source
        
        // Inject energy at the source location
        plane[IDX(x, y)] += energy;

        // Handle periodic boundary conditions
        // When a source is at the boundary, also inject energy
        // at the corresponding ghost cells to maintain periodicity
        if (periodic)
        {
            // Left boundary: if source is at x=1, also inject at right ghost cell
            if (x == 1)
                plane[IDX(mysize[_x_]+1, y)] += energy;
            // Right boundary: if source is at x=max, also inject at left ghost cell    
            if (x == mysize[_x_])
                plane[IDX(0, y)] += energy;
            // Bottom boundary: if source is at y=1, also inject at top ghost cell
            if (y == 1)
                plane[IDX(x, mysize[_y_]+1)] += energy;
            // Top boundary: if source is at y=max, also inject at bottom ghost cell
            if (y == mysize[_y_])
                plane[IDX(x, 0)] += energy;
        }
    }

    #undef IDX
    return 0;
}

/**
 * @brief Update simulation grid using 5-point stencil heat diffusion
 * 
 * Performs the main heat diffusion calculation using a 5-point stencil pattern.
 * Each interior grid point is updated based on its four neighbors using a weighted
 * average that simulates heat conduction. The function also handles periodic
 * boundary conditions and measures computation time for performance analysis.
 * 
 * @param periodic Flag for periodic boundary conditions
 * @param size Grid dimensions [x_size, y_size]
 * @param old Current state plane (input) 
 * @param new Updated state plane (output)
 * @param time_spent Pointer to store total computation time
 * @param thread_times Array to store per-thread timing for load balancing analysis
 * @return 0 on success
 * 
 * @note This function is parallelized with OpenMP and includes load balancing measurement
 */
inline int update_plane (   int periodic,
                            unsigned int size[2],
			                double *old,
                            double *new,
                            double *time_spent,
                            double *thread_times)
{
    register const unsigned fxsize = size[_x_]+2;  // Full grid width with ghost cells
    //register const unsigned fysize = size[_y_]+2; // Full grid height with ghost cells
    register const unsigned xsize = size[_x_];     // Interior grid width
    register const unsigned ysize = size[_y_];     // Interior grid height

    #define IDX( i, j ) ( (j)*fxsize + (i) )

    // Start timing the computation phase
    double begin = omp_get_wtime();

    #pragma omp parallel
    {
        // Get thread ID for timing measurements
        int tid = omp_get_thread_num();
        double t_start = omp_get_wtime();

    // Parallelize the nested loops with collapse and static scheduling
    #pragma omp for collapse(2) schedule(static)
    for (unsigned int j = 1; j <= ysize; j++)
        for (unsigned int i = 1; i <= xsize; i++)
        {
            // 5-point stencil heat diffusion formula
            // Uses a simplified stencil without explicit diffusivity coefficient
            // The formula conserves the total energy in the system
            // Alpha parameter controls heat transfer rate (0.6 = moderate diffusion)
            
            double alpha = 0.6;
            double result = old[ IDX(i,j) ] * alpha;
            double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]) / 4.0 * (1-alpha);
            double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]) / 4.0 * (1-alpha);
            result += (sum_i + sum_j );

            new[ IDX(i,j) ] = result;
        }
        
        // Record per-thread computation time for load balancing analysis
        double t_end = omp_get_wtime();
        thread_times[tid] = t_end - t_start;
    }
    
    // Record total computation time
    double end = omp_get_wtime();
    *time_spent = end - begin;
        
    // Handle periodic boundary conditions
    // Copy boundary values to ghost cells to maintain continuity
    if ( periodic )
    {
        // Vertical boundaries: top <-> bottom
        for (unsigned int i = 1; i <= xsize; i++ )
            {
                new[ i ] = new[ IDX(i, ysize) ];           // Copy bottom to top ghost
                new[ IDX(i, ysize+1) ] = new[ i ];         // Copy top to bottom ghost
            }
        // Horizontal boundaries: left <-> right  
        for (unsigned int j = 1; j <= ysize; j++ )
            {
                new[ IDX( 0, j) ] = new[ IDX(xsize, j) ];  // Copy right to left ghost
                new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ]; // Copy left to right ghost
            }
    }

    #undef IDX
    return 0;
}

/**
 * @brief Calculate total energy in the simulation grid
 * 
 * Sums up all energy values in the interior grid points to compute
 * the total system energy. Used for energy conservation verification
 * and debugging purposes. The function is parallelized with OpenMP
 * for efficient computation on large grids.
 * 
 * @param size Grid dimensions [x_size, y_size]
 * @param plane Simulation plane to analyze
 * @param energy Pointer to store the calculated total energy
 * @return 0 on success
 * 
 * @note This function is a good candidate for OpenMP parallelization
 * @note Can use LONG_ACCURACY for higher precision calculations
 */
inline int get_total_energy(unsigned int size[2],
                            double *plane,
                            double *energy)
{
    register const int xsize = size[_x_];
    
    #define IDX( i, j ) ( (j)*(xsize+2) + (i) )

    // Use long double for higher precision if enabled
    #if defined(LONG_ACCURACY)    
        long double totenergy = 0;
    #else
        double totenergy = 0;    
    #endif

    #pragma omp parallel for collapse(2) reduction(+:totenergy) schedule(static)
    for (unsigned int j = 1; j <= size[_y_]; j++ )
        for (unsigned int i = 1; i <= size[_x_]; i++ )
            totenergy += plane[ IDX(i, j) ];
    
    #undef IDX

    *energy = (double)totenergy;
    return 0;
}

/* ==========================================================================
   =   Initialization of Variables and Memory Allocation                    =
   ========================================================================== */

/**
 * @brief Parse command line arguments and initialize simulation parameters
 * 
 * Processes command line options to set up simulation configuration including
 * grid dimensions, boundary conditions, number of iterations, heat sources,
 * and timing parameters. Also allocates memory for simulation data structures.
 * 
 * @param argc Number of command line arguments
 * @param argv Command line arguments array
 * @param S Output array for grid dimensions [x_size, y_size]
 * @param periodic Output flag for periodic boundary conditions
 * @param Niterations Output number of simulation iterations
 * @param Nsources Output number of heat sources
 * @param Sources Output pointer to heat sources array
 * @param energy_per_source Output energy amount per source
 * @param planes Output pointer to simulation planes
 * @param output_energy_at_steps Output flag for energy reporting
 * @param injection_frequency Output frequency of energy injection
 * @return 0 on success, non-zero on error
 * 
 * Supported command line options:
 * - -x: x dimension of the grid (default: 1000)
 * - -y: y dimension of the grid (default: 1000)  
 * - -e: number of energy sources (default: 1)
 * - -E: energy per source (default: 1.0)
 * - -f: injection frequency (default: 0.0)
 * - -n: number of iterations (default: 99)
 * - -p: periodic boundaries flag (default: 0)
 * - -o: output energy at steps flag (default: 0)
 */
int initialize (int argc,                 // the argc from command line
                char **argv,              // the argv from command line
                unsigned int *S,          // two-unsigned-int array defining the x,y dimensions of the grid
                int *periodic,            // periodic-boundary tag
                int *Niterations,         // how many iterations
                int *Nsources,            // how many heat sources
                int **Sources,
                double *energy_per_source,// how much heat per source
                double **planes,
                int *output_energy_at_steps,
                int *injection_frequency)
{
  int ret;
  
  // Set default values for all simulation parameters

  S[_x_]            = 1000;
  S[_y_]            = 1000;
  *periodic         = 0;
  *Nsources         = 1;
  *Niterations      = 99;
  *output_energy_at_steps = 0;
  *energy_per_source = 1.0;
  *injection_frequency = *Niterations;

  double freq = 0;

  // Process command line arguments using getopt

  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":x:y:e:E:f:n:p:o:")) != -1)
    {
      switch( opt )
      {
        case 'x': S[_x_] = (unsigned int)atoi(optarg);
          break;

        case 'y': S[_y_] = (unsigned int)atoi(optarg);
          break;

        case 'e': *Nsources = atoi(optarg);
          break;

        case 'E': *energy_per_source = atof(optarg);
          break;

        case 'n': *Niterations = atoi(optarg);
          break;

        case 'p': *periodic = (atoi(optarg) > 0);
          break;

        case 'o': *output_energy_at_steps = (atoi(optarg) > 0);
          break;

        case 'f': freq = atof(optarg);
          break;
          
        case 'h': printf( "valid options are ( values btw [] are the default values ):\n"
              "-x    x size of the plate [1000]\n"
              "-y    y size of the plate [1000]\n"
              "-e    how many energy sources on the plate [1]\n"
              "-E    how many energy sources on the plate [1.0]\n"
              "-f    the frequency of energy injection [0.0]\n"
              "-n    how many iterations [100]\n"
              "-p    whether periodic boundaries applies  [0 = false]\n"
              "-o    whether to print the energy budgest at every step [0 = false]\n"
              );
          break;
          

        case ':': printf( "option -%c requires an argument\n", optopt);
          break;

        case '?': printf(" -------- help unavailable ----------\n");
          break;
      }
    }
    if ( opt == -1 )
      break;
  }

  // Calculate injection frequency based on provided frequency value
  if ( freq == 0 )
    *injection_frequency = 1.0;
  else
  {
    freq = (freq > 1.0 ? 1.0 : freq );
    *injection_frequency = freq * *Niterations;
  }

  // Allocate memory for simulation planes
  ret = memory_allocate( S, planes ); 
  if ( ret != 0 )
    return ret;

  // Allocate and initialize heat sources
  ret = initialize_sources( S, *Nsources, Sources );
  if ( ret != 0 )
    return ret;

  return 0;
}


int memory_allocate ( unsigned int size[2],
		                  double **planes_ptr )
/**
 * @brief Allocate memory for simulation planes
 * 
 * Allocates memory for two simulation planes using double buffering technique.
 * The first plane contains the current data, the second contains the updated data.
 * During the integration loop, the roles are swapped at every iteration for
 * efficient memory usage without copying data.
 * 
 * @param size Grid dimensions [x_size, y_size] 
 * @param planes_ptr Double pointer to store plane addresses
 * @return 0 on success, 1 on error
 */
{
  if (planes_ptr == NULL ){
    // an invalid pointer has been passed
    // manage the situation
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  unsigned int bytes = (size[_x_]+2)*(size[_y_]+2);

  planes_ptr[OLD] = (double*)malloc( 2*bytes*sizeof(double) );
  memset ( planes_ptr[OLD], 0, 2*bytes*sizeof(double) );
  planes_ptr[NEW] = planes_ptr[OLD] + bytes;
      
  return 0;
}


int initialize_sources( unsigned int size[2],
			                  int Nsources,
			                  int **Sources)
/**
 * @brief Initialize heat source positions randomly
 * 
 * Creates an array of randomly distributed heat sources on the simulation grid.
 * Each source is defined by a pair of (x,y) coordinates within the grid bounds.
 * Uses lrand48() for random number generation to ensure reproducible results
 * when seeded appropriately.
 * 
 * @param size Grid dimensions [x_size, y_size]
 * @param Nsources Number of heat sources to create
 * @param Sources Pointer to store allocated sources array
 * @return 0 on success, 1 on allocation failure
 */
{
  *Sources = (int*)malloc( Nsources * 2 *sizeof(unsigned int) );
  for ( int s = 0; s < Nsources; s++ )
  {
    (*Sources)[s*2] = 1+ lrand48() % size[_x_];
    (*Sources)[s*2+1] = 1+ lrand48() % size[_y_];
  }
  return 0;
}

/* ==========================================================================
   =   Memory Release                                                       =
   ========================================================================== */

int memory_release ( double *data, int *sources )  
{
  if( data != NULL )
    free( data );

  if( sources != NULL )
    free( sources );

  return 0;
}

int dump ( const double *data, unsigned int size[2], const char *filename, double *min, double *max )
{
  if ( (filename != NULL) && (filename[0] != '\0') )
  {
    FILE *outfile = fopen( filename, "w" );
    if ( outfile == NULL )
	    return 2;
      
    float *array = (float*)malloc( size[0] * sizeof(float) );
      
    double _min_ = DBL_MAX;
    double _max_ = 0;

    for (unsigned int j = 0; j < size[1]; j++ )
	  {
      /*
      float y = (float)j / size[1];
      fwrite ( &y, sizeof(float), 1, outfile );
      */
      
      const double * restrict line = data + j*size[0];
      for (unsigned int i = 0; i < size[0]; i++ )
      {
        array[i] = (float)line[i];
        _min_ = ( line[i] < _min_? line[i] : _min_ );
        _max_ = ( line[i] > _max_? line[i] : _max_ ); 
      }
      
      fwrite( array, sizeof(float), size[0], outfile );
	  }
      
    free( array );
      
    fclose( outfile );

    if ( min != NULL )
	    *min = _min_;
    if ( max != NULL )
	    *max = _max_;
  }

  else return 1;

  return 0;
}

/* ==========================================================================
   =   Plot                                                                 =
   ========================================================================== */

void export_and_plot(double *injecting, double *updating, double *simulation, int Niterations) {
    FILE *fp = fopen("timing_data.csv", "w");
    fprintf(fp, "iteration,injecting,updating,total\n");

    double accI = 0.0, accU = 0.0, accS = 0.0;
    for (int i = 0; i < Niterations; i++) {
        accI += injecting[i];
        accU += updating[i];
        accS += simulation[i];
        fprintf(fp, "%d,%.8f,%.8f,%.8f\n", i, accI, accU, accS);
    }
    fclose(fp);
    
    FILE *gp = fopen("plot.gp", "w");
    fprintf(gp,
        "set datafile separator \",\"\n"
        "set title \"Cumulative Execution Time\"\n"
        "set xlabel \"Iteration\"\n"
        "set ylabel \"Cumulative Time (s)\"\n"
        "set grid\n"
        "set key outside\n"
        "plot \"timing_data.csv\" using 1:2 with lines lw 2 title \"Injecting\", \\\n"
        "     \"timing_data.csv\" using 1:3 with lines lw 2 title \"Updating\", \\\n"
        "     \"timing_data.csv\" using 1:4 with lines lw 2 title \"Total\"\n"
        "pause -1 \"Press Enter to close\"\n");
    fclose(gp);

    system("gnuplot plot.gp");
}