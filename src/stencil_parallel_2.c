/**
 * @file stencil_parallel_2.c
 * @brief Optimized MPI+OpenMP hybrid parallel implementation with communication-computation overlap
 * 
 * This program represents an advanced implementation of 2D heat diffusion simulation using
 * highly optimized MPI+OpenMP hybrid parallelization. The key innovation is the separation
 * of stencil computation into interior and border updates, enabling perfect overlap of 
 * MPI communication with computation.
 * 
 * **Advanced Optimization Strategy:**
 * - **Split Stencil Approach**: Interior points (independent of ghost cells) are updated
 *   while non-blocking MPI communication transfers halo data
 * - **Communication-Computation Overlap**: Achieves near-perfect overlap by computing
 *   interior points during halo exchange
 * - **Hardware-Specific Optimizations**: Cache prefetching, NUMA-aware memory allocation,
 *   and aligned data structures for optimal performance
 * 
 * **Implementation Workflow:**
 * 1. **Energy Injection**: Add heat to local sources
 * 2. **Buffer Preparation**: Fill send buffers for halo exchange
 * 3. **Non-blocking Communication**: Start MPI_Isend/MPI_Irecv for halo data
 * 4. **Interior Update**: Compute interior points while communication occurs
 * 5. **Communication Completion**: Wait for MPI operations to finish
 * 6. **Halo Copy**: Copy received data to ghost cells
 * 7. **Border Update**: Update border points using received halo data
 * 
 * **Performance Benefits:**
 * - **Communication Hiding**: Nearly eliminates communication overhead for large domains
 * - **Cache Optimization**: Prefetch instructions improve memory hierarchy utilization
 * - **NUMA Awareness**: Memory allocation optimized for NUMA architectures
 * - **Thread Efficiency**: OpenMP scheduling optimized for HPC environments
 * 
 * Usage: mpirun -n <nprocs> ./stencil_parallel_2 [options]
 * Environment: Set OMP_NUM_THREADS=<nthreads> for OpenMP thread count
 * Options:
 *   -x size      Grid width (default: 1000)
 *   -y size      Grid height (default: 1000)
 *   -n iter      Number of iterations (default: 100)
 *   -e sources   Number of heat sources (default: 5)
 *   -E energy    Energy per source (default: 1.0)
 *   -p flag      Periodic boundaries (default: 0)
 *   -o flag      Output energy at each step (default: 0)
 *   -v level     Verbosity level (default: 0)
 * 
 * @author Davide M.
 * @date 2025
 */
#include "stencil_parallel_2.h"

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD;
  int Rank, Ntasks;
  int neighbours[4];

  int Niterations;
  int periodic;
  vec2_t S, N; // S : global size of the plate, N : grid decomposition of MPI tasks

  int Nsources;
  int Nsources_local;
  vec2_t *Sources_local;
  double energy_per_source;

  plane_t planes[2];
  buffers_t buffers[2];
  
  int output_energy_stat_perstep = 0;

  double comm_time = 0.0, comp_time = 0.0, total_time, init_time, wait_time;
	double start_time_comm, start_time_comp;
  
  /* initialize MPI envrionment */
  {
    int level_obtained;
    
    // NOTE: change MPI_FUNNELED if appropriate
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained );
    if ( level_obtained < MPI_THREAD_FUNNELED ) 
    {
      printf("MPI_thread level obtained is %d instead of %d\n",
	    level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1); 
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD);
  }

  init_time = MPI_Wtime();
  /* argument checking and setting */
  int ret = initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, 
                       &S, &N, &periodic, &output_energy_stat_perstep,
			                 neighbours, &Niterations, &Nsources, &Nsources_local, 
                       &Sources_local, &energy_per_source, &planes[0], &buffers[0]);

  /* After calling initialize():
   *
   * myCOMM_WORLD         : Duplicated MPI communicator for this job
   * Rank                 : Rank of this MPI process (0 ... Ntasks-1)
   * Ntasks               : Total number of MPI processes
   * neighbours[4]        : Ranks of neighboring processes (NORTH, EAST, SOUTH, WEST), or MPI_PROC_NULL if absent
   * Niterations          : Number of simulation iterations
   * periodic             : 1 if periodic boundaries are enabled, 0 otherwise
   * S                    : Global grid size [X, Y]
   * N                    : Process grid decomposition [Nx, Ny]
   * Nsources             : Total number of heat sources
   * Nsources_local       : Number of sources assigned to this process
   * Sources_local        : Array of local source coordinates (size Nsources_local)
   * energy_per_source    : Energy injected per source
   * planes[2]            : Two data planes (OLD/NEW), each allocated with size (mysize[0]+2) x (mysize[1]+2)
   * buffers[2]           : Communication buffers for neighbors (SEND/RECV for N, S, E, W)
   * output_energy_stat_perstep : If >0, print energy at every step
   */

  if ( ret )
    {
      printf("task %d is opting out with termination code %d\n",
	     Rank, ret );
      
      MPI_Finalize();
      return 0;
    }
  
  
  int current = OLD;

  uint ysize = planes[current].size[_y_];
	uint xsize = planes[current].size[_x_];

  init_time = MPI_Wtime() - init_time;
	total_time = MPI_Wtime();

  #define IDX(i, j) ((j) * (xsize + 2) + (i))
  
  /* Main simulation loop with communication-computation overlap optimization */
  for (int iter = 0; iter < Niterations; ++iter)
  {
    // Initialize MPI request array for non-blocking communication
    MPI_Request reqs[8];
    int nreqs = 0;
    for (int i = 0; i < 8; ++i)
      reqs[i] = MPI_REQUEST_NULL;
    
    /* Phase 1: Energy injection - inject energy into local heat sources */
    inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);

    /* Phase 2: Buffer preparation for halo exchange */
    // [A] Fill the send buffers for east/west neighbors (non-contiguous data)
    if (neighbours[WEST] != MPI_PROC_NULL && buffers[SEND][WEST] != NULL) 
        for (uint j = 0; j < ysize; ++j) 
            buffers[SEND][WEST][j] = planes[current].data[IDX(1, j+1)];

    if (neighbours[EAST] != MPI_PROC_NULL && buffers[SEND][EAST] != NULL) 
        for (uint j = 0; j < ysize; ++j) 
            buffers[SEND][EAST][j] = planes[current].data[IDX(xsize, j+1)];

    /* Phase 3: Start non-blocking halo exchange communications */
    // [B] Non-blocking MPI communication enables computation-communication overlap
    //     Use MPI_Isend/MPI_Irecv to allow interior computation during data transfer

    start_time_comm = MPI_Wtime();

    // For NORTH and SOUTH, use direct pointers to contiguous data (no separate allocation needed)
		if (neighbours[NORTH] != MPI_PROC_NULL) {
			buffers[SEND][NORTH] = &(planes[current].data[xsize + 3]); 		// First effective row
			buffers[RECV][NORTH] = &(planes[current].data[1]);              // Top ghost row
		}
		if (neighbours[SOUTH] != MPI_PROC_NULL) {
			buffers[SEND][SOUTH] = &(planes[current].data[ysize * (xsize + 2) + 1]); // Last effective row
			buffers[RECV][SOUTH] = &(planes[current].data[(ysize + 1) * (xsize + 2) + 1]); // Bottom ghost row
		}

    // East-West communication (using separate buffers for non-contiguous data)
    if (neighbours[EAST] != MPI_PROC_NULL) {
			// Optimization: if neighbor is same rank (wraparound), just copy data
			if (neighbours[EAST] == Rank) {
				for (uint i = 0; i < ysize; i++) {
					buffers[RECV][EAST][i] = buffers[SEND][EAST][i];
				}
			} else {
				MPI_Isend(buffers[SEND][EAST], (int)ysize, MPI_DOUBLE, neighbours[EAST], TAG_E, myCOMM_WORLD, &reqs[nreqs++]);
				MPI_Irecv(buffers[RECV][EAST], (int)ysize, MPI_DOUBLE, neighbours[EAST], TAG_W, myCOMM_WORLD, &reqs[nreqs++]);
			}
		}
    if (neighbours[WEST] != MPI_PROC_NULL) {
			if (neighbours[WEST] == Rank) {
				for (uint i = 0; i < ysize; i++) {
					buffers[RECV][WEST][i] = buffers[SEND][WEST][i];
				}
			} else {
				MPI_Isend(buffers[SEND][WEST], (int)ysize, MPI_DOUBLE, neighbours[WEST], TAG_W, myCOMM_WORLD, &reqs[nreqs++]);
				MPI_Irecv(buffers[RECV][WEST], (int)ysize, MPI_DOUBLE, neighbours[WEST], TAG_E, myCOMM_WORLD, &reqs[nreqs++]);
			}
		}
    
    // North-South communication (using direct pointers to contiguous data)
    if (neighbours[NORTH] != MPI_PROC_NULL) {
			if (neighbours[NORTH] == Rank) {
				for (uint i = 0; i < xsize; i++) {
					buffers[RECV][NORTH][i] = buffers[SEND][NORTH][i];
				}
			} else {
				MPI_Isend(buffers[SEND][NORTH], (int)xsize, MPI_DOUBLE, neighbours[NORTH], TAG_N, myCOMM_WORLD, &reqs[nreqs++]);
				MPI_Irecv(buffers[RECV][NORTH], (int)xsize, MPI_DOUBLE, neighbours[NORTH], TAG_S, myCOMM_WORLD, &reqs[nreqs++]);
			}
		}
		if (neighbours[SOUTH] != MPI_PROC_NULL) {
			if (neighbours[SOUTH] == Rank) {
				for (uint i = 0; i < xsize; i++) {
					buffers[RECV][SOUTH][i] = buffers[SEND][SOUTH][i];
				}
			} else {
				MPI_Isend(buffers[SEND][SOUTH], (int)xsize, MPI_DOUBLE, neighbours[SOUTH], TAG_S, myCOMM_WORLD, &reqs[nreqs++]);
				MPI_Irecv(buffers[RECV][SOUTH], (int)xsize, MPI_DOUBLE, neighbours[SOUTH], TAG_N, myCOMM_WORLD, &reqs[nreqs++]);
			}
		}
    
    // Add communication setup time
		comm_time += MPI_Wtime() - start_time_comm;

    /* Phase 4: Interior update while communication occurs (KEY OPTIMIZATION) */
    // Update interior points (rows 2 to ysize-1, cols 2 to xsize-1) 
    // These points don't depend on ghost cells, so computation can overlap with MPI transfer
		start_time_comp = MPI_Wtime();
    #pragma omp parallel
    {
        update_internal( &planes[current], &planes[!current] );
    }
		comp_time += MPI_Wtime() - start_time_comp;

    /* Phase 5: Wait for communication to complete */
		wait_time = MPI_Wtime();
		MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    wait_time = MPI_Wtime() - wait_time;

    /* Phase 6: Copy received halo data to ghost cells */
    start_time_comp = MPI_Wtime();
    if (neighbours[WEST] != MPI_PROC_NULL)
    for (uint j = 0; j < planes[current].size[_y_]; ++j)
        planes[current].data[IDX(0, j+1)] = buffers[RECV][WEST][j];

    if (neighbours[EAST] != MPI_PROC_NULL)
        for (uint j = 0; j < planes[current].size[_y_]; ++j)
            planes[current].data[IDX(xsize+1, j+1)] = buffers[RECV][EAST][j];

    /* Phase 7: Update border points using received ghost data */
    // Only border points (row/column 1 and row/column max) depend on ghost cells
    #pragma omp parallel
    {
        update_border(periodic, N, &planes[current], &planes[!current]);
    }
    comp_time += MPI_Wtime() - start_time_comp;

    /* Energy conservation check (if enabled) */
    if ( output_energy_stat_perstep )
      output_energy_stat (iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD);

    /* Plane swapping for double buffering - efficient memory reuse */
    current = !current;  
  }
  total_time = MPI_Wtime() - total_time;
  #undef IDX

  output_energy_stat (-1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
  memory_release(planes, buffers);

  /* Collect performance statistics across all MPI processes */
  double max_comp_time, max_comm_time, min_comp_time, sum_comp_time, max_wait_time, max_total_time;
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
    MPI_Reduce(&comp_time, &min_comp_time, 1, MPI_DOUBLE, MPI_MIN, 0, myCOMM_WORLD);
    MPI_Reduce(&comp_time, &sum_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
    MPI_Reduce(&wait_time, &max_wait_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);

    // Clean up the duplicated MPI communicator
    MPI_Comm_free(&myCOMM_WORLD);

  /* Output comprehensive performance statistics including communication overlap efficiency */
  if (Rank == 0) 
  {
        // Calculate load imbalance metrics for MPI processes
        double avg_comp_time = sum_comp_time / Ntasks;
        double load_imbalance = 0.0;
        if (avg_comp_time > 0) {
            load_imbalance = (max_comp_time - min_comp_time) / avg_comp_time;
        }
        
        printf("Total time: %f\n", total_time);
        printf("Initialization time: %f\n", init_time);
        printf("Computation time: %f\n", comp_time);
        printf("Communication time: %f\n", comm_time);
        printf("Communication/Total ratio: %.2f%%\n", (comm_time/total_time)*100.0);
        printf("Computation/Total ratio: %.2f%%\n", (comp_time/total_time)*100.0);
        printf("Wait time for communication: %f\n", wait_time);
        printf("Other time (overhead): %f (%.2f%%)\n", 
            total_time - comp_time - comm_time - wait_time, 
            ((total_time - comp_time - comm_time - wait_time)/total_time)*100.0);
        printf("Max total time: %f\n", max_total_time);
        printf("Max computation time: %f\n", max_comp_time);
        printf("Max communication time: %f\n", max_comm_time);
        printf("Max wait time for communication: %f\n", max_wait_time);
        printf("Load imbalance: %f\n", load_imbalance);
        printf("Load balance efficiency: %f\n", (avg_comp_time/max_comp_time));
        printf("Communication efficiency: %f\n", (max_comp_time/max_total_time));
  }

  MPI_Finalize();
  return 0;
}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */

/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */		      

int initialize (MPI_Comm *Comm,
                int      Me,                  // the rank of the calling process
                int      Ntasks,              // the total number of MPI ranks
                int      argc,                // the argc from command line
                char   **argv,                // the argv from command line
                vec2_t  *S,                   // the size of the plane
                vec2_t  *N,                   // two-uint array defining the MPI tasks' grid
                int     *periodic,            // periodic-boundary tag
                int     *output_energy_stat,  // output energy statistics flag
                int     *neighbours,          // four-int array that gives back the neighbours of the calling task
                int     *Niterations,         // how many iterations
                int     *Nsources,            // how many heat sources
                int     *Nsources_local,      // how many heat sources are local to each MPI task
                vec2_t **Sources_local,       // the heat sources
                double  *energy_per_source,   // how much heat per source
                plane_t *planes,              // the two planes, old and new  
                buffers_t *buffers)           // comunication buffers
{
  int halt = 0;
  int ret = 0;
  int verbose = 0;

  // set default values

  (*S)[_x_]         = 1000;
  (*S)[_y_]         = 1000;
  *periodic         = 0;
  *Nsources         = 5;
  *Nsources_local   = 0;
  *Sources_local    = NULL;
  *Niterations      = 100;
  *energy_per_source = 1.0;

  if ( planes == NULL ) {
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;

  for ( int i = 0; i < 4; i++ )
    neighbours[i] = MPI_PROC_NULL;

  for ( int b = 0; b < 2; b++ )
    for ( int d = 0; d < 4; d++ )
      buffers[b][d] = NULL;

  // process the command line
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":h:x:y:e:E:n:o:p:v:")) != -1)
    {
	    switch( opt )
      {
      case 'x': (*S)[_x_] = (uint)atoi(optarg);
        break;

      case 'y': (*S)[_y_] = (uint)atoi(optarg);
        break;

      case 'e': *Nsources = atoi(optarg);
        break;

      case 'E': *energy_per_source = atof(optarg);
        break;

      case 'n': *Niterations = atoi(optarg);
        break;

      case 'o': *output_energy_stat = (atoi(optarg) > 0);
        break;

      case 'p': *periodic = (atoi(optarg) > 0);
        break;

      case 'v': verbose = atoi(optarg);
        break;

      case 'h': {
        if ( Me == 0 )
          printf( "\nvalid options are ( values btw [] are the default values ):\n"
            "-x    x size of the plate [10000]\n"
            "-y    y size of the plate [10000]\n"
            "-e    how many energy sources on the plate [4]\n"
            "-E    how many energy sources on the plate [1.0]\n"
            "-n    how many iterations [1000]\n"
            "-p    whether periodic boundaries applies  [0 = false]\n\n"
            "-o    whether to print the energy budgets at every step [0 = false]\n"
            "-v    verbosity level [0]\n"
            );
        halt = 1; }
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

  if ( halt )
    return 1;

  /*
  * find a suitable domain decomposition
  * very simple algorithm, you may want to
  * substitute it with a better one
  *
  * the plane Sx x Sy will be solved with a grid
  * of Nx x Ny MPI tasks
  */

  vec2_t Grid;
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
  int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) );

  if ( dimensions == 1 )
  {
    if ( (*S)[_x_] >= (*S)[_y_] )
	    Grid[_x_] = Ntasks, Grid[_y_] = 1;
    else
	    Grid[_x_] = 1, Grid[_y_] = Ntasks;
  }
  else
  {
    int   Nf;
    uint *factors;
    uint  first = 1;
    ret = simple_factorization( Ntasks, &Nf, &factors );
     
    if (ret != 0) {
			printf("Error: factorization failed\n");
			return ret;
		}

    for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
	    first *= factors[i];

		uint factor1 = first;
		uint factor2 = (uint)Ntasks/first;
		uint who_max = (factor1 > factor2);

		if ( (*S)[_x_] >= (*S)[_y_] ) {
			// wide data, make grid wide
			Grid[_x_] = who_max ? factor1 : factor2;
			Grid[_y_] = who_max ? factor2 : factor1;
		} else {
			// tall or square data, make grid tall
			Grid[_y_] = who_max ? factor1 : factor2;
			Grid[_x_] = who_max ? factor2 : factor1;
		}
		
		// Free the factors array allocated by simple_factorization
		if (factors != NULL) {
			free(factors);
		}
  }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];

  // my coordinates in the grid of processors
  uint X = Me % Grid[_x_];
  uint Y = Me / Grid[_x_];

  // find my neighbours
	if ( *periodic ) {
		// Horizontal neighbours
		if (Grid[_x_] > 1 || *periodic) {
			neighbours[EAST]  = (int)(((uint)Y)*Grid[_x_] + (uint)(X + 1 ) % Grid[_x_]);
			neighbours[WEST]  = (int)(((uint)Y)*Grid[_x_] + (uint)(X - 1 + (int)Grid[_x_]) % Grid[_x_]);
		}
		// Vertical neighbours
		if (Grid[_y_] > 1 || *periodic) {
			neighbours[NORTH] = (int)(((uint)(Y - 1 + (int)Grid[_y_]) % Grid[_y_]) * Grid[_x_] + (uint)X);
			neighbours[SOUTH] = (int)(((uint)(Y + 1) % Grid[_y_]) * Grid[_x_] + (uint)X);
		}
	} else {
		// Horizontal neighbours
		if ( Grid[_x_] > 1 ) {  
			neighbours[EAST]  = ( X < (int)Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
			neighbours[WEST]  = ( X > 0 ? Me-1 : MPI_PROC_NULL ); 
		}
		// Vertical neighbours
		if ( Grid[_y_] > 1 ) {
			neighbours[NORTH] = ( Y > 0 ? Me - (int)Grid[_x_]: MPI_PROC_NULL );
			neighbours[SOUTH] = ( Y < (int)Grid[_y_]-1 ? Me + (int)Grid[_x_] : MPI_PROC_NULL );
		}
	}

  // the size of my patch

  /*
  * every MPI task determines the size sx x sy of its own domain
  * REMIND: the computational domain will be embedded into a frame
  *         that is (sx+2) x (sy+2)
  *         the outern frame will be used for halo communication or
  */
  
  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  planes[OLD].size[0] = mysize[0];
  planes[OLD].size[1] = mysize[1];
  planes[NEW].size[0] = mysize[0];
  planes[NEW].size[1] = mysize[1];
  

  if ( verbose > 0 )
  {
    if ( Me == 0 ) {
		  printf("Tasks are decomposed in a grid %d x %d\n\n",
			Grid[_x_], Grid[_y_] );
		  fflush(stdout);
	  }
    MPI_Barrier(*Comm);
    
    if (Me == 0) {
      printf("Neighbours:\n\n");
      printf("   Task   N     E     S     W\n");
      printf("  ============================\n");
      fflush(stdout);
    }

    MPI_Barrier(*Comm);
    for (int t = 0; t < Ntasks; t++) {
      if (t == Me) {
        printf("%5d %5d %5d %5d %5d\n",
          Me,
          neighbours[NORTH],
          neighbours[EAST],
          neighbours[SOUTH],
          neighbours[WEST]
        );
        fflush(stdout);
      }
      MPI_Barrier(*Comm);
    }
    if (Me == 0) {
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(*Comm);
  }

  // allocate the needed memory
  ret = memory_allocate(neighbours, buffers, planes);
	if (ret != 0) {
		printf("Error: failed to allocate memory for the buffers\n");
		return ret;
	}

  // allocate the heat sources
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
	if (ret != 0) {
		printf("Error: failed to initialize the sources\n");
		return ret;
	}

  return ret;
}

uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  uint N = 0;
  uint f = 2;
  uint _A_ = A;

  while ( f < A )
  {
    while( _A_ % f == 0 ) {
      N++;
      _A_ /= f;
    }

    f++;
  }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

  N   = 0;
  f   = 2;
  _A_ = A;

  while ( f < A )
  {
    while( _A_ % f == 0 ) {
      _factors_[N++] = f;
      _A_ /= f;
    }
    f++;
  }

  *factors = _factors_;
  return 0;
}

/**
 * @brief Allocate NUMA-aware memory for simulation planes and MPI communication buffers
 * 
 * Allocates aligned memory optimized for NUMA architectures with proper memory policies
 * for optimal performance on HPC clusters. This version implements advanced memory
 * optimization techniques including aligned allocation and first-touch policy.
 * 
 * **NUMA Optimization Features:**
 * - 64-byte aligned memory allocation for optimal SIMD performance
 * - First-touch memory policy: threads initialize the memory they will use
 * - OpenMP parallel initialization to ensure NUMA-local memory allocation
 * - Optimized buffer layout for communication patterns
 * 
 * **Memory Layout Visualization:**
 * [ X  R  R  R  X ] 
 * [ S  D  D  D  R ]
 * [ S  D  D  D  R ]
 * [ S  D  D  D  R ]
 * [ X  R  R  R  X ]
 * 
 * Where: D = Data; R = Halo for receiving; S = Halo for sending; X = Corner (unused)
 * 
 * @param neighbours Array of neighbor ranks
 * @param buffers_ptr Communication buffers (output)
 * @param planes_ptr Simulation planes (output)
 * @return 0 on success, non-zero on error
 * 
 * @note Uses first-touch NUMA policy for optimal memory locality
 * @note OpenMP parallel initialization ensures threads touch their data
 * @note 64-byte alignment optimizes for modern CPU cache lines
 */
int memory_allocate (const int *neighbours,
		                 buffers_t *buffers_ptr,
		                 plane_t *planes_ptr)
{
  /*
    Advanced memory allocation strategy for optimized MPI+OpenMP hybrid implementation:
    
    (i) --- Simulation data storage with NUMA optimization
    Two memory regions are needed: "OLD" plane containing results from step (i-1)
    and "NEW" plane for updated results from step i. These planes are swapped
    each iteration using double buffering for efficiency.
    
    - Uses 64-byte aligned allocation for optimal SIMD performance
    - Implements first-touch NUMA policy: threads initialize memory they will use
    - OpenMP parallel initialization ensures NUMA-local allocation

    (ii) --- MPI communication buffers
    Separate send and receive buffers for each neighbor (up to 4: N, S, E, W).
    Each buffer holds at most max(mysizex, mysizey) double values for halo exchange.
    
    - NORTH/SOUTH buffers use direct pointers to contiguous data (no allocation)
    - EAST/WEST buffers require separate allocation for non-contiguous data
    
    Buffer organization:
    (*buffers_ptr)[SEND/RECV][{NORTH, SOUTH, EAST, WEST}] = memory regions
  */

  if (planes_ptr == NULL ){
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  if (buffers_ptr == NULL ){
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  // Allocate memory for simulation planes with NUMA-aware optimization
  // Frame includes interior domain plus one ghost cell layer on each side
  uint sx = planes_ptr[OLD].size[_x_];
  uint sy = planes_ptr[OLD].size[_y_];

  unsigned int frame_size = ((sx + 2) * (sy + 2)); // Full size with ghost cells

  // Allocate OLD plane with 64-byte alignment for optimal SIMD performance
  planes_ptr[OLD].data = (double*)aligned_alloc(64, frame_size * sizeof(double));
  if ( planes_ptr[OLD].data == NULL ){
    printf("Error: aligned_alloc failed for OLD plane data\n");
    return 2;
  }
  
  // Initialize OLD plane data with OpenMP for NUMA first-touch policy
  // Each thread initializes the memory it will use, ensuring NUMA-local allocation
	#pragma omp parallel for schedule(static)
	for (uint i = 0; i < frame_size; i++) {
		planes_ptr[OLD].data[i] = 0.0;
	}

  // Allocate NEW plane with 64-byte alignment
  planes_ptr[NEW].data = (double*)aligned_alloc(64, frame_size * sizeof(double));
  if ( planes_ptr[NEW].data == NULL ){
    printf("Error: aligned_alloc failed for NEW plane data\n");
    return 2;
  }
  
  // Initialize NEW plane data with OpenMP for NUMA first-touch policy
  #pragma omp parallel for schedule(static)
	for (unsigned int i = 0; i < frame_size; i++) {
		planes_ptr[NEW].data[i] = 0.0;
	}

  // Set plane dimensions
  planes_ptr[OLD].size[_x_] = sx;
	planes_ptr[OLD].size[_y_] = sy;
	planes_ptr[NEW].size[_x_] = sx;
	planes_ptr[NEW].size[_y_] = sy;

  // Allocate MPI communication buffers with NUMA optimization
  // Note: NORTH and SOUTH communication uses direct pointers to contiguous
  // data rows, so no separate allocation is needed. Only EAST and WEST
  // require separate buffers due to non-contiguous data layout.

	// Allocate EAST communication buffers (if neighbor exists)
	if (neighbours[EAST] != MPI_PROC_NULL) {
		buffers_ptr[SEND][EAST] = (double*)aligned_alloc(64, sy * sizeof(double));
		buffers_ptr[RECV][EAST] = (double*)aligned_alloc(64, sy * sizeof(double));
		if (buffers_ptr[SEND][EAST] == NULL || buffers_ptr[RECV][EAST] == NULL) {
			printf("Error: failed to allocate memory for the EAST buffers\n");
			return 1;
		}
	}
	// Allocate WEST communication buffers (if neighbor exists)
	if (neighbours[WEST] != MPI_PROC_NULL) {
		buffers_ptr[SEND][WEST] = (double*)aligned_alloc(64, sy * sizeof(double));
		buffers_ptr[RECV][WEST] = (double*)aligned_alloc(64, sy * sizeof(double));
		if (buffers_ptr[SEND][WEST] == NULL || buffers_ptr[RECV][WEST] == NULL) {
			printf("Error: failed to allocate memory for the WEST buffers\n");
			return 1;
		}
	}

  return 0;
}

int initialize_sources(int Me,
                       int Ntasks,
                       MPI_Comm *Comm,
                       vec2_t mysize,
                       int Nsources,
                       int *Nsources_local,
                       vec2_t **Sources)
{
  srand48(0 + Me);
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
  
  if ( Me == 0 )
  {
    for ( int i = 0; i < Nsources; i++ )
  	  tasks_with_sources[i] = (int)lrand48() % Ntasks;
  }
  
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 )
  {
    vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
    for ( int s = 0; s < nlocal; s++ )
	  {
	    helper[s][_x_] = 1 + lrand48() % mysize[_x_];
	    helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	  }
    *Sources = helper;
  }
  
  free(tasks_with_sources);
  return 0;
}

int memory_release (plane_t *planes,
		                buffers_t *buffers)
{
  if ( planes != NULL )
  {
    if ( planes[OLD].data != NULL )
	    free (planes[OLD].data);
      
    if ( planes[NEW].data != NULL )
	    free (planes[NEW].data);
  }

  if ( buffers != NULL )
  {
    for ( int b = 0; b < 2; b++ )
      for ( int d = 2; d < 4; d++ )
      {
        if ( buffers[b][d] != NULL )
          free( buffers[b][d] );
      }
  }
      
  return 0;
}

int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{

  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy );
  
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
  
  if ( Me == 0 )
  {
    if ( step >= 0 ){
	    printf(" [ step %4d ] ", step ); 
      fflush(stdout);
    }

    printf( "total injected energy is %g, "
            "system energy is %g "
            "( in avg %g per grid point)\n",
            budget,
            tot_system_energy,
            tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
  }
  
  return 0;
}