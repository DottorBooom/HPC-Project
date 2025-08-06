#include "stencil_template_parallel.h"

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
  
  int output_energy_stat_perstep;
  
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
  double t1 = MPI_Wtime();   /* take wall-clock time */
  
  #define IDX(i, j) ((j) * (planes[current].size[_x_] + 2) + (i) - 2)
  for (int iter = 0; iter < Niterations; ++iter)
  {
    MPI_Request reqs[8];
    for (int i = 0; i < 8; ++i)
      reqs[i] = MPI_REQUEST_NULL;
    
    /* new energy from sources */
    inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);

    // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position
    if (neighbours[WEST] != MPI_PROC_NULL) 
        for (uint j = 0; j < planes[current].size[_y_]; ++j) 
            buffers[SEND][WEST][j] = planes[current].data[IDX(1, j+1)];
    
    if (neighbours[EAST] != MPI_PROC_NULL) 
        for (uint j = 0; j < planes[current].size[_y_]; ++j) 
            buffers[SEND][EAST][j] = planes[current].data[IDX(planes[current].size[_x_], j+1)];

    // [B] perfoem the halo communications
    //     (1) use Send / Recv
    //     (2) use Isend / Irecv
    //         --> can you overlap communication and compution in this way?

    for(int dir = 0; dir < 4; dir++){
      if(neighbours[dir] != MPI_PROC_NULL){
        int count = (dir == NORTH || dir == SOUTH) ? planes[current].size[_x_] : planes[current].size[_y_];

        if (buffers[SEND][dir] != NULL && buffers[RECV][dir] != NULL) {
          MPI_Isend(buffers[SEND][dir], count, MPI_DOUBLE, neighbours[dir], 0, myCOMM_WORLD, &reqs[dir*2]);
          MPI_Irecv(buffers[RECV][dir], count, MPI_DOUBLE, neighbours[dir], 0, myCOMM_WORLD, &reqs[dir*2+1]);
        }else{
          printf("Buffers for direction %d are NULL, skipping communication (Rank %d / %d)\n", dir, Rank, Ntasks);
          fflush(stdout);
          MPI_Barrier(myCOMM_WORLD);
        }
      }
    }
    
    MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

    // [C] copy the haloes data

    if (neighbours[WEST] != MPI_PROC_NULL)
    for (uint j = 0; j < planes[current].size[_y_]; ++j)
        planes[current].data[IDX(0, j+1)] = buffers[RECV][WEST][j];

    if (neighbours[EAST] != MPI_PROC_NULL)
        for (uint j = 0; j < planes[current].size[_y_]; ++j)
            planes[current].data[IDX(planes[current].size[_x_]+1, j+1)] = buffers[RECV][EAST][j];

    /* --------------------------------------  */
    /* update grid points */

    update_plane(periodic, N, &planes[current], &planes[!current]);

    /* output if needed */
    if ( output_energy_stat_perstep )
      output_energy_stat (iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD);

    /* swap plane indexes for the new iteration */
    current = !current;  
  }
  t1 = MPI_Wtime() - t1;
  #undef IDX

  output_energy_stat ( -1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
  
  memory_release(planes, buffers);
  
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
  *Nsources         = 1;
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
      
    for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
	    first *= factors[i];

    if ( (*S)[_x_] > (*S)[_y_] )
	    Grid[_x_] = Ntasks/first, Grid[_y_] = first;
    else
	    Grid[_x_] = first, Grid[_y_] = Ntasks/first;
  }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];

  // my coordinates in the grid of processors
  uint X = Me % Grid[_x_];
  uint Y = Me / Grid[_x_];

  // find my neighbours
  if ( Grid[_x_] > 1 )
  {  
    if ( *periodic ) {       
	    neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
	    neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); 
    }else {
	    neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
	    neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL ); 
    }  
  }

  if ( Grid[_y_] > 1 )
  {
    if ( *periodic ) {      
	    neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
	    neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks; 
    } else {    
	    neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
	    neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL ); 
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
      
    for ( int t = 0; t < Ntasks; t++ )
	  {
      if ( t == Me )
      {
        printf("Task %4d :: "
        "\tgrid coordinates : %3d, %3d\n"
        "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
        Me, X, Y,  // ··································································
        neighbours[NORTH], neighbours[EAST],
        neighbours[SOUTH], neighbours[WEST] );
        fflush(stdout);
      }
	    MPI_Barrier(*Comm);
	  }    
  }

  // allocate the needed memory
  ret = memory_allocate(neighbours, buffers, planes);
  if ( ret != 0 ) return ret;

  // allocate the heat sources
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
  if ( ret != 0 ) return ret;

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

int memory_allocate (const int *neighbours,
		                 buffers_t *buffers_ptr,
		                 plane_t *planes_ptr)
{
  /*
    here you allocate the memory buffers that you need to
    (i)  hold the results of your computation
    (ii) communicate with your neighbours

    The memory layout that I propose to you is as follows:

    (i) --- calculations
    you need 2 memory regions: the "OLD" one that contains
    results for the step (i-1)th, and the "NEW" one that will contain
    the updated results from the step ith.

    Then, the "NEW" will be treated as "OLD" and viceversa.

    These two memory regions are indexed by *plate_ptr:

    planew_ptr[0] ==> the "OLD" region
    plamew_ptr[1] ==> the "NEW" region


    (ii) --- communications

    you may need two buffers (one for sending and one for receiving)
    for each one of your neighnours, that are at most 4:
    north, south, east amd west.      

    To them you need to communicate at most mysizex or mysizey
    daouble data.

    These buffers are indexed by the buffer_ptr pointer so
    that

    (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
    (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions
    
    --->> Of course you can change this layout as you prefer
  */

  if (planes_ptr == NULL ){
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  if (buffers_ptr == NULL ){
    printf("Error: invalid pointer passed to memory_allocate\n");
    return 1;
  }

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks

  /*
  
  [ X  R  R  R  X ] 
  [ S  D  D  D  R ]
  [ S  D  D  D  R ]
  [ S  D  D  D  R ]
  [ x  R  R  R  X ]

  D = Data; R = Halo for reciving; x = Contour frame, never used
  */
 
  uint sx = planes_ptr[OLD].size[_x_];
  uint sy = planes_ptr[OLD].size[_y_];

  unsigned int frame_size = ((sx + 2) * (sy + 2) - 4); // -4 to exclude the corners

  planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[OLD].data == NULL ){
    printf("Error: malloc failed for OLD plane data\n");
    return 2;
  }
  memset ( planes_ptr[OLD].data, 0, frame_size * sizeof(double) );

  planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[NEW].data == NULL ){
    printf("Error: malloc failed for NEW plane data\n");
    return 2;
  }
  memset ( planes_ptr[NEW].data, 0, frame_size * sizeof(double) );

  // ··················································
  // buffers for north and south communication 
  // are not really needed
  //
  // in fact, they are already contiguous, just the
  // first and last line of every rank's plane
  //
  // you may just make some pointers pointing to the
  // correct positions

  // or, if you prefer, just go on and allocate buffers
  // also for north and south communications

  // ··················································
  // allocate buffers
  //

  for(int dir = 0; dir < 4; dir++){
      if (neighbours[dir] != MPI_PROC_NULL){
          if (dir == NORTH) {
              // Point to the first inner row (excluding the halo)
              buffers_ptr[SEND][NORTH] = &(planes_ptr[OLD].data[sx + 1]);
              buffers_ptr[RECV][NORTH] = &(planes_ptr[OLD].data[0]);
          } else if (dir == SOUTH) {
              // Point to the last inner row (excluding the halo)
              buffers_ptr[SEND][SOUTH] = &(planes_ptr[OLD].data[frame_size - 2*sx - 1]);
              buffers_ptr[RECV][SOUTH] = &(planes_ptr[OLD].data[frame_size - sx]);
          } else if (dir == EAST || dir == WEST) {
              // For columns, you still need to allocate buffers
              buffers_ptr[SEND][dir] = (double*)malloc(sy * sizeof(double));
              if (buffers_ptr[SEND][dir] == NULL) {
                  printf("Error: malloc failed for SEND buffer (dir %d)\n", dir);
                  return 3;
              }
              memset(buffers_ptr[SEND][dir], 0, sy * sizeof(double));

              buffers_ptr[RECV][dir] = (double*)malloc(sy * sizeof(double));
              if (buffers_ptr[RECV][dir] == NULL) {
                  printf("Error: malloc failed for RECV buffer (dir %d)\n", dir);
                  return 3;
              }
              memset(buffers_ptr[RECV][dir], 0, sy * sizeof(double));
          }
      } else {
          buffers_ptr[SEND][dir] = NULL;
          buffers_ptr[RECV][dir] = NULL;
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
  srand48(time(NULL) ^ Me);
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