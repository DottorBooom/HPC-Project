/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
*/

#include "stencil_template_serial.h"

int main(int argc, char **argv)
{

  int  Niterations; // how many iterations
  int  periodic; // whether periodic boundaries apply
  unsigned int S[2]; // size of the plate

  int     Nsources; // how many heat sources
  int    *Sources; // the heat sources
  double  energy_per_source; // how much energy per source

  double *planes[2]; // the two planes, old and new
  
  double injected_heat = 0; // how much energy has been injected in the system

  int injection_frequency; // how often to inject energy
  int output_energy_at_steps = 0; // whether to print the energy budget at every step
   
  int ret; 

  /* argument checking and setting */
  ret = initialize ( argc, argv, &S[0], &periodic, &Niterations,
	       &Nsources, &Sources, &energy_per_source, &planes[0],
	       &output_energy_at_steps, &injection_frequency );

  if (ret == 1)
  {
    printf("Error during memory allocation\n");
    return 1;
  }
  int current = OLD;
  double time_spent_injecting [Niterations];
  double time_spent_updating [Niterations];
  double time_spent_simulation [Niterations];

  if ( injection_frequency > 1 )
    inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current], &time_spent_injecting[0] );

  for (int iter = 0; iter < Niterations; iter++)
  {

    // Start timing
    double start = omp_get_wtime();

    /* new energy from sources */
    if ( iter % injection_frequency == 0 )
    {
      inject_energy( periodic, Nsources, Sources, energy_per_source, S, planes[current], &time_spent_injecting[iter] );
      injected_heat += Nsources*energy_per_source;
    }
                  
    /* update grid points */
    update_plane(periodic, S, planes[current], planes[!current], &time_spent_updating[iter] );

    if ( output_energy_at_steps )
    {
      double system_heat;
      get_total_energy( S, planes[!current], &system_heat);
              
      printf("step %d :: injected energy is %g, updated system energy is %g\n", 
              iter, injected_heat, system_heat );

      char filename[100];
      sprintf( filename, "plane_%05d.bin", iter );
      //dump( planes[!current], S, filename, NULL, NULL );        
    }

    /* swap planes for the new iteration */
      current = !current;

    // End timing
    time_spent_simulation[iter] = omp_get_wtime() - start;
  }

  double sumI = 0;
  double sumU = 0;
  double sumS = 0;
  #pragma omp parallel for reduction(+:sumI, sumU, sumS) schedule(static)
  for(int i = 0; i < Niterations; i++)
  {
    sumI += time_spent_injecting[i];
    sumU += time_spent_updating[i];
    sumS += time_spent_simulation[i];
  }
  printf("Average time spent injecting energy: %f ms\n", (sumI/Niterations)*1000);
  printf("Average time spent updating: %f ms\n", (sumU/Niterations)*1000);
  printf("Total time spent in the simulation: %f s\n", sumS);

  /* get final heat in the system */
  double system_heat;
  get_total_energy( S, planes[current], &system_heat);

  printf("injected energy is %g, system energy is %g\n", injected_heat, system_heat );

  memory_release( planes[OLD], Sources );

  export_and_plot(time_spent_injecting, time_spent_updating, time_spent_simulation, Niterations);

  return 0;
}

/* ==========================================================================
   =   routines called within the integration loop                          =
   ========================================================================== */



/* ==========================================================================
   =   initialization                                                       =
   ========================================================================== */

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
  
  // set default values

  S[_x_]            = 1000;
  S[_y_]            = 1000;
  *periodic         = 0;
  *Nsources         = 1;
  *Niterations      = 99;
  *output_energy_at_steps = 0;
  *energy_per_source = 1.0;
  *injection_frequency = *Niterations;

  double freq = 0;

  // process the command line

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

  if ( freq == 0 )
    *injection_frequency = 1;
  else
  {
    freq = (freq > 1.0 ? 1.0 : freq );
    *injection_frequency = freq * *Niterations;
  }

  // allocate the needed memory
  ret = memory_allocate( S, planes ); 
  if ( ret != 0 )
    return ret;

  // allocate the heat sources
  ret = initialize_sources( S, *Nsources, Sources );
  if ( ret != 0 )
    return ret;

  return 0;
}


int memory_allocate ( unsigned int size[2],
		                  double **planes_ptr )
/*
 * allocate the memory for the planes
 * we need 2 planes: the first contains the
 * current data, the second the updated data
 *
 * in the integration loop then the roles are
 * swapped at every iteration
 *
 */
{
  if (planes_ptr == NULL )
    // an invalid pointer has been passed
    // manage the situation
    return 1;

  unsigned int bytes = (size[_x_]+2)*(size[_y_]+2);

  planes_ptr[OLD] = (double*)malloc( 2*bytes*sizeof(double) );
  memset ( planes_ptr[OLD], 0, 2*bytes*sizeof(double) );
  planes_ptr[NEW] = planes_ptr[OLD] + bytes;
      
  return 0;
}


int initialize_sources( unsigned int size[2],
			                  int Nsources,
			                  int **Sources)
/*
 * randomly spread heat suources
 *
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
    fprintf(fp, "iteration,injecting,updating,simulation,total\n");

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

